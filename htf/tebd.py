"""Time-Evolving Block Decimation (TEBD) and single-site DMRG sweep.

TEBD evolves an MPS under a nearest-neighbour Hamiltonian by alternating
even and odd bond 2-site gates (Trotter decomposition).

Honest scope [工程]
-------------------
* Trotter error is O(dt²) per step for 1st-order and O(dt³) for 2nd-order
  (Strang splitting).  The Trotter error is NOT certified — use
  temple_lanczos / variational_bound for certified energy bounds.
* Works only for nearest-neighbour Hamiltonians.
* GPU / JAX acceleration requires the optional 'accel' extra.
* DMRG (dmrg_sweep) is a single-site variational algorithm; it minimises
  energy within the MPS manifold for the given chi, not globally.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import scipy.linalg

from .mps import (
    MPS,
    _left_canonicalise,
    mps_apply_gate,
    mps_inner,
    mps_normalise,
    mps_to_state,
)

# ── Hamiltonian helpers ────────────────────────────────────────────────────

def nn_hamiltonian(
    h_terms: list[np.ndarray],
    n: int,
    d: int = 2,
    periodic: bool = False,
) -> np.ndarray:
    """Build full d^n × d^n Hamiltonian from nearest-neighbour bond matrices.

    Parameters
    ----------
    h_terms:  list of (d², d²) bond Hamiltonians for bonds 0-1, 1-2, …
              When ``periodic=True`` the list must have length ``n`` — the
              last entry is the wrap-around bond (n-1) → 0.
    n:        number of sites.
    d:        physical dimension per site.
    periodic: add a periodic wrap-around coupling (site n-1 ↔ site 0)
              using ``h_terms[-1]``.

    Returns
    -------
    Full Hamiltonian matrix of shape (d^n, d^n).
    """
    dtype = complex if any(np.iscomplexobj(h) for h in h_terms) else float
    H = np.zeros((d**n, d**n), dtype=dtype)

    obc_terms = h_terms[:-1] if periodic else h_terms
    for bond, h in enumerate(obc_terms):
        left_id  = np.eye(d**bond, dtype=dtype)
        right_id = np.eye(d**(n - bond - 2), dtype=dtype)
        H += np.kron(np.kron(left_id, h), right_id)

    if periodic:
        # Wrap-around bond acts on (site n-1, site 0).
        # h_pbc convention: h_pbc[sn1'*d+s0', sn1*d+s0]
        # In big-endian state vectors site 0 is the most-significant index.
        # H_pbc[s0',mid',sn1', s0,mid,sn1]
        #   = h4_T[s0',s0, sn1',sn1] * I_mid[mid',mid]
        h_pbc = np.asarray(h_terms[-1], dtype=dtype)
        h4    = h_pbc.reshape(d, d, d, d)       # [sn1', s0', sn1, s0]
        h4_T  = h4.transpose(1, 3, 0, 2)        # [s0', s0, sn1', sn1]
        I_mid = np.eye(d**(n - 2), dtype=dtype)
        H += np.einsum('abcd,ef->aecbfd', h4_T, I_mid, optimize=True).reshape(d**n, d**n)

    return H


def tfim_bonds(
    n: int,
    J: float = 1.0,
    h: float = 0.5,
    periodic: bool = False,
) -> list[np.ndarray]:
    """Bond Hamiltonians for the transverse-field Ising model.

    H = -J Σ Z_i Z_{i+1} - h Σ X_i

    Each site-i term (-h X_i) is split evenly across the bonds that touch it.
    With ``periodic=False`` boundary sites (touching one bond) receive full
    weight ``h``; interior sites (touching two bonds) receive ``h/2`` each.
    With ``periodic=True`` all sites touch exactly two bonds, so every bond
    contributes ``h/2`` per endpoint; an extra wrap-around bond is appended.
    """
    Z = np.array([[1, 0], [0, -1]], dtype=float)
    X = np.array([[0, 1], [1, 0]], dtype=float)
    I = np.eye(2, dtype=float)
    bonds = []
    for i in range(n - 1):
        if periodic:
            x_left = x_right = h / 2
        else:
            x_left  = h if i == 0     else h / 2
            x_right = h if i == n - 2 else h / 2
        bonds.append(-J * np.kron(Z, Z) - x_left * np.kron(X, I) - x_right * np.kron(I, X))
    if periodic:
        bonds.append(-J * np.kron(Z, Z) - h / 2 * np.kron(X, I) - h / 2 * np.kron(I, X))
    return bonds


def xx_bonds(
    n: int,
    J: float = 1.0,
    h: float = 0.5,
    periodic: bool = False,
) -> list[np.ndarray]:
    """Bond Hamiltonians for the XX + transverse-field model.

    H = -J Σ (X_i X_{i+1} + Y_i Y_{i+1}) - h Σ Z_i
    """
    X = np.array([[0, 1], [1, 0]], dtype=float)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=float)
    I = np.eye(2, dtype=float)
    bonds = []
    for i in range(n - 1):
        if periodic:
            z_left = z_right = h / 2
        else:
            z_left  = h if i == 0     else h / 2
            z_right = h if i == n - 2 else h / 2
        b = (-J * (np.kron(X, X) + np.kron(Y, Y))
             - z_left  * np.kron(Z, I)
             - z_right * np.kron(I, Z))
        bonds.append(b)
    if periodic:
        b = (-J * (np.kron(X, X) + np.kron(Y, Y))
             - h / 2 * np.kron(Z, I)
             - h / 2 * np.kron(I, Z))
        bonds.append(b)
    return bonds


def heisenberg_bonds(
    n: int,
    J: float = 1.0,
    h: float = 0.0,
    periodic: bool = False,
) -> list[np.ndarray]:
    """Bond Hamiltonians for the Heisenberg XXX model.

    H = -J Σ (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}) - h Σ Z_i

    The on-site longitudinal field (-h Z_i) is distributed identically to
    ``tfim_bonds``/``xx_bonds``: boundary sites get full weight ``h``,
    interior sites get ``h/2`` per touching bond; PBC makes all sites
    interior (``h/2`` each, plus a wrap-around bond).
    """
    X = np.array([[0, 1], [1, 0]], dtype=float)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=float)
    I = np.eye(2, dtype=float)
    H_exch = -J * (np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z))
    bonds = []
    for i in range(n - 1):
        if periodic:
            z_left = z_right = h / 2
        else:
            z_left  = h if i == 0     else h / 2
            z_right = h if i == n - 2 else h / 2
        bonds.append(H_exch - z_left * np.kron(Z, I) - z_right * np.kron(I, Z))
    if periodic:
        bonds.append(H_exch - h / 2 * np.kron(Z, I) - h / 2 * np.kron(I, Z))
    return bonds


def bose_hubbard_bonds(
    n: int,
    t: float = 1.0,
    U: float = 4.0,
    mu: float = 2.0,
    max_occ: int = 3,
) -> list[np.ndarray]:
    """Bond Hamiltonians for the 1-D Bose-Hubbard model (open boundaries).

    H = -t Σ (a†_i a_{i+1} + h.c.) + U/2 Σ n_i(n_i-1) - μ Σ n_i

    Physical dimension: ``d = max_occ + 1`` (Fock states 0 … max_occ).
    On-site terms are distributed across bonds identically to ``tfim_bonds``.
    """
    d = max_occ + 1
    a   = np.diag([np.sqrt(k) for k in range(1, d)], 1).astype(float)
    adag = a.T
    num  = np.diag(np.arange(d, dtype=float))
    I    = np.eye(d, dtype=float)

    onsite = U / 2 * num @ (num - I) - mu * num
    hop    = -t * (np.kron(adag, a) + np.kron(a, adag))

    bonds = []
    for i in range(n - 1):
        w_left  = 1.0 if i == 0     else 0.5
        w_right = 1.0 if i == n - 2 else 0.5
        bonds.append(hop + w_left * np.kron(onsite, I) + w_right * np.kron(I, onsite))
    return bonds


@dataclass
class TEBDResult:
    """Result of a TEBD time evolution run."""
    mps_final: MPS
    times: list[float]
    energies: list[float]
    max_bonds: list[int]
    total_discarded: float
    dt: float
    n_steps: int
    trotter_order: int


def _bond_gate(h: np.ndarray, dt: float, imaginary: bool = False) -> np.ndarray:
    """Compute 2-site gate exp(-i dt h) or exp(-dt h) for one bond.

    Parameters
    ----------
    h:         local Hamiltonian matrix of shape (d², d²).
    dt:        time step.
    imaginary: True for imaginary-time evolution.

    Returns
    -------
    gate of shape (d, d, d, d) with indices (s0', s1', s0, s1).
    """
    d = round(h.shape[0] ** 0.5)
    if imaginary:
        G = scipy.linalg.expm(-dt * h)
    else:
        G = scipy.linalg.expm(-1j * dt * h)
    # G[row, col] = G[(s0'*d + s1'), (s0*d + s1)]
    # Reshape: G[s0', s1', s0, s1]
    return G.reshape(d, d, d, d)


def tebd_step(
    mps: MPS,
    h_terms: list[np.ndarray],
    dt: float,
    chi: int | None = None,
    imaginary: bool = False,
    trotter_order: int = 1,
    n_threads: int | None = None,
) -> tuple[MPS, float]:
    """One TEBD Trotter step.

    Parameters
    ----------
    mps:          input MPS.
    h_terms:      list of (d², d²) local Hamiltonian matrices for bonds
                  0-1, 1-2, …, (n-2)-(n-1).
    dt:           time step.
    chi:          bond dimension truncation; None = no truncation.
    imaginary:    True for imaginary-time evolution.
    trotter_order: 1 or 2 (Strang splitting).
    n_threads:    Thread count for intra-step bond parallelism.  Only
                  effective when *trotter_order=2*: the even and odd bond
                  groups each contain non-overlapping site pairs that can
                  be applied concurrently.  ``None`` or ``1`` = sequential.

    Returns
    -------
    (new_mps, total_discarded_weight)
    """
    n = mps.n_sites
    if len(h_terms) != n - 1:
        raise ValueError(
            f"Expected {n-1} Hamiltonian terms for {n} sites, got {len(h_terms)}"
        )
    if trotter_order not in (1, 2):
        raise ValueError("trotter_order must be 1 or 2")

    if trotter_order == 1:
        gates = [_bond_gate(h, dt, imaginary) for h in h_terms]
        return _apply_all_bonds(mps, gates, chi)

    # 2nd-order Strang splitting: even(dt/2) · odd(dt) · even(dt/2)
    gates_half = [_bond_gate(h, dt / 2, imaginary) for h in h_terms]
    gates_full = [_bond_gate(h, dt,     imaginary) for h in h_terms]
    mps, d1 = _apply_bond_parity(mps, gates_half, chi, even=True,  n_threads=n_threads)
    mps, d2 = _apply_bond_parity(mps, gates_full, chi, even=False, n_threads=n_threads)
    mps, d3 = _apply_bond_parity(mps, gates_half, chi, even=True,  n_threads=n_threads)
    return mps, d1 + d2 + d3


def _apply_all_bonds(
    mps: MPS,
    gates: list[np.ndarray],
    chi: int | None,
) -> tuple[MPS, float]:
    """Apply all bond gates sequentially (1st-order)."""
    total_disc = 0.0
    for i, gate in enumerate(gates):
        mps, disc = mps_apply_gate(mps, gate, [i, i + 1], chi=chi)
        total_disc += disc
    return mps, total_disc


def _apply_bond_tensors(
    A_l: np.ndarray,
    A_r: np.ndarray,
    gate: np.ndarray,
    chi: int | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply a 2-site gate to (A_l, A_r); return (new_Al, new_Ar, discarded).

    Pure function with no MPS state — safe to call from multiple threads
    simultaneously as long as each call uses distinct input arrays.
    """
    chi_l, d, _   = A_l.shape
    _, _, chi_r   = A_r.shape
    AB   = np.einsum("asb,btc->astc", A_l, A_r)
    AB_g = np.einsum("stpq,apqc->astc", gate, AB)
    M    = AB_g.reshape(chi_l * d, d * chi_r)
    scale = float(np.max(np.abs(M)))
    if scale > 0:
        M = M / scale
    U, s, Vh = np.linalg.svd(M, full_matrices=False)
    s        = s * scale
    keep     = min(chi, len(s)) if chi is not None else len(s)
    discarded = float(np.sum(s[keep:] ** 2))
    new_Al   = U[:, :keep].reshape(chi_l, d, keep)
    new_Ar   = (np.diag(s[:keep]) @ Vh[:keep, :]).reshape(keep, d, chi_r)
    return new_Al, new_Ar, discarded


def _apply_bond_parity(
    mps: MPS,
    gates: list[np.ndarray],
    chi: int | None,
    even: bool,
    n_threads: int | None = None,
) -> tuple[MPS, float]:
    """Apply even (0-1, 2-3, …) or odd (1-2, 3-4, …) bond gates.

    When *n_threads* > 1 the bonds within the parity group are applied in
    parallel via ``ThreadPoolExecutor``.  This is safe because bonds in the
    same parity group act on non-overlapping site pairs; each worker reads
    the *input* tensors (read-only) and returns new tensors that are written
    back to the MPS only after all threads complete.  NumPy's SVD and einsum
    release the GIL, so threads run concurrently on multi-core CPUs.
    """
    start        = 0 if even else 1
    bond_indices = list(range(start, len(gates), 2))

    in_t = list(mps.tensors)

    if n_threads == 1 or len(bond_indices) <= 1 or n_threads is None:
        total_disc = 0.0
        for i in bond_indices:
            new_Al, new_Ar, disc = _apply_bond_tensors(in_t[i], in_t[i + 1], gates[i], chi)
            in_t[i]     = new_Al
            in_t[i + 1] = new_Ar
            total_disc  += disc
        return MPS(in_t), total_disc

    # Parallel path — workers read from the snapshot, results collected after all finish.
    # Limit BLAS to 1 thread per worker so concurrent SVD calls are deterministic.
    try:
        from threadpoolctl import threadpool_limits as _tpl
        def _worker(i: int):
            with _tpl(limits=1, user_api="blas"):
                return i, *_apply_bond_tensors(in_t[i], in_t[i + 1], gates[i], chi)
    except ImportError:
        def _worker(i: int):  # type: ignore[misc]
            return i, *_apply_bond_tensors(in_t[i], in_t[i + 1], gates[i], chi)

    effective = min(n_threads, len(bond_indices))
    with ThreadPoolExecutor(max_workers=effective) as pool:
        results = list(pool.map(_worker, bond_indices))

    new_tensors = list(in_t)   # shallow copy of list; elements replaced below
    total_disc  = 0.0
    for i, new_Al, new_Ar, disc in results:
        new_tensors[i]     = new_Al
        new_tensors[i + 1] = new_Ar
        total_disc         += disc
    return MPS(new_tensors), total_disc


def tebd_evolve(
    mps: MPS,
    h_terms: list[np.ndarray],
    dt: float,
    n_steps: int,
    chi: int | None = None,
    imaginary: bool = False,
    trotter_order: int = 1,
    measure_every: int = 1,
    n_threads: int | None = None,
) -> TEBDResult:
    """Full TEBD time evolution.

    Parameters
    ----------
    mps:           initial MPS.
    h_terms:       list of (d², d²) Hamiltonian matrices for each bond.
    dt:            time step.
    n_steps:       number of Trotter steps.
    chi:           maximum bond dimension; None = no truncation.
    imaginary:     True for imaginary-time evolution (ground-state search).
    trotter_order: 1 or 2.
    measure_every: record energy/bond every this many steps.
    n_threads:     Thread count for intra-step bond parallelism; passed to
                   :func:`tebd_step`.  Only effective with *trotter_order=2*.

    Returns
    -------
    TEBDResult with trajectory data.
    """
    times, energies, max_bonds = [], [], []
    total_disc = 0.0
    t = 0.0

    def _record() -> None:
        E = _nn_energy(mps, h_terms)
        times.append(t)
        energies.append(E)
        max_bonds.append(mps.max_bond)

    _record()

    for step in range(n_steps):
        mps, disc = tebd_step(
            mps, h_terms, dt, chi=chi,
            imaginary=imaginary, trotter_order=trotter_order,
            n_threads=n_threads,
        )
        total_disc += disc
        t += dt

        if imaginary:
            nrm = abs(mps_inner(mps, mps)) ** 0.5
            if nrm > 0:
                mps = mps_normalise(mps)

        if (step + 1) % measure_every == 0:
            _record()

    return TEBDResult(
        mps_final=mps,
        times=times,
        energies=energies,
        max_bonds=max_bonds,
        total_discarded=total_disc,
        dt=dt,
        n_steps=n_steps,
        trotter_order=trotter_order,
    )


def _nn_energy(mps: MPS, h_terms: list[np.ndarray]) -> float:
    """⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩ via dense state vector (correct for small n)."""
    psi   = mps_to_state(mps)
    norm2 = float(np.dot(psi.conj(), psi).real)
    if norm2 == 0:
        return 0.0
    H_full = nn_hamiltonian(h_terms, mps.n_sites, mps.phys_dim)
    return float(np.dot(psi.conj(), H_full @ psi).real) / norm2


# ── single-site DMRG sweep ──────────────────────────────────────────────────

@dataclass
class DMRGResult:
    """Result of a single-site DMRG variational ground-state optimisation."""
    mps_final: MPS
    energies: list[float]
    n_sweeps: int
    converged: bool


def dmrg_sweep(
    mps: MPS,
    h_terms: list[np.ndarray],
    n_sweeps: int = 10,
    chi: int | None = None,
    tol: float = 1e-8,
) -> DMRGResult:
    """Single-site DMRG variational ground-state search.

    Alternates left→right and right→left sweeps, optimising one site at a
    time by solving the local effective eigenvalue problem.

    This implementation builds the effective Hamiltonian via dense projections
    and is exact for any bond dimension — suitable for small systems (n ≤ 8).
    For large systems with chi << 2^(n/2) it still gives the correct MPS-
    manifold minimum but does not exploit block sparsity.

    Honest scope: converges to a local minimum of the energy on the MPS
    manifold for the given chi.  Use temple_lanczos for a certified lower bound.

    Parameters
    ----------
    mps:      initial MPS (left-canonicalised internally).
    h_terms:  list of (d², d²) nearest-neighbour Hamiltonian matrices.
    n_sweeps: maximum number of full sweeps (L→R + R→L).
    chi:      bond dimension cap; None keeps the current bond dimension.
    tol:      energy convergence tolerance per sweep.
    """
    mps = _left_canonicalise(mps.copy())
    n   = mps.n_sites
    d   = mps.phys_dim
    H_full   = nn_hamiltonian(h_terms, n, d)
    energies: list[float] = []
    converged = False

    for sweep_idx in range(n_sweeps):
        E_before = energies[-1] if energies else None

        # ── L→R half-sweep ────────────────────────────────────────────
        for i in range(n - 1):
            H_eff = _heff_dense(mps, H_full, i)
            mps.tensors[i], E = _local_eig(H_eff, mps.tensors[i])
            energies.append(E)
            # Left-canonicalise site i before moving right
            A = mps.tensors[i]
            chi_l, d_i, chi_r = A.shape
            Q, R_mat = scipy.linalg.qr(A.reshape(chi_l * d_i, chi_r),
                                       mode="economic")
            mps.tensors[i]     = Q.reshape(chi_l, d_i, Q.shape[1])
            mps.tensors[i + 1] = np.einsum("ab, bsc -> asc",
                                           R_mat, mps.tensors[i + 1])

        # Last site (rightmost)
        H_eff = _heff_dense(mps, H_full, n - 1)
        mps.tensors[n - 1], E = _local_eig(H_eff, mps.tensors[n - 1])
        energies.append(E)

        # ── R→L half-sweep ────────────────────────────────────────────
        for i in range(n - 2, -1, -1):
            H_eff = _heff_dense(mps, H_full, i)
            mps.tensors[i], E = _local_eig(H_eff, mps.tensors[i],
                                           chi=chi, go_right=False)
            energies.append(E)
            # Right-canonicalise via LQ = QR of A^T
            A = mps.tensors[i]
            chi_l, d_i, chi_r = A.shape
            Q_0, R_0 = scipy.linalg.qr(A.reshape(chi_l, d_i * chi_r).T,
                                        mode="economic")
            # A = R_0.T @ Q_0.T  (LQ decomposition)
            k = Q_0.shape[1]
            mps.tensors[i] = Q_0.T.reshape(k, d_i, chi_r)
            if i > 0:
                mps.tensors[i - 1] = np.einsum("asc, cb -> asb",
                                               mps.tensors[i - 1], R_0.T)

        if E_before is not None and abs(energies[-1] - E_before) < tol:
            converged = True
            break

    return DMRGResult(
        mps_final=mps,
        energies=energies,
        n_sweeps=sweep_idx + 1,
        converged=converged,
    )


def _heff_dense(mps: MPS, H_full: np.ndarray, site: int) -> np.ndarray:
    """Build the local effective Hamiltonian for `site` by dense projection.

    Returns a (chi_l*d*chi_r, chi_l*d*chi_r) Hermitian matrix.
    P[:, col] is the global state when the tensor at `site` is the col-th
    local basis vector and all other tensors are fixed.
    """
    A       = mps.tensors[site]
    chi_l, d, chi_r = A.shape
    dim     = chi_l * d * chi_r
    state_d = H_full.shape[0]
    cplx    = np.iscomplexobj(H_full) or any(np.iscomplexobj(t) for t in mps.tensors)
    dtype   = complex if cplx else float

    P = np.zeros((state_d, dim), dtype=dtype)
    for col in range(dim):
        v = np.zeros(dim, dtype=dtype)
        v[col] = 1.0
        tensors_v = [t.copy() for t in mps.tensors]
        tensors_v[site] = v.reshape(chi_l, d, chi_r)
        P[:, col] = mps_to_state(MPS(tensors_v))

    Heff = P.conj().T @ H_full @ P
    return (Heff + Heff.conj().T) * 0.5


def _heff_dense_bond(mps: MPS, H_full: np.ndarray, left_site: int) -> np.ndarray:
    """Build the zero-site effective Hamiltonian for bond (left_site, left_site+1).

    Used in single-site TDVP for the backward half-step on the bond tensor C.
    Returns a (chi_r * chi_l_next, chi_r * chi_l_next) Hermitian matrix where
    chi_r  = mps.tensors[left_site].shape[2]
    chi_l_next = mps.tensors[left_site+1].shape[0].
    """
    n           = mps.n_sites
    d           = mps.phys_dim
    chi_r       = mps.tensors[left_site].shape[2]
    chi_l_next  = mps.tensors[left_site + 1].shape[0]
    dim         = chi_r * chi_l_next

    # L_flat: shape (d^{left_site+1}, chi_r) — contract A_0 ... A_{left_site}
    L: np.ndarray = np.ones((1, 1))
    for i in range(left_site + 1):
        A   = mps.tensors[i]
        dL  = L.shape[0]
        chi_l_i, _, chi_r_i = A.shape
        L = np.einsum("ia,ajb->ijb", L.reshape(dL, chi_l_i), A).reshape(dL * d, chi_r_i)

    # R_flat: shape (chi_l_next, d^{n-left_site-1}) — contract A_{left_site+1}...A_{n-1}
    R: np.ndarray = np.ones((1, 1))
    for i in range(n - 1, left_site, -1):
        A   = mps.tensors[i]
        dR  = R.shape[1]
        chi_l_i, _, chi_r_i = A.shape
        R = np.einsum("aib,bj->aij", A, R.reshape(chi_r_i, dR)).reshape(chi_l_i, d * dR)

    cplx  = np.iscomplexobj(H_full) or any(np.iscomplexobj(t) for t in mps.tensors)
    dtype = complex if cplx else float
    P     = np.zeros((H_full.shape[0], dim), dtype=dtype)
    for col in range(dim):
        al = col // chi_l_next
        ar = col % chi_l_next
        P[:, col] = np.outer(L[:, al], R[ar, :]).ravel()

    Heff = P.conj().T @ H_full @ P
    return (Heff + Heff.conj().T) * 0.5


def _local_eig(
    H_eff: np.ndarray,
    A_old: np.ndarray,
    chi: int | None = None,
    go_right: bool = True,
) -> tuple[np.ndarray, float]:
    """Solve H_eff v = E v for the lowest eigenvalue and reshape as a tensor."""
    chi_l, d, chi_r = A_old.shape
    evals, evecs = np.linalg.eigh(H_eff)
    E   = float(evals[0])
    v   = evecs[:, 0].reshape(chi_l, d, chi_r)
    return v, E


# ── two-site DMRG sweep ────────────────────────────────────────────────────


def dmrg_sweep_2site(
    mps: MPS,
    h_terms: list[np.ndarray],
    n_sweeps: int = 10,
    chi: int | None = None,
    tol: float = 1e-8,
) -> DMRGResult:
    """Two-site DMRG variational ground-state search.

    Optimises two neighbouring sites simultaneously, then splits via SVD.
    This allows the bond dimension to adapt during sweeps and can escape
    local minima that trap single-site DMRG.

    Honest scope: converges to a local minimum of the energy on the MPS
    manifold.  Exact for ``chi ≥ 2^(n/2)``; use ``temple_lanczos`` for a
    certified lower bound.  Dense effective Hamiltonian — suitable for
    small systems (``n ≤ 8``).

    Parameters
    ----------
    mps:      initial MPS.
    h_terms:  list of (d², d²) nearest-neighbour Hamiltonian matrices.
    n_sweeps: maximum number of full sweeps (L→R + R→L).
    chi:      bond-dimension cap; ``None`` keeps the full SVD rank.
    tol:      energy convergence tolerance per sweep.
    """
    mps = _left_canonicalise(mps.copy())
    n   = mps.n_sites
    d   = mps.phys_dim
    H_full    = nn_hamiltonian(h_terms, n, d)
    energies: list[float] = []
    converged = False

    for sweep_idx in range(n_sweeps):
        E_before = energies[-1] if energies else None

        # ── L→R half-sweep ─────────────────────────────────────────────
        for i in range(n - 1):
            H_eff = _heff_dense_2site(mps, H_full, i)
            theta, E = _local_eig_2site(H_eff, mps, i)
            energies.append(E)
            chi_l, _, _, chi_r = theta.shape
            U, s, Vh, k = _svd_truncate(theta.reshape(chi_l * d, d * chi_r), chi)
            mps.tensors[i]     = U.reshape(chi_l, d, k)
            mps.tensors[i + 1] = (s[:, None] * Vh).reshape(k, d, chi_r)

        # ── R→L half-sweep ─────────────────────────────────────────────
        for i in range(n - 2, -1, -1):
            H_eff = _heff_dense_2site(mps, H_full, i)
            theta, E = _local_eig_2site(H_eff, mps, i)
            energies.append(E)
            chi_l, _, _, chi_r = theta.shape
            U, s, Vh, k = _svd_truncate(theta.reshape(chi_l * d, d * chi_r), chi)
            mps.tensors[i]     = (U * s[None, :]).reshape(chi_l, d, k)
            mps.tensors[i + 1] = Vh.reshape(k, d, chi_r)

        if E_before is not None and abs(energies[-1] - E_before) < tol:
            converged = True
            break

    return DMRGResult(
        mps_final=mps,
        energies=energies,
        n_sweeps=sweep_idx + 1,
        converged=converged,
    )


def _svd_truncate(
    M: np.ndarray,
    chi: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """SVD of M with optional truncation to chi singular values."""
    U, s, Vh = scipy.linalg.svd(M, full_matrices=False)
    k = len(s) if chi is None else min(chi, len(s))
    k = max(k, 1)
    return U[:, :k], s[:k], Vh[:k], k


def _heff_dense_2site(mps: MPS, H_full: np.ndarray, site: int) -> np.ndarray:
    """Build the 2-site effective Hamiltonian for sites (site, site+1).

    Returns a (chi_l*d²*chi_r × chi_l*d²*chi_r) symmetric real matrix.
    Column ``col`` of the projection P is the full state when the two-site
    tensor theta is the ``col``-th standard basis vector.
    """
    A_i   = mps.tensors[site]
    A_i1  = mps.tensors[site + 1]
    chi_l, d, _    = A_i.shape
    _,     _, chi_r = A_i1.shape
    dim     = chi_l * d * d * chi_r
    state_d = H_full.shape[0]

    cplx = np.iscomplexobj(H_full)
    dtype = complex if cplx else float
    P = np.zeros((state_d, dim), dtype=dtype)
    for col in range(dim):
        # Decode col → (alpha_l, si, si1, alpha_r) in C-order
        alpha_r = col % chi_r
        rest    = col // chi_r
        si1     = rest % d
        rest  //= d
        si      = rest % d
        alpha_l = rest // d

        # Build two-site basis state: theta = δ_{alpha_l,si,si1,alpha_r}
        # Represented as A_i_v[alpha_l, si, 0]=1, A_i1_v[0, si1, alpha_r]=1
        tensors_v          = [t.copy() for t in mps.tensors]
        A_i_v              = np.zeros_like(A_i)
        A_i1_v             = np.zeros_like(A_i1)
        A_i_v[alpha_l, si, 0]    = 1.0
        A_i1_v[0, si1, alpha_r]  = 1.0
        tensors_v[site]     = A_i_v
        tensors_v[site + 1] = A_i1_v

        P[:, col] = mps_to_state(MPS(tensors_v))

    Heff = P.conj().T @ H_full @ P
    return (Heff + Heff.conj().T) * 0.5


def _local_eig_2site(
    H_eff: np.ndarray,
    mps: MPS,
    site: int,
) -> tuple[np.ndarray, float]:
    """Solve the 2-site local eigenvalue problem."""
    chi_l = mps.tensors[site].shape[0]
    chi_r = mps.tensors[site + 1].shape[2]
    d     = mps.phys_dim
    evals, evecs = np.linalg.eigh(H_eff)
    E = float(evals[0])
    return evecs[:, 0].reshape(chi_l, d, d, chi_r), E


# ── single-site TDVP ──────────────────────────────────────────────────────


def tdvp_evolve(
    mps: MPS,
    h_terms: list[np.ndarray],
    dt: float,
    n_steps: int,
    imaginary: bool = False,
    measure_every: int = 1,
) -> TEBDResult:
    """Single-site TDVP time evolution.

    More accurate than TEBD within the current bond dimension: there is no
    Trotter error for states already representable in the MPS manifold.
    The bond dimension is preserved exactly; use ``mps_truncate`` or
    ``dmrg_sweep_2site`` to change it.

    Algorithm: 2nd-order symmetric 1-site TDVP (Haegeman et al., 2016).
    Each step consists of a forward L→R half-sweep (site expm + bond backward
    expm), a full step on the pivot site, and a backward R→L half-sweep.

    Honest scope
    ------------
    * Dense effective Hamiltonian — suitable for small systems (``n ≤ 8``).
    * For imaginary time (``imaginary=True``) the evolution is non-unitary;
      the MPS is normalised after every step.
    * No certification of errors; use ``temple_lanczos`` for a lower bound.
      ``[工程]``

    Parameters
    ----------
    mps:           initial MPS.
    h_terms:       nearest-neighbour Hamiltonian bond matrices.
    dt:            time step.
    n_steps:       total number of steps.
    imaginary:     use imaginary-time evolution (ground-state search).
    measure_every: record energy every this many steps (plus step 0).
    """
    mps   = _left_canonicalise(mps.copy())
    n     = mps.n_sites
    d     = mps.phys_dim
    H_full = nn_hamiltonian(h_terms, n, d)

    # sign convention: forward site step = expm(alpha_fwd * H_eff)
    alpha_fwd = -dt / 2 if imaginary else -1j * dt / 2

    energies: list[float] = []
    times: list[float]    = []

    for step in range(n_steps):
        if step % measure_every == 0:
            norm2 = float(abs(mps_inner(mps, mps)).real)
            energies.append(_nn_energy(mps, h_terms) / norm2)
            times.append(step * dt)

        # ── L→R half-sweep ────────────────────────────────────────────
        for i in range(n - 1):
            # Forward evolve site i by dt/2
            H_eff = _heff_dense(mps, H_full, i)
            chi_l, d_i, chi_r = mps.tensors[i].shape
            v = mps.tensors[i].ravel().astype(complex)
            mps.tensors[i] = (scipy.linalg.expm(alpha_fwd * H_eff) @ v).reshape(chi_l, d_i, chi_r)

            # QR → left-canonicalise site i
            A = mps.tensors[i]
            Q, R_mat = scipy.linalg.qr(A.reshape(chi_l * d_i, chi_r), mode="economic")
            k = Q.shape[1]
            mps.tensors[i] = Q.reshape(chi_l, d_i, k)

            # Backward evolve bond R_mat by dt/2
            H_bond = _heff_dense_bond(mps, H_full, i)
            c = R_mat.ravel().astype(complex)
            c = scipy.linalg.expm(-alpha_fwd * H_bond) @ c
            R_mat = c.reshape(k, chi_r)

            # Absorb R_mat into site i+1
            mps.tensors[i + 1] = np.einsum("ab,bsc->asc", R_mat, mps.tensors[i + 1])

        # Pivot site (last site): full dt step
        H_eff = _heff_dense(mps, H_full, n - 1)
        chi_l, d_i, chi_r = mps.tensors[n - 1].shape
        v = mps.tensors[n - 1].ravel().astype(complex)
        mps.tensors[n - 1] = (scipy.linalg.expm(2 * alpha_fwd * H_eff) @ v).reshape(chi_l, d_i, chi_r)

        # ── R→L half-sweep ────────────────────────────────────────────
        for i in range(n - 2, -1, -1):
            # LQ of site i+1 (QR of A^T)
            A = mps.tensors[i + 1]
            chi_l_i1, d_i1, chi_r_i1 = A.shape
            Q_T, R_T = scipy.linalg.qr(A.reshape(chi_l_i1, d_i1 * chi_r_i1).T, mode="economic")
            k = Q_T.shape[1]
            mps.tensors[i + 1] = Q_T.T.reshape(k, d_i1, chi_r_i1)
            C = R_T.T  # shape (chi_l_i1, k)

            # Backward evolve bond C by dt/2
            H_bond = _heff_dense_bond(mps, H_full, i)
            c = C.ravel().astype(complex)
            c = scipy.linalg.expm(-alpha_fwd * H_bond) @ c
            C = c.reshape(chi_l_i1, k)

            # Absorb C into site i
            mps.tensors[i] = np.einsum("asc,cb->asb", mps.tensors[i], C)

            # Forward evolve site i by dt/2
            H_eff = _heff_dense(mps, H_full, i)
            chi_l, d_i_s, chi_r = mps.tensors[i].shape
            v = mps.tensors[i].ravel().astype(complex)
            mps.tensors[i] = (scipy.linalg.expm(alpha_fwd * H_eff) @ v).reshape(chi_l, d_i_s, chi_r)

        if imaginary:
            mps = mps_normalise(mps)

    # Final energy measurement
    norm2 = float(abs(mps_inner(mps, mps)).real)
    energies.append(_nn_energy(mps, h_terms) / norm2)
    times.append(n_steps * dt)
    max_chi = max(t.shape[0] for t in mps.tensors)

    return TEBDResult(
        mps_final=mps,
        times=times,
        energies=energies,
        max_bonds=[max_chi] * len(energies),
        total_discarded=0.0,
        dt=dt,
        n_steps=n_steps,
        trotter_order=0,
    )
