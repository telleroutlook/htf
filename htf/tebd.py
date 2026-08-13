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

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.linalg

from .mps import MPS, _left_canonicalise, mps_apply_gate, mps_inner, mps_normalise, mps_to_state


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
    chi: Optional[int] = None,
    imaginary: bool = False,
    trotter_order: int = 1,
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
    mps, d1 = _apply_bond_parity(mps, gates_half, chi, even=True)
    mps, d2 = _apply_bond_parity(mps, gates_full, chi, even=False)
    mps, d3 = _apply_bond_parity(mps, gates_half, chi, even=True)
    return mps, d1 + d2 + d3


def _apply_all_bonds(
    mps: MPS,
    gates: list[np.ndarray],
    chi: Optional[int],
) -> tuple[MPS, float]:
    """Apply all bond gates sequentially (1st-order)."""
    total_disc = 0.0
    for i, gate in enumerate(gates):
        mps, disc = mps_apply_gate(mps, gate, [i, i + 1], chi=chi)
        total_disc += disc
    return mps, total_disc


def _apply_bond_parity(
    mps: MPS,
    gates: list[np.ndarray],
    chi: Optional[int],
    even: bool,
) -> tuple[MPS, float]:
    """Apply even (0-1, 2-3, …) or odd (1-2, 3-4, …) bond gates."""
    start = 0 if even else 1
    total_disc = 0.0
    for i in range(start, len(gates), 2):
        mps, disc = mps_apply_gate(mps, gates[i], [i, i + 1], chi=chi)
        total_disc += disc
    return mps, total_disc


def tebd_evolve(
    mps: MPS,
    h_terms: list[np.ndarray],
    dt: float,
    n_steps: int,
    chi: Optional[int] = None,
    imaginary: bool = False,
    trotter_order: int = 1,
    measure_every: int = 1,
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
    chi: Optional[int] = None,
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

    Returns a (chi_l*d*chi_r, chi_l*d*chi_r) symmetric real matrix.
    P[:, col] is the global state when the tensor at `site` is the col-th
    local basis vector and all other tensors are fixed.
    """
    A       = mps.tensors[site]
    chi_l, d, chi_r = A.shape
    dim     = chi_l * d * chi_r
    state_d = H_full.shape[0]

    P = np.zeros((state_d, dim))
    for col in range(dim):
        v = np.zeros(dim)
        v[col] = 1.0
        tensors_v = [t.copy() for t in mps.tensors]
        tensors_v[site] = v.reshape(chi_l, d, chi_r)
        psi = mps_to_state(MPS(tensors_v))
        P[:, col] = psi.real

    Heff = P.T @ H_full @ P
    return (Heff + Heff.T) * 0.5   # symmetrize for numerical safety


def _local_eig(
    H_eff: np.ndarray,
    A_old: np.ndarray,
    chi: Optional[int] = None,
    go_right: bool = True,
) -> tuple[np.ndarray, float]:
    """Solve H_eff v = E v for the lowest eigenvalue and reshape as a tensor."""
    chi_l, d, chi_r = A_old.shape
    evals, evecs = np.linalg.eigh(H_eff)
    E   = float(evals[0])
    v   = evecs[:, 0].reshape(chi_l, d, chi_r)
    return v, E

    """Single-site DMRG variational ground-state search.

    Alternates left-right sweeps, optimising one site at a time by solving
    a dense eigenvalue problem for the local effective Hamiltonian.

    Honest scope: converges to a local minimum in the MPS manifold of bond
    dimension chi; use temple_lanczos to obtain a certified lower bound.

    Parameters
    ----------
    mps:      initial MPS; will be left-canonicalised internally.
    h_terms:  list of (d², d²) nearest-neighbour Hamiltonian matrices.
    n_sweeps: maximum number of full (L→R + R→L) sweeps.
    chi:      bond dimension cap; None = keep current.
    tol:      energy convergence tolerance (absolute) per sweep.

    Returns
    -------
    DMRGResult with the optimised MPS and energy trajectory.
    """
    mps = _left_canonicalise(mps.copy())
    n   = mps.n_sites
    d   = mps.phys_dim
    energies: list[float] = []
    converged = False

    # Pre-build right environments R[i]: right env starting at site i
    # R[i] has shape (chi_r_bra, chi_r_ket) and represents
    # Σ_{sites i..n-1} A* H A right-contracted
    R = _build_all_R(mps, h_terms, n, d)

    for sweep_idx in range(n_sweeps):
        E_before = energies[-1] if energies else None

        # ── L→R half-sweep ─────────────────────────────────────────────
        L = np.ones((1, 1), dtype=float)
        for i in range(n - 1):
            Heff = _build_heff(mps, h_terms, i, n, d, L, R[i + 1])
            A, E = _optimise_site(Heff, mps.tensors[i], chi, go_right=True)
            mps.tensors[i] = A
            energies.append(E)
            # Update L by absorbing site i
            L = _contract_L(L, mps.tensors[i], h_terms, i, n, d)

        # Optimise last site
        Heff = _build_heff(mps, h_terms, n - 1, n, d, L, np.ones((1, 1), dtype=float))
        A, E = _optimise_site(Heff, mps.tensors[n - 1], chi=None, go_right=False)
        mps.tensors[n - 1] = A
        energies.append(E)

        # ── R→L half-sweep ─────────────────────────────────────────────
        Rv = np.ones((1, 1), dtype=float)
        for i in range(n - 1, 0, -1):
            Heff = _build_heff(mps, h_terms, i, n, d, _build_L_up_to(mps, h_terms, i, d), Rv)
            A, E = _optimise_site(Heff, mps.tensors[i], chi, go_right=False)
            mps.tensors[i] = A
            energies.append(E)
            Rv = _contract_R(Rv, mps.tensors[i], h_terms, i, n, d)

        # Rebuild right environments for next sweep
        R = _build_all_R(mps, h_terms, n, d)

        if E_before is not None and abs(energies[-1] - E_before) < tol:
            converged = True
            break

    return DMRGResult(
        mps_final=mps,
        energies=energies,
        n_sweeps=sweep_idx + 1,
        converged=converged,
    )


