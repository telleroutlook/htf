"""HTF §9-E — Matrix Product Operator (MPO) data structure and operations.

An MPO represents a quantum operator O on an n-site lattice as a product of
rank-4 tensors W[i] with shape (W_l, d, d, W_r):

    O[s0',s1',...; s0,s1,...] =
        W[0][0,s0',s0,:] · W[1][:,s1',s1,:] · ... · W[n-1][:,sn-1',sn-1,0]

Tensor convention: W[alpha, s', s, beta] where
  * s' = bra (output) physical index
  * s  = ket (input)  physical index

Honest scope [工程]
-------------------
* ``nn_hamiltonian_mpo`` uses a finite-automaton construction: bond dim at
  bond j = 2 + rank(h_j).  This avoids building the O(d^{2n}) full matrix.
* ``mpo_from_matrix`` finds the exact minimal MPO via sequential SVD; useful
  for round-trip tests but impractical for large n.
* Truncation error from ``mpo_from_matrix`` chi cap is not certified.
* Continuum limit is ``[OUT]``.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.linalg

from .mps import MPS, _left_canonicalise, mps_inner, random_mps


@dataclass
class MPO:
    """Matrix Product Operator.

    Each tensor has shape ``(W_l, d, d, W_r)`` where the second axis is the
    bra (output) physical index and the third is the ket (input) physical
    index.  Boundary conditions: ``tensors[0].shape[0] == 1`` and
    ``tensors[-1].shape[3] == 1``.
    """

    tensors: list[np.ndarray]

    @property
    def n_sites(self) -> int:
        return len(self.tensors)

    @property
    def phys_dim(self) -> int:
        return self.tensors[0].shape[1]

    def copy(self) -> "MPO":
        return MPO([t.copy() for t in self.tensors])


# ─────────────────────────────── constructors ─────────────────────────────


def identity_mpo(n: int, d: int = 2) -> MPO:
    """Identity operator as an MPO with bond dimension 1 everywhere."""
    I = np.eye(d, dtype=float).reshape(1, d, d, 1)
    return MPO([I.copy() for _ in range(n)])


def random_mpo(
    n: int,
    d: int = 2,
    chi: int = 2,
    seed: Optional[int] = None,
) -> MPO:
    """Random MPO with given bond dimension (not Hermitian in general)."""
    rng = np.random.default_rng(seed)
    tensors = []
    for i in range(n):
        W_l = 1 if i == 0 else chi
        W_r = 1 if i == n - 1 else chi
        tensors.append(rng.standard_normal((W_l, d, d, W_r)))
    return MPO(tensors)


def mpo_from_matrix(
    H: np.ndarray,
    n: int,
    d: int,
    chi: Optional[int] = None,
) -> MPO:
    """Convert a full operator matrix H (shape d^n × d^n) to MPO via SVD.

    Rows of H are bra (output) indices; columns are ket (input) indices.
    An optional ``chi`` cap limits the MPO bond dimension (lossy).
    """
    is_cplx = np.iscomplexobj(H)
    H_r = H.reshape([d] * n + [d] * n)
    # Interleave axes to (s0_bra, s0_ket, s1_bra, s1_ket, ...)
    idx = [j for i in range(n) for j in (i, n + i)]
    H_int = np.transpose(H_r, idx)   # shape (d, d, ..., d, d) with 2n dims

    tensors: list[np.ndarray] = []
    W_l = 1
    remaining = H_int.reshape(1, (d * d) ** n)

    for i in range(n - 1):
        rest = (d * d) ** (n - i - 1)
        M = remaining.reshape(W_l * d * d, rest)
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        if chi is not None:
            k = min(chi, len(s))
        else:
            tol = 1e-12 * (float(s[0]) if len(s) > 0 else 1.0)
            k = max(1, int(np.sum(s > tol)))
        W_r = k
        tensors.append(U[:, :k].reshape(W_l, d, d, W_r))
        remaining = (s[:k, None] * Vt[:k, :]).reshape(W_r, rest)
        W_l = W_r

    tensors.append(remaining.reshape(W_l, d, d, 1))
    if is_cplx:
        tensors = [t.astype(complex) for t in tensors]
    return MPO(tensors)


def nn_hamiltonian_mpo(
    h_terms: list[np.ndarray],
    n: int,
    d: int = 2,
) -> MPO:
    """Build MPO from n-1 nearest-neighbour bond Hamiltonians.

    Uses the finite-automaton construction.  The bond dimension at each
    virtual bond equals 2 + rank(h_i), making the total cost O(n d^4 r^2)
    for contraction — far cheaper than the d^{2n} full-matrix route.

    Parameters
    ----------
    h_terms: list of n-1 arrays, each shape (d^2, d^2), where rows are bra
             and columns are ket.  h_terms[i] acts on sites (i, i+1).
    n:       number of lattice sites.
    d:       physical dimension per site.
    """
    if len(h_terms) != n - 1:
        raise ValueError(f"need {n - 1} terms for {n} sites, got {len(h_terms)}")

    is_cplx = any(np.iscomplexobj(h) for h in h_terms)

    # SVD-factorize each bond term h_i into left ⊗ right operators.
    # h_i has shape (d^2, d^2) with row=(s_i', s_{i+1}') and col=(s_i, s_{i+1}).
    # Reshape to (d,d,d,d)[s_i',s_{i+1}',s_i,s_{i+1}], then transpose to
    # (s_i',s_i, s_{i+1}',s_{i+1}) and SVD over the (s_i',s_i) bipartition.
    svds: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for h in h_terms:
        h4 = h.reshape(d, d, d, d)
        M = h4.transpose(0, 2, 1, 3).reshape(d * d, d * d)
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        tol = 1e-12 * float(s[0]) if len(s) > 0 and s[0] != 0 else 1e-12
        r = max(1, int(np.sum(s > tol)))
        svds.append((U[:, :r], s[:r], Vt[:r, :]))

    tensors: list[np.ndarray] = []

    for i in range(n):
        dtype = complex if is_cplx else float

        if i == 0:
            U0, s0, Vt0 = svds[0]
            r0 = len(s0)
            W_r = 2 + r0
            W = np.zeros((1, d, d, W_r), dtype=dtype)
            W[0, :, :, 0] = np.eye(d)                              # |L⟩ → |L⟩
            for k in range(r0):
                A_k = np.sqrt(s0[k]) * U0[:, k].reshape(d, d)
                W[0, :, :, k + 1] = A_k                            # |L⟩ → |k⟩
            # W[0,:,:,W_r-1] stays 0: can't accumulate a complete term at site 0
            tensors.append(W)

        elif i == n - 1:
            Uprev, sprev, Vtprev = svds[n - 2]
            r_prev = len(sprev)
            W_l = 2 + r_prev
            W = np.zeros((W_l, d, d, 1), dtype=dtype)
            for k in range(r_prev):
                B_k = np.sqrt(sprev[k]) * Vtprev[k, :].reshape(d, d)
                W[k + 1, :, :, 0] = B_k                            # |k⟩ → |R⟩
            W[W_l - 1, :, :, 0] = np.eye(d)                        # |R⟩ → |R⟩
            tensors.append(W)

        else:
            Uprev, sprev, Vtprev = svds[i - 1]
            Ucurr, scurr, Vtcurr = svds[i]
            r_prev, r_curr = len(sprev), len(scurr)
            W_l = 2 + r_prev
            W_r = 2 + r_curr
            W = np.zeros((W_l, d, d, W_r), dtype=dtype)
            W[0, :, :, 0] = np.eye(d)                              # |L⟩ → |L⟩
            for k in range(r_curr):
                A_k = np.sqrt(scurr[k]) * Ucurr[:, k].reshape(d, d)
                W[0, :, :, k + 1] = A_k                            # |L⟩ → |k⟩
            for k in range(r_prev):
                B_k = np.sqrt(sprev[k]) * Vtprev[k, :].reshape(d, d)
                W[k + 1, :, :, W_r - 1] = B_k                     # |k⟩ → |R⟩
            W[W_l - 1, :, :, W_r - 1] = np.eye(d)                 # |R⟩ → |R⟩
            tensors.append(W)

    return MPO(tensors)


# ─────────────────────────────── linear algebra ────────────────────────────


def mpo_to_matrix(mpo: MPO) -> np.ndarray:
    """Reconstruct the full d^n × d^n operator matrix from an MPO.

    Row/column index ordering: rows = bra = (s0', s1', …), cols = ket.
    """
    n, d = mpo.n_sites, mpo.phys_dim
    # Site 0: squeeze W_l=1 → shape (d, d, W_r)
    result = mpo.tensors[0][0]   # (d, d, W_r) = (s0', s0, W_r)
    for i in range(1, n):
        W = mpo.tensors[i]       # (W_l, d, d, W_r)
        # Accumulate: result (…, W_l) × W (W_l, d, d, W_r) → (…, d, d, W_r)
        result = np.tensordot(result, W, axes=([-1], [0]))
    # result shape: (d, d, d, d, …, d, d, W_r=1), last dim squeezed
    result = result[..., 0]      # 2n dims: (s0', s0, s1', s1, …, sn-1', sn-1)
    bra = list(range(0, 2 * n, 2))
    ket = list(range(1, 2 * n, 2))
    return np.transpose(result, bra + ket).reshape(d ** n, d ** n)


def mpo_hermitian_conjugate(mpo: MPO) -> MPO:
    """Return the Hermitian conjugate O†.

    Swaps bra ↔ ket physical indices and complex-conjugates all entries.
    """
    return MPO([W.conj().transpose(0, 2, 1, 3) for W in mpo.tensors])


def mpo_apply_mps(mpo: MPO, mps: MPS) -> MPS:
    """Apply MPO to MPS, returning a new MPS representing O|ψ⟩.

    The result has bond dimension W_l·chi_l at each site (uncompressed).
    Use ``mps_truncate`` afterwards if needed.
    """
    new_tensors: list[np.ndarray] = []
    for i in range(mpo.n_sites):
        W = mpo.tensors[i]           # (W_l, d_out, d_in, W_r)
        A = mps.tensors[i]           # (chi_l, d_in, chi_r)
        W_l, d_out, _, W_r = W.shape
        chi_l, _, chi_r = A.shape
        # B[(W_l, chi_l), d_out, (W_r, chi_r)] = Σ_s W[W_l, d_out, s, W_r] * A[chi_l, s, chi_r]
        B = np.einsum("woiv,aib->waovb", W, A).reshape(W_l * chi_l, d_out, W_r * chi_r)
        new_tensors.append(B)
    return MPS(new_tensors)


def mpo_expectation(mpo: MPO, mps: MPS) -> complex:
    """Compute ⟨ψ|O|ψ⟩ using the MPO and MPS.

    Returns the (generally complex) expectation value.  For Hermitian O and
    a normalised |ψ⟩ this should be real.
    """
    O_psi = mpo_apply_mps(mpo, mps)
    return mps_inner(mps, O_psi)


# ─────────────────────────── MPO-based DMRG ───────────────────────────────


@dataclass
class MPODMRGResult:
    """Result of MPO-environment DMRG variational optimisation.

    Attributes
    ----------
    mps_final:  Optimised MPS at ground-state energy minimum.
    energies:   Energy per local site update (one per half-sweep site).
    n_sweeps:   Number of full L↔R sweeps completed.
    converged:  True if energy change per sweep fell below ``tol``.
    """
    mps_final: MPS
    energies: list[float] = field(default_factory=list)
    n_sweeps: int = 0
    converged: bool = False


@dataclass
class MultiStartDMRGResult:
    """Result from a parallel multi-start DMRG calculation.

    Attributes
    ----------
    best:         :class:`MPODMRGResult` with the lowest final energy.
    all_energies: Final energy from each seed (length = n_seeds).
    seeds_used:   Seeds passed to each run.
    best_seed:    Which seed produced the lowest energy.
    n_workers:    Number of parallel worker processes used.
    """
    best:         MPODMRGResult
    all_energies: list
    seeds_used:   list
    best_seed:    int
    n_workers:    int


def _update_left_env(
    L_old: np.ndarray,
    W: np.ndarray,
    A: np.ndarray,
) -> np.ndarray:
    """Grow the left environment by one site.

    Parameters
    ----------
    L_old : (chi_bra_l, W_bond_l, chi_ket_l)
    W     : (W_bond_l, d_bra, d_ket, W_bond_r)
    A     : (chi_ket_l, d, chi_ket_r)  — left-canonicalised site tensor

    Returns
    -------
    L_new : (chi_bra_r, W_bond_r, chi_ket_r)

    Index legend: i=chi_l_bra, p=W_l, j=chi_l_ket,
                  s=d_bra, t=d_ket, q=W_r, c=chi_r_bra, e=chi_r_ket
    """
    # conj(A)[i,s,c], L_old[i,p,j], W[p,s,t,q], A[j,t,e] -> L_new[c,q,e]
    return np.einsum("isc,ipj,pstq,jte->cqe", A.conj(), L_old, W, A,
                     optimize=True)


def _update_right_env(
    R_old: np.ndarray,
    W: np.ndarray,
    A: np.ndarray,
) -> np.ndarray:
    """Grow the right environment by one site.

    Parameters
    ----------
    R_old : (chi_bra_r, W_bond_r, chi_ket_r)
    W     : (W_bond_l, d_bra, d_ket, W_bond_r)
    A     : (chi_ket_l, d, chi_ket_r)  — right-canonicalised site tensor

    Returns
    -------
    R_new : (chi_bra_l, W_bond_l, chi_ket_l)

    Index legend: a=chi_r_bra, p=W_r, b=chi_r_ket,
                  c=chi_l_bra, r=W_l, e=chi_l_ket, s=d_bra, t=d_ket
    """
    # conj(A)[c,s,a], W[r,s,t,p], R_old[a,p,b], A[e,t,b] -> R_new[c,r,e]
    return np.einsum("csa,rstp,apb,etb->cre", A.conj(), W, R_old, A,
                     optimize=True)


def _heff_mpo_local(
    L: np.ndarray,
    W: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """Local effective Hamiltonian from MPO environments.

    H_eff[(i,s,k),(j,t,l)] = Σ_{p,q} L[i,p,j] · W[p,s,t,q] · R[k,q,l]

    Parameters
    ----------
    L : (chi_l, W_l, chi_l)
    W : (W_l, d_bra, d_ket, W_r)
    R : (chi_r, W_r, chi_r)

    Returns
    -------
    H_eff : (chi_l*d*chi_r, chi_l*d*chi_r)  Hermitian matrix
    """
    chi_l = L.shape[0]
    d     = W.shape[1]
    chi_r = R.shape[0]
    H = np.einsum("ipj,pstq,kql->iskjtl", L, W, R, optimize=True)
    H_eff = H.reshape(chi_l * d * chi_r, chi_l * d * chi_r)
    return (H_eff + H_eff.conj().T) * 0.5   # enforce Hermitian symmetry


def dmrg_sweep_mpo(
    mps: MPS,
    mpo: MPO,
    n_sweeps: int = 10,
    chi: Optional[int] = None,
    tol: float = 1e-8,
) -> MPODMRGResult:
    """MPO-environment single-site DMRG variational ground-state search.

    Uses incremental left/right environment tensors built from MPO
    contractions.  Each site update costs O(χ²·W·d) instead of O(d^{2n}),
    making this approach practical for n up to hundreds of sites.

    Parameters
    ----------
    mps      : initial guess MPS (left-canonicalised internally).
    mpo      : Hamiltonian as MPO (from ``nn_hamiltonian_mpo`` or
               ``mpo_from_matrix``).
    n_sweeps : maximum number of full sweeps (L→R + R→L).
    chi      : bond-dimension cap applied after SVD; None = no compression.
    tol      : convergence threshold on |ΔE| per sweep.

    Returns
    -------
    :class:`MPODMRGResult`
    """
    mps = _left_canonicalise(mps.copy())
    n   = mps.n_sites

    # Right-canonicalise the initial MPS to put the orthogonality centre at
    # site 0 and simultaneously build all right environments.  This ensures
    # H_eff eigenvalues are valid energy estimates from the first update.
    R_envs: list[np.ndarray] = [np.ones((1, 1, 1))] * (n + 1)
    for i in range(n - 1, 0, -1):
        A = mps.tensors[i]
        chi_l, d_i, chi_r = A.shape
        Q_T, R_T = scipy.linalg.qr(A.reshape(chi_l, d_i * chi_r).T, mode="economic")
        k = Q_T.shape[1]
        mps.tensors[i] = Q_T.T.reshape(k, d_i, chi_r)
        mps.tensors[i - 1] = np.einsum("asc,cb->asb", mps.tensors[i - 1], R_T.T)
        R_envs[i] = _update_right_env(R_envs[i + 1], mpo.tensors[i], mps.tensors[i])
    R_envs[0] = _update_right_env(R_envs[1], mpo.tensors[0], mps.tensors[0])

    # L_envs[i]: left environment that covers sites 0, 1, …, i-1.
    L_envs: list[np.ndarray] = [np.ones((1, 1, 1))] * (n + 1)

    energies: list[float] = []
    converged = False
    sweep_idx = 0

    for sweep_idx in range(n_sweeps):
        E_start = energies[-1] if energies else None

        # ── L→R half-sweep: sites 0 … n-2 ────────────────────────────────
        for i in range(n - 1):
            H_eff = _heff_mpo_local(L_envs[i], mpo.tensors[i], R_envs[i + 1])
            chi_l, d_i, chi_r = mps.tensors[i].shape
            evals, evecs = scipy.linalg.eigh(H_eff)
            energies.append(float(evals[0]))
            theta = evecs[:, 0].reshape(chi_l, d_i, chi_r)
            # QR → left-canonical A[i], pass R factor to next site
            Q, R_mat = scipy.linalg.qr(
                theta.reshape(chi_l * d_i, chi_r), mode="economic"
            )
            if chi is not None:
                k = min(chi, Q.shape[1])
                Q, R_mat = Q[:, :k], R_mat[:k, :]
            k = Q.shape[1]
            mps.tensors[i]     = Q.reshape(chi_l, d_i, k)
            mps.tensors[i + 1] = np.einsum("ab,bsc->asc", R_mat,
                                           mps.tensors[i + 1])
            L_envs[i + 1] = _update_left_env(
                L_envs[i], mpo.tensors[i], mps.tensors[i]
            )

        # Pivot: optimise site n-1 (full step)
        i = n - 1
        H_eff = _heff_mpo_local(L_envs[i], mpo.tensors[i], R_envs[i + 1])
        chi_l, d_i, chi_r = mps.tensors[i].shape
        evals, evecs = scipy.linalg.eigh(H_eff)
        energies.append(float(evals[0]))
        mps.tensors[i] = evecs[:, 0].reshape(chi_l, d_i, chi_r)

        # Seed R envs for R→L sweep using updated pivot tensor.
        R_envs[n]     = np.ones((1, 1, 1))
        R_envs[n - 1] = _update_right_env(
            R_envs[n], mpo.tensors[n - 1], mps.tensors[n - 1]
        )

        # ── R→L half-sweep: sites n-2 … 0 ────────────────────────────────
        for i in range(n - 2, -1, -1):
            H_eff = _heff_mpo_local(L_envs[i], mpo.tensors[i], R_envs[i + 1])
            chi_l, d_i, chi_r = mps.tensors[i].shape
            evals, evecs = scipy.linalg.eigh(H_eff)
            energies.append(float(evals[0]))
            theta = evecs[:, 0].reshape(chi_l, d_i, chi_r)
            # LQ → right-canonical A[i], pass L factor to previous site
            Q_T, R_T = scipy.linalg.qr(
                theta.reshape(chi_l, d_i * chi_r).T, mode="economic"
            )
            if chi is not None:
                k = min(chi, Q_T.shape[1])
                Q_T, R_T = Q_T[:, :k], R_T[:k, :]
            k = Q_T.shape[1]
            mps.tensors[i] = Q_T.T.reshape(k, d_i, chi_r)
            if i > 0:
                mps.tensors[i - 1] = np.einsum(
                    "asc,cb->asb", mps.tensors[i - 1], R_T.T
                )
            # Update R env using newly right-canonical A[i]
            R_envs[i] = _update_right_env(
                R_envs[i + 1], mpo.tensors[i], mps.tensors[i]
            )

        # Reset L boundary for next L→R sweep
        L_envs[0] = np.ones((1, 1, 1))

        if E_start is not None and abs(energies[-1] - E_start) < tol:
            converged = True
            break

    return MPODMRGResult(
        mps_final=mps,
        energies=energies,
        n_sweeps=sweep_idx + 1,
        converged=converged,
    )


# ── §9-G helpers: two-site effective Hamiltonian ─────────────────────────────


def _heff_mpo_2site(
    L: np.ndarray,
    Wi: np.ndarray,
    Wj: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """Two-site effective Hamiltonian from L/Wi/Wj/R environment tensors.

    H_eff[(i,s,u,k),(j,t,v,l)] = Σ_{p,q,r} L[i,p,j] · Wi[p,s,t,q]
                                            · Wj[q,u,v,r] · R[k,r,l]
    where row=(i,s,u,k) and col=(j,t,v,l) are (chi_l,d,d,chi_r) tuples.

    Parameters
    ----------
    L  : left environment, shape (chi_l, W_l, chi_l).
    Wi : MPO tensor at site i, shape (W_l, d, d, W_m).
    Wj : MPO tensor at site i+1, shape (W_m, d, d, W_r).
    R  : right environment, shape (chi_r, W_r, chi_r).

    Returns
    -------
    Symmetrised H_eff of shape (chi_l*d*d*chi_r, chi_l*d*d*chi_r).
    """
    chi_l = L.shape[0]
    d     = Wi.shape[1]
    chi_r = R.shape[0]
    # einsum axes: i=χ_l_bra, p=W_l, j=χ_l_ket, s=d_bra_i, t=d_ket_i,
    #              q=W_m, u=d_bra_j, v=d_ket_j, r=W_r, k=χ_r_bra, l=χ_r_ket
    H = np.einsum("ipj,pstq,quvr,krl->isukjtvl", L, Wi, Wj, R, optimize=True)
    H_eff = H.reshape(chi_l * d * d * chi_r, -1)
    return (H_eff + H_eff.conj().T) * 0.5


def dmrg_sweep_mpo_2site(
    mps: MPS,
    mpo: MPO,
    n_sweeps: int = 10,
    chi: Optional[int] = None,
    tol: float = 1e-8,
) -> MPODMRGResult:
    """MPO-environment two-site DMRG variational ground-state search.

    Each update jointly optimises a pair of sites by diagonalising a
    two-site effective Hamiltonian and splitting the result via SVD.  The
    SVD truncation allows the bond dimension to **grow** from the initial
    MPS, avoiding the local-minimum traps of single-site DMRG.

    Algorithm sketch per half-sweep (L→R shown; R→L mirrors)::

        for each pair (i, i+1):
            H_2site = _heff_mpo_2site(L[i], W[i], W[i+1], R[i+2])
            theta   = leading eigenvector, shape (χ_l·d, d·χ_r)
            U, s, Vt = svd(theta, truncate to chi)
            A[i]    = U.reshape(χ_l, d, k)          # left-canonical
            A[i+1]  = (s * Vt).reshape(k, d, χ_r)  # absorb singular values
            update L[i+1]

    Parameters
    ----------
    mps      : initial guess MPS (left-canonicalised internally).
    mpo      : Hamiltonian as MPO.
    n_sweeps : maximum number of full sweeps (L→R + R→L).
    chi      : bond-dimension cap applied after SVD.  ``None`` = no cap
               (bond dim grows up to min(d·χ_l, d·χ_r)).
    tol      : convergence threshold on |ΔE| between consecutive sweeps.

    Returns
    -------
    :class:`MPODMRGResult`
    """
    mps = _left_canonicalise(mps.copy())
    n   = mps.n_sites
    d   = mps.phys_dim

    # Right-canonicalise to place the orthogonality centre at site 0
    # and build all initial right environments.
    R_envs: list[np.ndarray] = [np.ones((1, 1, 1))] * (n + 1)
    for i in range(n - 1, 0, -1):
        A = mps.tensors[i]
        cl, di, cr = A.shape
        Q_T, R_T = scipy.linalg.qr(A.reshape(cl, di * cr).T, mode="economic")
        k = Q_T.shape[1]
        mps.tensors[i]     = Q_T.T.reshape(k, di, cr)
        mps.tensors[i - 1] = np.einsum("asc,cb->asb", mps.tensors[i - 1], R_T.T)
        R_envs[i] = _update_right_env(R_envs[i + 1], mpo.tensors[i], mps.tensors[i])
    R_envs[0] = _update_right_env(R_envs[1], mpo.tensors[0], mps.tensors[0])

    L_envs: list[np.ndarray] = [np.ones((1, 1, 1))] * (n + 1)

    energies: list[float] = []
    converged = False
    sweep_idx = 0

    for sweep_idx in range(n_sweeps):
        E_start = energies[-1] if energies else None

        # ── L→R half-sweep: pairs (0,1), (1,2), …, (n-2, n-1) ─────────────
        for i in range(n - 1):
            chi_l = mps.tensors[i].shape[0]
            chi_r = mps.tensors[i + 1].shape[2]
            H_2site = _heff_mpo_2site(
                L_envs[i], mpo.tensors[i], mpo.tensors[i + 1], R_envs[i + 2]
            )
            evals, evecs = scipy.linalg.eigh(H_2site)
            energies.append(float(evals[0]))
            theta = evecs[:, 0].reshape(chi_l * d, d * chi_r)
            U, s, Vt = np.linalg.svd(theta, full_matrices=False)
            k = min(chi, len(s)) if chi is not None else len(s)
            mps.tensors[i]     = U[:, :k].reshape(chi_l, d, k)
            mps.tensors[i + 1] = (s[:k, None] * Vt[:k, :]).reshape(k, d, chi_r)
            L_envs[i + 1] = _update_left_env(
                L_envs[i], mpo.tensors[i], mps.tensors[i]
            )

        # ── R→L half-sweep: pairs (n-2, n-1), …, (0, 1) ───────────────────
        for i in range(n - 2, -1, -1):
            chi_l = mps.tensors[i].shape[0]
            chi_r = mps.tensors[i + 1].shape[2]
            H_2site = _heff_mpo_2site(
                L_envs[i], mpo.tensors[i], mpo.tensors[i + 1], R_envs[i + 2]
            )
            evals, evecs = scipy.linalg.eigh(H_2site)
            energies.append(float(evals[0]))
            theta = evecs[:, 0].reshape(chi_l * d, d * chi_r)
            U, s, Vt = np.linalg.svd(theta, full_matrices=False)
            k = min(chi, len(s)) if chi is not None else len(s)
            mps.tensors[i + 1] = Vt[:k, :].reshape(k, d, chi_r)
            mps.tensors[i]     = (U[:, :k] * s[:k]).reshape(chi_l, d, k)
            R_envs[i + 1] = _update_right_env(
                R_envs[i + 2], mpo.tensors[i + 1], mps.tensors[i + 1]
            )

        # Reset L boundary for the next L→R sweep
        L_envs[0] = np.ones((1, 1, 1))

        if E_start is not None and abs(energies[-1] - E_start) < tol:
            converged = True
            break

    return MPODMRGResult(
        mps_final=mps,
        energies=energies,
        n_sweeps=sweep_idx + 1,
        converged=converged,
    )


# ── §9-H: parallel multi-start DMRG ─────────────────────────────────────────


def _dmrg_worker(packed):
    """Top-level worker for ProcessPoolExecutor — must stay at module level."""
    tensors, seed, chi_init, chi, n_sweeps, tol = packed
    mpo = MPO(list(tensors))
    n, d = mpo.n_sites, mpo.phys_dim
    mps = random_mps(n, d, chi=chi_init, seed=seed)
    return dmrg_sweep_mpo_2site(mps, mpo, n_sweeps=n_sweeps, chi=chi, tol=tol)


def dmrg_multistart(
    mpo: MPO,
    n_seeds: int = 8,
    chi: int = 16,
    chi_init: int = 2,
    n_sweeps: int = 10,
    tol: float = 1e-8,
    n_workers: Optional[int] = None,
    seeds: Optional[list] = None,
) -> MultiStartDMRGResult:
    """Parallel multi-start two-site MPO-DMRG.

    Runs :func:`dmrg_sweep_mpo_2site` from *n_seeds* independent random
    initial states in parallel via ``ProcessPoolExecutor``, then returns the
    run with the lowest final energy.

    Because two-site DMRG can converge to different local minima depending
    on the starting state, multiple seeds substantially improve the chance of
    finding the true ground state — at zero extra hardware cost on any
    multi-core CPU.

    Parameters
    ----------
    mpo      : Hamiltonian as MPO.
    n_seeds  : Number of independent random starting states to try.
    chi      : Bond-dimension cap for each individual DMRG run.
    chi_init : Initial random MPS bond dimension (small so the 2-site SVD
               can freely grow bonds during optimisation).
    n_sweeps : Maximum sweeps per run.
    tol      : Energy convergence threshold per run.
    n_workers: Worker processes.  ``None`` uses ``os.cpu_count()``.  Pass
               ``1`` to run sequentially (no subprocess overhead).
    seeds    : Explicit seed list; overrides ``n_seeds`` when provided.

    Returns
    -------
    :class:`MultiStartDMRGResult`
    """
    if seeds is None:
        seeds = list(range(n_seeds))

    packed = [
        (mpo.tensors, seed, chi_init, chi, n_sweeps, tol)
        for seed in seeds
    ]

    effective_workers = n_workers if n_workers is not None else (os.cpu_count() or 1)

    if effective_workers == 1 or len(seeds) == 1:
        results = [_dmrg_worker(p) for p in packed]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_dmrg_worker, packed))

    final_energies = [r.energies[-1] for r in results]
    best_idx = int(np.argmin(final_energies))

    return MultiStartDMRGResult(
        best=results[best_idx],
        all_energies=final_energies,
        seeds_used=list(seeds),
        best_seed=seeds[best_idx],
        n_workers=effective_workers,
    )
