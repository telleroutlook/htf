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

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .mps import MPS, mps_inner


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
