"""Matrix Product State (MPS) data structure and basic operations.

An MPS for n sites with physical dimension d and bond dimension chi is a
sequence of rank-3 tensors  A[i]  of shape  (chi_l, d, chi_r), where
chi_l=1 for i=0 and chi_r=1 for i=n-1.  The state it represents is:

    |ψ⟩ = Σ_{s_0…s_{n-1}} A[0][:,s_0,:] @ A[1][:,s_1,:] @ … @ A[n-1][:,s_{n-1},:]
              |s_0 s_1 … s_{n-1}⟩

All computations are float64 unless the input tensors are complex.

Honest scope [工程]
-------------------
* SVD-based operations work with numpy.  GPU and JAX acceleration require
  installing the optional 'accel' extra.
* TEBD Trotter error is O(dt²) per step; not certified.  For certified
  bounds on E0 use temple_lanczos or variational_bound.
* Bond-dimension truncation is optimal in Frobenius norm (Schmidt
  decomposition) but the discarded weight is NOT propagated to a
  certificate — the truncation error is reported, not bounded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.linalg


@dataclass
class MPS:
    """A Matrix Product State.

    tensors[i] has shape (chi_l, d, chi_r):
      * tensors[0]  has chi_l = 1
      * tensors[-1] has chi_r = 1
    """
    tensors: list[np.ndarray]

    @property
    def n_sites(self) -> int:
        return len(self.tensors)

    @property
    def phys_dim(self) -> int:
        return self.tensors[0].shape[1]

    @property
    def bond_dims(self) -> list[int]:
        """Bond dimensions: [chi_0_1, chi_1_2, …, chi_{n-2}_{n-1}]."""
        return [t.shape[2] for t in self.tensors[:-1]]

    @property
    def max_bond(self) -> int:
        bonds = self.bond_dims
        return max(bonds) if bonds else 1

    def copy(self) -> "MPS":
        return MPS([t.copy() for t in self.tensors])


def random_mps(
    n: int,
    d: int,
    chi: int,
    seed: Optional[int] = None,
    dtype=np.float64,
) -> MPS:
    """Random MPS in left-canonical form with bond dimension chi."""
    rng = np.random.default_rng(seed)
    tensors: list[np.ndarray] = []
    for i in range(n):
        chi_l = 1 if i == 0 else min(chi, d**i, d**(n - i))
        chi_r = 1 if i == n - 1 else min(chi, d**(i + 1), d**(n - i - 1))
        A = rng.standard_normal((chi_l, d, chi_r)).astype(dtype)
        tensors.append(A)
    mps = MPS(tensors)
    return _left_canonicalise(mps)


def mps_from_state(
    psi: np.ndarray,
    d: int,
    chi: Optional[int] = None,
) -> MPS:
    """Convert a dense state vector to MPS via successive SVD.

    Parameters
    ----------
    psi:  state vector of length d^n.
    d:    physical dimension per site.
    chi:  maximum bond dimension; None = no truncation.

    Returns
    -------
    MPS in left-canonical form.
    """
    n = round(math.log(len(psi), d))
    if d**n != len(psi):
        raise ValueError(f"State length {len(psi)} is not d^n for d={d}")

    tensors: list[np.ndarray] = []
    M = psi.reshape(1, -1)          # (1, d^n)
    for i in range(n - 1):
        chi_l = M.shape[0]
        M = M.reshape(chi_l * d, -1)   # (chi_l*d, d^{n-i-1})
        U, s, Vh = scipy.linalg.svd(M, full_matrices=False)
        if chi is not None:
            keep = min(chi, len(s))
            U  = U[:, :keep]
            s  = s[:keep]
            Vh = Vh[:keep, :]
        A = U.reshape(chi_l, d, -1)    # (chi_l, d, new_chi)
        tensors.append(A)
        M = np.diag(s) @ Vh            # (new_chi, …)

    chi_l = M.shape[0]
    tensors.append(M.reshape(chi_l, d, 1))
    return MPS(tensors)


def mps_to_state(mps: MPS) -> np.ndarray:
    """Contract MPS to a dense state vector of length d^n."""
    result = mps.tensors[0]           # (1, d, chi_r)
    result = result.reshape(mps.phys_dim, -1)  # (d, chi_r)
    for A in mps.tensors[1:]:
        chi_m, d, chi_r = A.shape
        # result: (d^i, chi_m) × A: (chi_m, d, chi_r) → (d^{i+1}, chi_r)
        result = np.tensordot(result, A, axes=([1], [0]))
        # result now: (d^i, d, chi_r) → reshape
        result = result.reshape(-1, chi_r)
    return result.ravel()


def mps_inner(bra: MPS, ket: MPS) -> complex:
    """Compute ⟨bra|ket⟩ using the transfer matrix method.

    Complexity O(n · χ³ · d).
    """
    n = bra.n_sites
    if n != ket.n_sites:
        raise ValueError("MPS site counts differ")
    # Transfer matrix: start from left boundary
    # T has shape (chi_bra, chi_ket)
    T = np.ones((1, 1), dtype=complex)
    for i in range(n):
        A = bra.tensors[i]   # (chi_l_b, d, chi_r_b)
        B = ket.tensors[i]   # (chi_l_k, d, chi_r_k)
        # T_new[α', β'] = Σ_{α,β,s} T[α,β] * conj(A[α,s,α']) * B[β,s,β']
        # Contract T with A* over α:  (chi_r_b, chi_ket)  ... use einsum
        T = np.einsum("ab, asc, bsd -> cd", T, A.conj(), B)
    return complex(T[0, 0])


def mps_norm(mps: MPS) -> float:
    """Euclidean norm ‖|ψ⟩‖."""
    return math.sqrt(abs(mps_inner(mps, mps)))


def mps_normalise(mps: MPS) -> MPS:
    """Return a copy normalised to unit norm (divides last tensor only)."""
    nrm = mps_norm(mps)
    if nrm == 0.0:
        raise ValueError("Cannot normalise zero MPS")
    tensors = [t.copy() for t in mps.tensors]
    tensors[-1] = tensors[-1] / nrm
    return MPS(tensors)


def mps_add(psi: MPS, phi: MPS) -> MPS:
    """Sum |ψ⟩ + |φ⟩ by direct-sum of bond tensors.

    The result has bond dimension chi_psi + chi_phi (uncompressed).
    """
    n = psi.n_sites
    if n != phi.n_sites:
        raise ValueError("MPS site counts differ")
    tensors = []
    for i in range(n):
        A = psi.tensors[i]   # (a_l, d, a_r)
        B = phi.tensors[i]   # (b_l, d, b_r)
        a_l, d, a_r = A.shape
        b_l, _, b_r = B.shape
        if i == 0:
            # Left boundary: (1, d, a_r + b_r)
            C = np.zeros((1, d, a_r + b_r), dtype=np.result_type(A, B))
            C[0, :, :a_r] = A[0]
            C[0, :, a_r:] = B[0]
        elif i == n - 1:
            # Right boundary: (a_l + b_l, d, 1)
            C = np.zeros((a_l + b_l, d, 1), dtype=np.result_type(A, B))
            C[:a_l, :, 0] = A[:, :, 0]
            C[a_l:, :, 0] = B[:, :, 0]
        else:
            C = np.zeros((a_l + b_l, d, a_r + b_r), dtype=np.result_type(A, B))
            C[:a_l, :, :a_r] = A
            C[a_l:, :, a_r:] = B
        tensors.append(C)
    return MPS(tensors)


def mps_truncate(mps: MPS, chi: int) -> tuple["MPS", float]:
    """SVD-compress MPS to bond dimension chi.

    Returns (compressed_mps, discarded_weight).  The discarded weight is
    the sum of squared singular values that were dropped; it is NOT a
    certified error bound.
    """
    tensors = [t.copy() for t in mps.tensors]
    discarded = 0.0
    # Right-to-left SVD sweep
    for i in range(len(tensors) - 1, 0, -1):
        A = tensors[i]
        chi_l, d, chi_r = A.shape
        M = A.reshape(chi_l, d * chi_r)
        U, s, Vh = scipy.linalg.svd(M, full_matrices=False)
        keep = min(chi, len(s))
        discarded += float(np.sum(s[keep:] ** 2))
        Vh = Vh[:keep, :]
        s  = s[:keep]
        tensors[i] = Vh.reshape(keep, d, chi_r)
        # Absorb U * diag(s) into the left tensor
        tensors[i - 1] = np.tensordot(tensors[i - 1], U[:, :keep] * s[None, :],
                                       axes=([-1], [0]))
        # tensors[i-1] was (chi_ll, d, chi_l); now (chi_ll, d, keep)
        # but tensordot gives (chi_ll, d, keep) — correct shape
    return MPS(tensors), discarded


def mps_expectation(
    mps: MPS,
    operators: list[tuple[int, np.ndarray]],
) -> complex:
    """Compute ⟨ψ| O_1 ⊗ O_2 ⊗ … |ψ⟩ for a product of local operators.

    Parameters
    ----------
    mps:       the state.
    operators: list of (site_index, matrix) pairs; site_index in 0..n-1.
               Matrix must be (d, d).  Sites not listed get identity.
    """
    op_dict: dict[int, np.ndarray] = {}
    for site, mat in operators:
        op_dict[site] = mat
    n = mps.n_sites
    T = np.ones((1, 1), dtype=complex)
    for i in range(n):
        A = mps.tensors[i]       # (chi_l, d, chi_r)
        if i in op_dict:
            # Apply operator: A_op[α, s, β] = Σ_t op[s,t] * A[α,t,β]
            A_op = np.einsum("st, atb -> asb", op_dict[i], A)
        else:
            A_op = A
        T = np.einsum("ab, asc, bsd -> cd", T, A.conj(), A_op)
    return complex(T[0, 0])


def mps_apply_gate(
    mps: MPS,
    gate: np.ndarray,
    sites: list[int],
    chi: Optional[int] = None,
) -> tuple["MPS", float]:
    """Apply a 1- or 2-site gate to an MPS with optional SVD truncation.

    Parameters
    ----------
    mps:   the input MPS.
    gate:  for 1 site, shape (d, d); for 2 sites, shape (d, d, d, d) with
           indices (s0', s1', s0, s1).
    sites: list of one or two adjacent site indices.
    chi:   maximum bond dimension after SVD truncation.

    Returns
    -------
    (new_mps, discarded_weight)
    """
    tensors = [t.copy() for t in mps.tensors]
    discarded = 0.0

    if len(sites) == 1:
        i = sites[0]
        # gate shape (d, d); apply: A_new[α,s',β] = Σ_s gate[s',s] * A[α,s,β]
        tensors[i] = np.einsum("st, atb -> asb", gate, tensors[i])

    elif len(sites) == 2:
        i, j = sites
        if j != i + 1:
            raise ValueError("2-site gate must be applied to adjacent sites")
        A = tensors[i]   # (chi_l, d, chi_m)
        B = tensors[j]   # (chi_m, d, chi_r)
        chi_l, d, chi_m = A.shape
        _, _, chi_r      = B.shape
        # Contract A-B into a rank-4 tensor, apply gate, then SVD
        AB = np.einsum("asb, btc -> astc", A, B)          # (chi_l, d, d, chi_r)
        # gate: (d, d, d, d) → (s'0, s'1, s0, s1)
        AB_g = np.einsum("stpq, apqc -> astc", gate, AB)  # (chi_l, d, d, chi_r)
        # Reshape for SVD; pre-scale to avoid SVD convergence failure
        M = AB_g.reshape(chi_l * d, d * chi_r)
        scale = float(np.max(np.abs(M)))
        if scale > 0:
            M = M / scale
        U, s, Vh = scipy.linalg.svd(M, full_matrices=False)
        s = s * scale   # restore scale in singular values
        keep = len(s)
        if chi is not None:
            keep = min(chi, keep)
        discarded = float(np.sum(s[keep:] ** 2))
        s_trunc = s[:keep]
        tensors[i] = U[:, :keep].reshape(chi_l, d, keep)
        tensors[j] = (np.diag(s_trunc) @ Vh[:keep, :]).reshape(keep, d, chi_r)
    else:
        raise ValueError("Only 1- or 2-site gates supported")

    return MPS(tensors), discarded


def _left_canonicalise(mps: MPS) -> MPS:
    """Left-canonicalise an MPS in-place and return it."""
    tensors = [t.copy() for t in mps.tensors]
    for i in range(len(tensors) - 1):
        A = tensors[i]
        chi_l, d, chi_r = A.shape
        M = A.reshape(chi_l * d, chi_r)
        Q, R = np.linalg.qr(M)
        tensors[i] = Q.reshape(chi_l, d, -1)
        tensors[i + 1] = np.einsum("ab, bsc -> asc", R, tensors[i + 1])
    return MPS(tensors)
