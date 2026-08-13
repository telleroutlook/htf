"""HTF §4-G — Symmetric / gauge-invariant tensors (U(1) block-sparse).

Many physical Hamiltonians conserve a U(1) charge (particle number, total
spin-z, etc.).  Tensors that respect this symmetry are block-diagonal in
the charge basis: the only non-zero elements connect configurations whose
total incoming charge equals their total outgoing charge.

Representing such tensors in block-sparse form reduces both memory and
contraction cost by the number of charge sectors.

Provides
--------
* ``ChargedBasis``       — list of (dim, charge) sector pairs for one wire.
* ``check_u1_invariance``— verify a dense tensor is U(1)-invariant.
* ``project_to_u1``     — zero out elements that violate charge conservation.
* ``u1_blocks``         — decompose a U(1)-invariant tensor into charge blocks.
* ``BlockSparseTensor`` — lightweight container storing only non-zero blocks.
* ``block_sparse_matmul``— block-wise matrix multiplication (preserves U(1)).
* ``spin_half_basis``   — standard S=½ basis with charges ±1.
* ``number_basis``      — occupation-number basis with charges 0,1.

Honest scope [研究]
-------------------
* Only abelian U(1) symmetry is implemented; non-abelian SU(N) requires
  Clebsch–Gordan coefficients and is ``[研究]``.
* Block-sparse contraction for general tensor networks is ``[研究]``.
* This module works on dense numpy arrays; a production block-sparse
  backend (e.g. using TensorNetwork or BlockSparse libraries) is
  ``[工程]`` but outside the current scope.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ────────────────────── charge basis ──────────────────────────────────────

@dataclass
class ChargedBasis:
    """A single wire's basis decomposed into U(1) charge sectors.

    Attributes
    ----------
    sectors : list of ``(size, charge)`` pairs where ``size`` is the
              dimension of the sector and ``charge`` is an integer charge.
    """
    sectors: list[tuple[int, int]]

    @property
    def dim(self) -> int:
        """Total dimension of the space."""
        return sum(s for s, _ in self.sectors)

    @property
    def charge_array(self) -> np.ndarray:
        """Per-basis-index charge array of shape (dim,)."""
        charges = []
        for size, q in self.sectors:
            charges.extend([q] * size)
        return np.array(charges, dtype=int)

    def __repr__(self) -> str:
        return f"ChargedBasis({self.sectors})"


# ─────── standard bases ───────────────────────────────────────────────────

def spin_half_basis() -> ChargedBasis:
    """S=½ basis: |↑⟩ with charge +1, |↓⟩ with charge −1."""
    return ChargedBasis([(1, 1), (1, -1)])


def number_basis() -> ChargedBasis:
    """Occupation-number basis: |0⟩ with charge 0, |1⟩ with charge 1."""
    return ChargedBasis([(1, 0), (1, 1)])


# ────────────────────── invariance checks ────────────────────────────────

def check_u1_invariance(
    tensor:      np.ndarray,
    dom_bases:   list[ChargedBasis],
    cod_bases:   list[ChargedBasis],
    tol:         float = 1e-10,
) -> dict:
    """Check whether a dense tensor respects U(1) charge conservation.

    A tensor is U(1)-invariant if every element ``T[i₀,…,iₘ₋₁, j₀,…,jₙ₋₁]``
    is zero unless the total incoming charge equals the total outgoing charge:

        Σ cod_charges[k][iₖ] = Σ dom_charges[k][jₖ]

    Parameters
    ----------
    tensor     : dense ndarray; shape must match the concatenated dimensions
                 of cod_bases then dom_bases (following HTF convention).
    dom_bases  : list of :class:`ChargedBasis` for the domain wires.
    cod_bases  : list of :class:`ChargedBasis` for the codomain wires.
    tol        : tolerance for "near-zero" elements.

    Returns
    -------
    dict with keys:
      ``is_invariant``    — bool,
      ``n_violations``    — count of non-zero elements that violate conservation,
      ``max_violation``   — largest absolute value of a violating element,
      ``charge_sectors``  — list of charge sectors (total Q) that are non-zero.
    """
    T = np.asarray(tensor)
    dom_charges_list = [b.charge_array for b in dom_bases]
    cod_charges_list = [b.charge_array for b in cod_bases]

    # Iterate over all multi-indices
    shape_cod = tuple(b.dim for b in cod_bases)
    shape_dom = tuple(b.dim for b in dom_bases)
    n_cod = len(cod_bases)
    n_dom = len(dom_bases)

    T_flat = T.reshape(shape_cod + shape_dom)

    n_violations = 0
    max_violation = 0.0
    seen_sectors: set[int] = set()

    it = np.nditer(T_flat, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        val = float(abs(it[0]))
        cod_idx = idx[:n_cod]
        dom_idx = idx[n_cod:]
        Q_cod = sum(cod_charges_list[k][cod_idx[k]] for k in range(n_cod))
        Q_dom = sum(dom_charges_list[k][dom_idx[k]] for k in range(n_dom))
        if val > tol:
            if Q_cod != Q_dom:
                n_violations += 1
                max_violation = max(max_violation, val)
            else:
                seen_sectors.add(int(Q_cod))
        it.iternext()

    return {
        "is_invariant":   n_violations == 0,
        "n_violations":   n_violations,
        "max_violation":  max_violation,
        "charge_sectors": sorted(seen_sectors),
    }


def project_to_u1(
    tensor:    np.ndarray,
    dom_bases: list[ChargedBasis],
    cod_bases: list[ChargedBasis],
) -> np.ndarray:
    """Zero out tensor elements that violate U(1) charge conservation.

    Parameters
    ----------
    tensor     : dense ndarray to project.
    dom_bases  : domain :class:`ChargedBasis` list.
    cod_bases  : codomain :class:`ChargedBasis` list.

    Returns
    -------
    New ndarray with charge-violating elements set to zero.
    """
    T = np.array(tensor, copy=True)
    dom_charges_list = [b.charge_array for b in dom_bases]
    cod_charges_list = [b.charge_array for b in cod_bases]

    shape_cod = tuple(b.dim for b in cod_bases)
    shape_dom = tuple(b.dim for b in dom_bases)
    n_cod = len(cod_bases)
    n_dom = len(dom_bases)
    T_flat = T.reshape(shape_cod + shape_dom)

    it = np.nditer(T_flat, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        cod_idx = idx[:n_cod]
        dom_idx = idx[n_cod:]
        Q_cod = sum(cod_charges_list[k][cod_idx[k]] for k in range(n_cod))
        Q_dom = sum(dom_charges_list[k][dom_idx[k]] for k in range(n_dom))
        if Q_cod != Q_dom:
            it[0][...] = 0.0
        it.iternext()
    return T_flat.reshape(tensor.shape)


# ────────────────────── block decomposition ──────────────────────────────

@dataclass
class BlockSparseTensor:
    """A U(1)-invariant tensor stored as a dict of charge-sector blocks.

    Attributes
    ----------
    blocks       : dict mapping ``charge → np.ndarray`` (the 2D block for
                   that total charge sector).
    dom_bases    : list of :class:`ChargedBasis` for domain wires.
    cod_bases    : list of :class:`ChargedBasis` for codomain wires.
    dom_shape    : shape of the full domain multi-index space.
    cod_shape    : shape of the full codomain multi-index space.
    """
    blocks:    dict[int, np.ndarray]
    dom_bases: list[ChargedBasis]
    cod_bases: list[ChargedBasis]
    dom_shape: tuple[int, ...]
    cod_shape: tuple[int, ...]

    @property
    def total_dim_dom(self) -> int:
        d = 1
        for s in self.dom_shape:
            d *= s
        return d

    @property
    def total_dim_cod(self) -> int:
        d = 1
        for s in self.cod_shape:
            d *= s
        return d

    def to_dense(self) -> np.ndarray:
        """Reconstruct the full dense tensor from blocks."""
        d_cod = self.total_dim_cod
        d_dom = self.total_dim_dom
        T = np.zeros((d_cod, d_dom), dtype=complex)
        dom_charges = _flat_charges(self.dom_bases)
        cod_charges = _flat_charges(self.cod_bases)
        for Q, block in self.blocks.items():
            cod_idx = np.where(cod_charges == Q)[0]
            dom_idx = np.where(dom_charges == Q)[0]
            for ic, i in enumerate(cod_idx):
                for jc, j in enumerate(dom_idx):
                    T[i, j] = block[ic, jc]
        return T.reshape(self.cod_shape + self.dom_shape)

    def nnz(self) -> int:
        """Total number of stored (non-zero block) elements."""
        return sum(b.size for b in self.blocks.values())

    def sparsity(self) -> float:
        """Fraction of the total matrix that is represented by blocks."""
        total = self.total_dim_cod * self.total_dim_dom
        return self.nnz() / total if total > 0 else 0.0


def _flat_charges(bases: list[ChargedBasis]) -> np.ndarray:
    """Flatten a list of bases to a combined charge array (tensor product)."""
    if not bases:
        return np.array([0], dtype=int)
    charges = [b.charge_array for b in bases]
    # Take the tensor product of charges (sum for each combined index)
    result = charges[0]
    for c in charges[1:]:
        result = (result[:, None] + c[None, :]).ravel()
    return result


def u1_blocks(
    tensor:    np.ndarray,
    dom_bases: list[ChargedBasis],
    cod_bases: list[ChargedBasis],
) -> BlockSparseTensor:
    """Decompose a U(1)-invariant tensor into charge-sector blocks.

    The tensor is reshaped as (dim_cod, dim_dom) and split along rows
    (codomain) and columns (domain) by their total charge.

    Parameters
    ----------
    tensor     : dense ndarray (U(1)-invariant; non-invariant elements ignored).
    dom_bases  : list of :class:`ChargedBasis` for domain wires.
    cod_bases  : list of :class:`ChargedBasis` for codomain wires.

    Returns
    -------
    :class:`BlockSparseTensor` with one block per active charge sector.
    """
    dom_shape = tuple(b.dim for b in dom_bases)
    cod_shape = tuple(b.dim for b in cod_bases)
    d_dom = 1
    for s in dom_shape:
        d_dom *= s
    d_cod = 1
    for s in cod_shape:
        d_cod *= s

    T = np.asarray(tensor, dtype=complex).reshape(d_cod, d_dom)
    dom_charges = _flat_charges(dom_bases)
    cod_charges = _flat_charges(cod_bases)

    # Collect all charge sectors
    all_Q = set(dom_charges.tolist()) & set(cod_charges.tolist())
    blocks: dict[int, np.ndarray] = {}
    for Q in sorted(all_Q):
        ri = np.where(cod_charges == Q)[0]
        ci = np.where(dom_charges == Q)[0]
        if len(ri) > 0 and len(ci) > 0:
            block = T[np.ix_(ri, ci)]
            if np.any(np.abs(block) > 1e-15):
                blocks[Q] = block.copy()

    return BlockSparseTensor(
        blocks=blocks,
        dom_bases=dom_bases,
        cod_bases=cod_bases,
        dom_shape=dom_shape,
        cod_shape=cod_shape,
    )


# ────────────────────── block matmul ─────────────────────────────────────

def block_sparse_matmul(
    A: BlockSparseTensor,
    B: BlockSparseTensor,
) -> BlockSparseTensor:
    """Block-wise matrix multiplication A @ B.

    Both tensors must share the same intermediate bases (A's domain ==
    B's codomain in terms of charges).

    Returns
    -------
    :class:`BlockSparseTensor` representing the composed map.
    """
    result_blocks: dict[int, np.ndarray] = {}
    all_Q = set(A.blocks.keys()) | set(B.blocks.keys())
    for Q in all_Q:
        if Q in A.blocks and Q in B.blocks:
            result_blocks[Q] = A.blocks[Q] @ B.blocks[Q]
    return BlockSparseTensor(
        blocks=result_blocks,
        dom_bases=B.dom_bases,
        cod_bases=A.cod_bases,
        dom_shape=B.dom_shape,
        cod_shape=A.cod_shape,
    )
