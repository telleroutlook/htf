"""HTF Phase 3 — Binary MERA tensor network.

Implements a binary MERA (Multi-scale Entanglement Renormalization Ansatz)
on N physical sites (N must be a power of 2) with bond dimension ``chi``.

Each coarse-graining layer has:
* **Disentanglers** – one real orthogonal matrix per pair of sites
  (shape ``(chi, chi, chi, chi)`` viewed as ``(chi², chi²)`` unitary).
* **Isometries** – one isometric matrix per pair of sites
  (shape ``(chi, chi, chi)`` viewed as ``(chi, chi²)``; ``M @ M.T = I``).

The state vector is computed by **top-down contraction**: start from the
top-level state and iteratively expand virtual sites to physical sites
by applying the adjoint of each layer.

Honest scope
------------
* Float evaluation only (Phase 3). Certified energy bound available via
  :func:`htf.variational.variational_bound`.
* Bond-dimension truncation error is **not** certified here (Phase 4).
* Practical state-vector limit: ``N ≤ 16`` (RAM = ``chi^N``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .structure import enforce_isometry, enforce_unitary


# ──────────────────── low-level contraction helpers ───────────────

def _apply_isometry_adjoint(
    state: np.ndarray, W: np.ndarray, axis: int, chi: int
) -> np.ndarray:
    """Apply ``W†`` to ``axis`` of *state*, expanding one chi-dim axis → two.

    ``W`` has shape ``(chi, chi, chi)`` = ``(out, in1, in2)``.
    ``W†[in1, in2, out] = W[out, in1, in2]`` (no conjugation for real W).
    Returns a tensor with ``state.ndim + 1`` axes.
    """
    # tensordot(W, state, ([0], [axis])) contracts W's 'out' axis with state's axis.
    # Result shape: (chi, chi, *remaining_state_axes)  = (in1, in2, ...)
    result = np.tensordot(W, state, axes=([0], [axis]))
    n_result = result.ndim  # state.ndim + 1
    # Move (in1, in2) from positions (0, 1) to (axis, axis+1).
    perm = list(range(2, axis + 2)) + [0, 1] + list(range(axis + 2, n_result))
    return np.transpose(result, perm)


def _apply_unitary_adjoint(
    state: np.ndarray,
    U: np.ndarray,
    site1: int,
    site2: int,
    chi: int,
) -> np.ndarray:
    """Apply ``U†`` to axes ``(site1, site2)`` of *state*.

    ``U`` has shape ``(d1, d2, d1, d2)`` = ``(out1, out2, in1, in2)``.
    For real orthogonal U: ``U† = U.T``.  Dimensions are derived from
    ``U.shape`` so this works for any local dimension (not just ``chi``).
    """
    d1, d2 = U.shape[0], U.shape[1]
    U_mat = U.reshape(d1 * d2, d1 * d2)
    Ut_4d = U_mat.T.reshape(d1, d2, d1, d2)   # (in1, in2, out1, out2)
    result = np.tensordot(Ut_4d, state, axes=([2, 3], [site1, site2]))
    # result shape: (chi, chi, *all_other_state_axes)  = (in1, in2, ...)
    n_axes = state.ndim
    perm, other_idx = [], 2
    for i in range(n_axes):
        if i == site1:
            perm.append(0)
        elif i == site2:
            perm.append(1)
        else:
            perm.append(other_idx)
            other_idx += 1
    return np.transpose(result, perm)


# ─────────────────────────── data classes ─────────────────────────

@dataclass
class MERALayer:
    """One coarse-graining layer: ``n_in`` sites → ``n_in // 2`` sites.

    Attributes
    ----------
    n_in         : number of input sites.
    chi          : bond / physical dimension.
    disentanglers: ``n_in // 2`` real orthogonal tensors, each ``(chi,chi,chi,chi)``.
    isometries   : ``n_in // 2`` isometric tensors, each ``(chi,chi,chi)``.
    """

    n_in: int
    chi: int
    disentanglers: List[np.ndarray]
    isometries: List[np.ndarray]

    @property
    def n_out(self) -> int:
        return self.n_in // 2

    def enforce_constraints(self) -> None:
        """Project all tensors to unitary / isometric in-place (SVD retraction)."""
        self.disentanglers = [enforce_unitary(d) for d in self.disentanglers]
        self.isometries = [enforce_isometry(w) for w in self.isometries]


@dataclass
class MERA:
    """Binary MERA on ``n_sites`` physical sites with bond dimension ``chi``.

    ``layers`` are ordered **bottom-to-top** (``layers[0]`` acts on physical
    sites; ``layers[-1]`` feeds into ``top``).
    ``top`` is the normalised top-level state vector of shape ``(chi,)``.
    """

    n_sites: int
    chi: int
    layers: List[MERALayer]
    top: np.ndarray

    # ── state vector ──────────────────────────────────────────────

    def state_vector(self) -> np.ndarray:
        """Compute the physical state by top-down contraction.

        Returns a flattened real array of shape ``(chi^n_sites,)``.
        Practical: only call for ``n_sites ≤ 16``.
        """
        chi = self.chi
        state = self.top.copy()          # (chi,) — 1 virtual site

        for layer in reversed(self.layers):
            n_virtual = state.ndim       # = number of virtual sites at this level

            # Step 1: expand each virtual site → 2 physical/virtual sites
            for k in range(n_virtual):
                state = _apply_isometry_adjoint(
                    state, layer.isometries[k], 2 * k, chi
                )
            # state now has shape (chi,)^{2 * n_virtual}

            # Step 2: apply disentangler adjoints to each adjacent pair
            for k in range(n_virtual):
                state = _apply_unitary_adjoint(
                    state, layer.disentanglers[k], 2 * k, 2 * k + 1, chi
                )

        return state.reshape(-1)

    # ── constraint enforcement ────────────────────────────────────

    def enforce_constraints(self) -> None:
        """Project every layer in-place + normalise the top state."""
        for layer in self.layers:
            layer.enforce_constraints()
        norm = float(np.linalg.norm(self.top))
        if norm > 1e-10:
            self.top = self.top / norm

    # ── parameter serialisation ───────────────────────────────────

    def to_flat_params(self) -> np.ndarray:
        """Flatten all MERA tensors into a single 1-D array."""
        parts: list[np.ndarray] = []
        for layer in self.layers:
            for d in layer.disentanglers:
                parts.append(d.reshape(-1))
            for w in layer.isometries:
                parts.append(w.reshape(-1))
        parts.append(self.top)
        return np.concatenate(parts)

    def from_flat_params(self, params: np.ndarray) -> "MERA":
        """Reconstruct a MERA with the same structure from a flat parameter vector."""
        chi = self.chi
        offset = 0
        new_layers: list[MERALayer] = []
        for layer in self.layers:
            new_dis = []
            for _ in layer.disentanglers:
                size = chi ** 4
                new_dis.append(params[offset: offset + size].reshape(chi, chi, chi, chi))
                offset += size
            new_iso = []
            for _ in layer.isometries:
                size = chi ** 3
                new_iso.append(params[offset: offset + size].reshape(chi, chi, chi))
                offset += size
            new_layers.append(MERALayer(layer.n_in, chi, new_dis, new_iso))
        new_top = params[offset: offset + chi].copy()
        return MERA(self.n_sites, chi, new_layers, new_top)


# ──────────────────────── factory ─────────────────────────────────

def random_mera(
    n_sites: int,
    chi: int = 2,
    seed: int = 0,
    with_disentanglers: bool = True,
) -> MERA:
    """Create a random MERA with isometry constraints enforced by construction.

    Parameters
    ----------
    n_sites            : number of physical sites (must be a power of 2, ≥ 2).
    chi                : bond dimension = physical site dimension.
    seed               : RNG seed for reproducibility.
    with_disentanglers : if ``False``, disentanglers are identity (tree TN).
    """
    if n_sites < 2 or (n_sites & (n_sites - 1)) != 0:
        raise ValueError(f"n_sites must be a power of 2, got {n_sites}")
    rng = np.random.default_rng(seed)
    n_layers = int(np.log2(n_sites))

    layers: list[MERALayer] = []
    n_in = n_sites
    for _ in range(n_layers):
        n_pairs = n_in // 2

        if with_disentanglers:
            dis = []
            for _ in range(n_pairs):
                raw = rng.standard_normal((chi ** 2, chi ** 2))
                U, _, Vt = np.linalg.svd(raw, full_matrices=True)
                dis.append((U @ Vt).reshape(chi, chi, chi, chi))
        else:
            eye4 = np.eye(chi ** 2).reshape(chi, chi, chi, chi)
            dis = [eye4.copy() for _ in range(n_pairs)]

        iso = []
        for _ in range(n_pairs):
            raw = rng.standard_normal((chi, chi ** 2))
            U, _, Vt = np.linalg.svd(raw, full_matrices=False)
            iso.append((U @ Vt).reshape(chi, chi, chi))

        layers.append(MERALayer(n_in=n_in, chi=chi, disentanglers=dis, isometries=iso))
        n_in = n_pairs

    top = rng.standard_normal(chi)
    top /= np.linalg.norm(top)
    return MERA(n_sites=n_sites, chi=chi, layers=layers, top=top)
