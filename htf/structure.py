"""Structure verification for proof-carrying diagrams.

Phase 3 Track B: machine-check physical structure properties so that
"violations cannot pass the check." Status is derived from computation,
never self-declared — no PASS without a replayable defect value.

Checked properties
------------------
* **Isometry** : ``M @ M.T = I_cod``  (left inverse; cod_size ≤ dom_size).
* **Unitarity** : ``M @ M.T = M.T @ M = I``  (square matrix required).
* **Reflection positivity** (OS-positivity proxy): Gram-matrix min-eig ≥ 0.

The ``enforce_*`` helpers project a tensor to the nearest isometric /
unitary form via SVD polar decomposition — useful for initialising MERA
layers or repairing numerical drift.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .functor import TensorFunctor
from .topology import Box, dims


@dataclass
class StructureReport:
    """Result of one structure-verification check.

    ``defect`` is the max-norm deviation from the ideal (lower = better);
    ``passed`` is ``defect ≤ tolerance``.  Use :meth:`__str__` for a
    human-readable summary.
    """

    property_name: str
    passed: bool
    defect: float
    tolerance: float
    notes: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        msg = (
            f"[{status}] {self.property_name}: "
            f"defect={self.defect:.3e}, tol={self.tolerance:.3e}"
        )
        return msg + (f" — {self.notes}" if self.notes else "")


# ─────────────────────── matrix helpers ───────────────────────────

def _as_matrix(tensor: np.ndarray, n_cod: int | None = None) -> np.ndarray:
    """Reshape *tensor* to a 2-D matrix ``(cod_size, dom_size)``.

    *n_cod* is the number of leading (cod) axes.  If ``None``, the axes
    are split in half (standard HTF convention for symmetric boxes).
    """
    ndim = tensor.ndim
    if n_cod is None:
        n_cod = ndim // 2
    cod_shape = tensor.shape[:n_cod]
    dom_shape = tensor.shape[n_cod:]
    rows = int(np.prod(cod_shape)) if cod_shape else 1
    cols = int(np.prod(dom_shape)) if dom_shape else 1
    return tensor.reshape(rows, cols)


# ──────────────────── defect computations ─────────────────────────

def isometry_defect(tensor: np.ndarray, n_cod: int | None = None) -> float:
    """``‖M @ M.T − I_cod‖_max``  (0 for a perfect isometry)."""
    M = _as_matrix(tensor, n_cod)
    rows = M.shape[0]
    return float(np.abs(M @ M.T - np.eye(rows)).max())


def unitary_defect(tensor: np.ndarray, n_cod: int | None = None) -> float:
    """``max(‖M M.T − I‖, ‖M.T M − I‖)_max``  (0 for a perfect unitary)."""
    M = _as_matrix(tensor, n_cod)
    rows, cols = M.shape
    left = float(np.abs(M @ M.T - np.eye(rows)).max())
    right = float(np.abs(M.T @ M - np.eye(cols)).max())
    return max(left, right)


# ───────────────────── check functions ────────────────────────────

def check_isometry(
    tensor: np.ndarray,
    tol: float = 1e-10,
    n_cod: int | None = None,
) -> StructureReport:
    """Check ``M @ M.T = I_cod`` (isometry / left-inverse)."""
    defect = isometry_defect(tensor, n_cod)
    return StructureReport(
        property_name="isometry",
        passed=defect <= tol,
        defect=defect,
        tolerance=tol,
    )


def check_unitary(
    tensor: np.ndarray,
    tol: float = 1e-10,
    n_cod: int | None = None,
) -> StructureReport:
    """Check ``M @ M.T = M.T @ M = I`` (unitarity)."""
    defect = unitary_defect(tensor, n_cod)
    return StructureReport(
        property_name="unitary",
        passed=defect <= tol,
        defect=defect,
        tolerance=tol,
    )


def check_box_isometry(
    box: Box,
    functor: TensorFunctor,
    tol: float = 1e-10,
) -> StructureReport:
    """Verify isometry for *box* using its tensor from *functor*."""
    tensor = functor.tensor(box)
    n_cod = len(box.cod)
    report = check_isometry(tensor, tol, n_cod)
    report.notes = f"box={box.name!r}, cod={dims(box.cod)}, dom={dims(box.dom)}"
    return report


def check_box_unitary(
    box: Box,
    functor: TensorFunctor,
    tol: float = 1e-10,
) -> StructureReport:
    """Verify unitarity for *box* using its tensor from *functor*."""
    tensor = functor.tensor(box)
    n_cod = len(box.cod)
    report = check_unitary(tensor, tol, n_cod)
    report.notes = f"box={box.name!r}, cod={dims(box.cod)}, dom={dims(box.dom)}"
    return report


# ──────────────── reflection positivity ───────────────────────────

def gram_min_eig(gram: np.ndarray) -> float:
    """Minimum eigenvalue of a symmetric matrix (used for RP / OS-positivity)."""
    return float(np.linalg.eigvalsh(gram).min())


def check_reflection_positivity(
    gram: np.ndarray,
    tol: float = 0.0,
) -> StructureReport:
    """Check that a Gram matrix is positive semidefinite (min-eig ≥ −tol).

    Reflection positivity requires all eigenvalues ≥ 0.  *tol* provides a
    small numerical slack for near-zero eigenvalues.
    """
    min_ev = gram_min_eig(gram)
    passed = min_ev >= -tol
    return StructureReport(
        property_name="reflection_positivity",
        passed=passed,
        defect=float(max(0.0, -min_ev)),
        tolerance=tol,
        notes=f"min_eig={min_ev:.3e}",
    )


# ─────────────────── enforcement helpers ──────────────────────────

def enforce_isometry(
    tensor: np.ndarray,
    n_cod: int | None = None,
) -> np.ndarray:
    """Return the nearest isometric tensor via SVD polar factor.

    Given ``M = U S V.T``, returns ``(U V.T).reshape(original_shape)``.
    Guarantees ``M_new @ M_new.T = I_cod`` provided ``cod_size ≤ dom_size``.
    """
    shape = tensor.shape
    M = _as_matrix(tensor, n_cod)
    rows, cols = M.shape
    if rows > cols:
        raise ValueError(
            f"enforce_isometry: cod_size ({rows}) > dom_size ({cols}); "
            "cannot project to isometric form."
        )
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return (U @ Vt).reshape(shape)


def enforce_unitary(
    tensor: np.ndarray,
    n_cod: int | None = None,
) -> np.ndarray:
    """Return the nearest unitary tensor via SVD polar factor.

    Requires ``cod_size == dom_size`` (square matrix).
    """
    shape = tensor.shape
    M = _as_matrix(tensor, n_cod)
    rows, cols = M.shape
    if rows != cols:
        raise ValueError(
            f"enforce_unitary: not square ({rows} × {cols})."
        )
    U, _, Vt = np.linalg.svd(M, full_matrices=True)
    return (U @ Vt).reshape(shape)
