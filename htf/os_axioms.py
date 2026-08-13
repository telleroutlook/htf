"""HTF Phase 4 — Finite-lattice reflection diagnostics.

Provides structural diagnostics for finite-lattice Hamiltonians via
transfer-matrix analysis.

.. warning:: **P0-5 — These are NOT OS-positivity checks.**
    For any real symmetric Hamiltonian:

    * ``check_transfer_positivity``: T = exp(−βH) is *always* PSD
      (eigenvalues exp(−β λ_i) > 0 by construction).
    * ``os_gram_positivity``: G = T + RTR is a sum of two PSD matrices,
      hence *always* PSD.
    * Therefore checks 1 and 3 pass for every real symmetric H by definition.
      The *only* non-trivial check is ``reflection_symmetry`` ([H,R]=0).

    True Osterwalder-Schrader reflection positivity requires a Gram form
    ⟨ΘF_i, F_j⟩ on positive-time observable algebras — not a plain PSD
    check on T.  Use ``finite_lattice_reflection_diagnostics`` (the
    canonical name); the old ``os_positivity_report`` is an alias kept for
    backwards compatibility but will be removed in a future release.

1. ``check_transfer_positivity``: T = exp(−βH) is PSD (always true).
2. ``check_reflection_symmetry``: [H, R] = 0 (only non-trivial check).
3. ``finite_lattice_reflection_diagnostics``: checks 1+2 + Gram PSD (always true).

Honest scope
------------
* All checks cover **finite lattices** (n_sites sites, local dim d).
* A passing report is only a reflection-symmetry check; it is **not** an
  OS-positivity certificate and does not imply OS-RP of any continuum QFT.
* Works for real symmetric Hamiltonians (standard lattice spin models).
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import expm

from .structure import StructureReport, check_reflection_positivity


def transfer_matrix(ham: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Transfer matrix T = exp(−β H) via Padé matrix exponentiation.

    Parameters
    ----------
    ham  : real symmetric Hamiltonian, shape (D, D).
    beta : imaginary-time step > 0 (default 1.0).

    Returns
    -------
    T : (D × D) matrix.  For Hermitian H, all eigenvalues exp(−β λ_i) > 0.
    """
    return expm(-beta * np.asarray(ham, dtype=float))


def reflection_operator(n_sites: int, d: int = 2) -> np.ndarray:
    """Spatial reflection operator for a 1-D chain.

    Acts as: R |s_0 s_1 … s_{n−1}⟩ = |s_{n−1} … s_1 s_0⟩
    (big-endian tensor-product basis: index = Σ s_k d^{n−1−k}).

    Returns
    -------
    R : (d^n × d^n) real orthogonal matrix satisfying R² = I.
    """
    dim = d ** n_sites
    powers = d ** np.arange(n_sites - 1, -1, -1)  # [d^{n-1}, …, 1]
    perm = np.empty(dim, dtype=int)
    for i in range(dim):
        digits = (i // powers) % d            # s_0, s_1, …, s_{n-1}
        perm[i] = int(np.dot(digits[::-1], powers))  # reflected index
    R = np.zeros((dim, dim))
    R[perm, np.arange(dim)] = 1.0
    return R


def check_transfer_positivity(
    ham: np.ndarray,
    beta: float = 1.0,
    tol: float = 1e-10,
) -> StructureReport:
    """Check that the transfer matrix T = exp(−β H) is positive semi-definite.

    For a Hermitian Hamiltonian this is guaranteed analytically (all
    eigenvalues of T are exp(−β λ_i) > 0); this function verifies it
    numerically and reports the defect max(0, −min_eig(T)).
    """
    T = transfer_matrix(ham, beta)
    min_ev = float(np.linalg.eigvalsh(T).min())
    defect = max(0.0, -min_ev)
    return StructureReport(
        property_name="transfer_matrix_positivity",
        passed=defect <= tol,
        defect=defect,
        tolerance=tol,
        notes=f"beta={beta:.4g}, min_eig(T)={min_ev:.3e}",
    )


def check_reflection_symmetry(
    ham: np.ndarray,
    n_sites: int,
    d: int = 2,
    tol: float = 1e-10,
) -> StructureReport:
    """Check that the Hamiltonian commutes with spatial reflection: [H, R] = 0.

    This is a prerequisite for OS-positivity in reflection-symmetric theories.
    defect = ||HR − RH||_max.
    """
    H = np.asarray(ham, dtype=float)
    R = reflection_operator(n_sites, d)
    comm = H @ R - R @ H
    defect = float(np.abs(comm).max())
    return StructureReport(
        property_name="reflection_symmetry",
        passed=defect <= tol,
        defect=defect,
        tolerance=tol,
        notes=f"n_sites={n_sites}, d={d}, ||[H,R]||_max={defect:.3e}",
    )


def os_positivity_report(
    ham: np.ndarray,
    n_sites: int,
    beta: float = 1.0,
    d: int = 2,
    tol: float = 1e-10,
) -> dict:
    """Deprecated alias for :func:`finite_lattice_reflection_diagnostics`.

    .. deprecated::
        Use ``finite_lattice_reflection_diagnostics`` instead.
        This name implied OS-positivity verification, which it does not
        perform (P0-5).
    """
    warnings.warn(
        "os_positivity_report is deprecated and misleadingly named (P0-5): "
        "checks 1 and 3 pass for all real symmetric H by construction; "
        "use finite_lattice_reflection_diagnostics instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return finite_lattice_reflection_diagnostics(ham, n_sites, beta=beta, d=d, tol=tol)


def finite_lattice_reflection_diagnostics(
    ham: np.ndarray,
    n_sites: int,
    beta: float = 1.0,
    d: int = 2,
    tol: float = 1e-10,
) -> dict:
    """Finite-lattice structural diagnostics for a Hamiltonian.

    .. warning::
        This is **not** a true OS-positivity check (P0-5).  Checks 1 and 3
        pass for *all* real symmetric Hamiltonians by construction (transfer
        matrix and G=T+RTR are always PSD).  The only non-trivial check is
        ``reflection_symmetry`` ([H,R]=0).

    Performs three diagnostics:

    1. ``transfer_positivity``: T = exp(−βH) is PSD — always true.
    2. ``reflection_symmetry``: [H, R] = 0 — the only informative check.
    3. ``os_gram_positivity``: G = T + RTR is PSD — always true.

    Parameters
    ----------
    ham     : dense (D × D) Hamiltonian, D = d^n_sites.
    n_sites : number of physical sites.
    beta    : imaginary-time inverse-temperature (default 1.0).
    d       : local Hilbert-space dimension (default 2 for qubits).
    tol     : numerical tolerance for "passed" (default 1e-10).

    Returns
    -------
    dict with keys ``transfer_positivity``, ``reflection_symmetry``,
    ``os_gram_positivity``, ``all_passed``, and ``notes``.
    """
    r_trans = check_transfer_positivity(ham, beta, tol)
    r_refl  = check_reflection_symmetry(ham, n_sites, d, tol)

    R = reflection_operator(n_sites, d)
    T = transfer_matrix(ham, beta)
    G = T + R @ T @ R
    r_gram = check_reflection_positivity(G, tol=tol)
    r_gram.property_name = "os_gram_positivity"
    r_gram.notes = (
        f"G = T + RTR, min_eig={float(np.linalg.eigvalsh(G).min()):.3e}; "
        "always PSD for real symmetric H — not an independent OS check (P0-5)"
    )

    all_passed = r_trans.passed and r_refl.passed and r_gram.passed
    return {
        "transfer_positivity": r_trans,
        "reflection_symmetry": r_refl,
        "os_gram_positivity": r_gram,
        "all_passed": all_passed,
        "notes": (
            f"finite-lattice reflection diagnostics: n_sites={n_sites}, d={d}, beta={beta}; "
            "checks 1+3 always pass for real symmetric H; "
            "only check 2 ([H,R]=0) is non-trivial; "
            "OS-positivity of any continuum QFT is [OUT]; P0-5"
        ),
    }
