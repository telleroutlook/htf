"""HTF Phase 4 — Bond-dimension (χ) convergence study.

Runs the MERA variational optimiser at a sequence of bond dimensions χ and
collects the resulting energies.  The idea is to trace how E(χ) converges
toward the infinite-χ limit.

Honest scope
------------
* ``chi_convergence_study`` uses float optimisation; the ``error_bound`` in
  each :class:`ChiPoint` covers floating-point rounding only (via flint Arb),
  **not** the residual truncation bias from finite χ — that is ``[OUT]``.
* Power-law extrapolation fits ``E(χ) ≈ E_∞ + a / χ^b`` and returns the
  result, but the extrapolation uncertainty is **NOT certified**.  It is a
  discovery-tier heuristic labelled as such.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class ChiPoint:
    """One data point in a χ-convergence study."""
    chi: int
    energy: float      # variational energy after optimisation (certified midpoint)
    error_bound: float # floating-point rounding bound
    n_iter_used: int   # number of L-BFGS-B callback steps recorded


@dataclass
class ScalingReport:
    """Full χ-convergence study result."""
    n_sites: int
    chi_points: List[ChiPoint] = field(default_factory=list)
    E_extrapolated: float = float("nan")  # power-law fit; [OUT] scope
    E_extrap_stderr: float = float("nan")
    fit_exponent: float = float("nan")    # b in a/χ^b
    notes: str = ""

    def summary(self) -> str:
        lines = [
            f"χ-convergence study: n_sites={self.n_sites}",
            f"{'chi':>6}  {'E(chi)':>14}  {'error_bound':>12}",
        ]
        for p in self.chi_points:
            lines.append(f"{p.chi:>6}  {p.energy:>14.8f}  {p.error_bound:>12.2e}")
        if not np.isnan(self.E_extrapolated):
            lines.append(
                f"\nPower-law extrapolation (χ→∞): E_∞ ≈ {self.E_extrapolated:.8f} "
                f"± {self.E_extrap_stderr:.2e}  [heuristic, NOT certified]"
            )
        if self.notes:
            lines.append(f"Note: {self.notes}")
        return "\n".join(lines)


def chi_convergence_study(
    n_sites: int,
    chi_list: List[int],
    ham_factory,
    n_iter: int = 50,
    seed: int = 0,
) -> ScalingReport:
    """Run MERA variational optimisation at each χ in *chi_list*.

    Parameters
    ----------
    n_sites     : number of physical sites (must be a power of 2).
    chi_list    : bond dimensions to try (increasing order recommended).
    ham_factory : callable ``(n_sites, chi) -> np.ndarray`` that returns the
                  dense ``(chi^n_sites × chi^n_sites)`` Hamiltonian for the
                  given local dimension ``chi``.  For qubits, pass
                  ``lambda n, chi: transverse_ising_ham(n)`` and restrict to
                  ``chi_list=[2]``; for general spin-χ models supply an
                  appropriate factory.
    n_iter      : L-BFGS-B iterations per χ.
    seed        : RNG seed for initial MERA.

    Returns
    -------
    :class:`ScalingReport`
        One :class:`ChiPoint` per χ, plus an optional power-law extrapolation
        when ≥ 3 χ values are provided.
    """
    from .mera import random_mera
    from .variational import optimize_mera, variational_bound

    points: list[ChiPoint] = []
    for chi in chi_list:
        ham = np.asarray(ham_factory(n_sites, chi), dtype=float)
        expected_dim = chi ** n_sites
        if ham.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"ham_factory({n_sites}, {chi}) returned shape {ham.shape}; "
                f"expected ({expected_dim}, {expected_dim}).  "
                f"For qubit Hamiltonians use chi_list=[2] only."
            )
        mera0 = random_mera(n_sites, chi=chi, seed=seed)
        mera_opt, history = optimize_mera(ham, mera0, n_iter=n_iter, tol=1e-6)
        cert = variational_bound(ham, mera_opt)
        points.append(ChiPoint(
            chi=chi,
            energy=cert.result,
            error_bound=cert.error_bound,
            n_iter_used=len(history),
        ))

    report = ScalingReport(n_sites=n_sites, chi_points=points)

    if len(points) >= 3:
        try:
            E_inf, stderr, b = _power_law_fit(
                [p.chi for p in points],
                [p.energy for p in points],
            )
            report.E_extrapolated = E_inf
            report.E_extrap_stderr = stderr
            report.fit_exponent = b
            report.notes = (
                "power-law extrapolation is heuristic (discovery-tier); "
                "residual χ-truncation bias is [OUT] of certified scope"
            )
        except Exception as exc:
            report.notes = f"power-law fit failed: {exc}"

    return report


def _power_law_fit(
    chi_vals: List[int],
    energies: List[float],
) -> tuple[float, float, float]:
    """Fit ``E(χ) = E_∞ + a / χ^b`` via log-linearised least squares.

    Uses the largest-χ energy as an estimate of E_∞ and fits the residuals
    ``E(χ) - E_∞ ≈ a / χ^b`` in log space.

    Returns
    -------
    (E_inf, stderr_E_inf, b)
        ``stderr_E_inf`` is the propagated standard error from the linear fit.

    Raises
    ------
    ValueError
        If fewer than 2 usable residuals remain after thresholding.
    """
    chi_arr = np.array(chi_vals, dtype=float)
    E_arr   = np.array(energies, dtype=float)

    # Anchor at largest-χ energy; fit residuals E(chi) - E_ref
    E_ref     = E_arr[-1]
    residuals = E_arr - E_ref
    mask      = residuals > 1e-12
    if mask.sum() < 2:
        raise ValueError("not enough positive residuals for power-law fit")

    log_chi = np.log(chi_arr[mask])
    log_res = np.log(residuals[mask])

    # log_res = c0 - b * log_chi  →  linear regression
    A = np.column_stack([np.ones_like(log_chi), log_chi])
    result = np.linalg.lstsq(A, log_res, rcond=None)
    coeffs = result[0]
    resid_sq = result[1]
    c0, neg_b = coeffs
    b = -neg_b

    n = int(mask.sum())
    if resid_sq.size > 0 and n > 2:
        sigma2   = float(resid_sq[0]) / (n - 2)
        AtA_inv  = np.linalg.pinv(A.T @ A)
        var_c0   = float(sigma2 * AtA_inv[0, 0])
        stderr   = float(np.exp(c0) * np.sqrt(var_c0)) if var_c0 >= 0 else float("nan")
    else:
        stderr = float("nan")

    return E_ref, stderr, b
