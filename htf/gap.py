"""HTF Phase 4 — Spectral gap estimation and certified bounds.

Provides:
* Exact spectral gap for small systems (full diagonalisation).
* Variational upper bound on first excited state energy via orthogonal trial state.
* Temple's inequality lower bound on ground-state energy (rigorous, finite-lattice).
* Certified (flint Arb) upper bound on the spectral gap.

Honest scope
------------
* ``certified_gap_upper`` certifies **floating-point rounding** in the gap
  computation only.  Bond-dimension truncation bias is **not** certified here.
* Temple's lower bound is a rigorous **finite-lattice** lower bound on E_0,
  not a bound on the spectral gap itself.  The tighter ``E1_upper``, the
  tighter the bound.
* The continuum spectral gap (χ→∞, thermodynamic limit) is ``[OUT]``.
"""
from __future__ import annotations

import math

import numpy as np

from .certificate import Certificate


def spectral_gap_exact(ham: np.ndarray) -> float:
    """Exact spectral gap ``E_1 - E_0`` via full diagonalisation.

    Practical limit: ``n_sites ≤ 14`` (matrix size ``2^n × 2^n``).
    Returns 0.0 when the ground state is degenerate.
    """
    evals = np.linalg.eigvalsh(ham)
    return float(evals[1] - evals[0])


def h2_expectation(ham: np.ndarray, state_vec: np.ndarray) -> float:
    """Compute ``⟨ψ|H²|ψ⟩ / ⟨ψ|ψ⟩``.

    Used by :func:`temple_lower_bound`.  For real symmetric H: H² = H†H.
    """
    psi = np.asarray(state_vec, dtype=float).reshape(-1)
    norm_sq = float(psi @ psi)
    if norm_sq < 1e-15:
        raise ValueError("state vector has zero norm")
    Hpsi = ham @ psi
    return float(Hpsi @ Hpsi) / norm_sq


def temple_lower_bound(
    E_var: float,
    h2_exp: float,
    E1_lower: float,
) -> float:
    """Temple's inequality: **heuristic** estimate of a lower bound on E_0.

    Formula::

        result = E_var − (⟨H²⟩ − E_var²) / (E1_lower − E_var)

    **Critical requirement:** ``E1_lower`` must be a *true lower bound* on E_1
    (i.e. ``E1_lower ≤ E_1_exact``).  If an *upper* bound (e.g. a Ritz value) is
    passed, the denominator is too large, the subtracted term too small, and the
    result can **exceed E_0** — it is no longer a valid lower bound.

    The current call-sites in :func:`gap_report` and :func:`temple_lanczos` pass
    a Ritz variational upper bound.  Until a genuine lower bound on E_1 is
    available, treat the returned value as a **heuristic diagnostic only**,
    not a rigorous lower bound (P0-1).

    Returns ``-inf`` when ``E1_lower ≤ E_var`` (bound not applicable).
    """
    denom = E1_lower - E_var
    if denom <= 0.0:
        return float("-inf")
    variance = h2_exp - E_var ** 2
    return E_var - variance / denom


def first_excited_upper(
    ham: np.ndarray,
    state_gs: np.ndarray,
    state_es: np.ndarray,
) -> float:
    """Variational upper bound on E_1 via two-state Ritz.

    By the min-max theorem, the larger eigenvalue of the 2×2 compression of H
    to span{state_gs, state_es} satisfies E_1 ≤ θ_1(S) for any 2D subspace S,
    regardless of whether state_gs is the exact ground eigenvector.

    Parameters
    ----------
    ham      : dense Hamiltonian.
    state_gs : approximate ground state (normalisation applied internally).
    state_es : trial excited state (must be linearly independent of state_gs).
    """
    psi0 = np.asarray(state_gs, dtype=float).reshape(-1)
    phi  = np.asarray(state_es, dtype=float).reshape(-1)

    norm0 = np.linalg.norm(psi0)
    if norm0 < 1e-15:
        raise ValueError("ground-state vector has zero norm")
    psi0 = psi0 / norm0

    phi_perp = phi - float(psi0 @ phi) * psi0
    norm_perp = np.linalg.norm(phi_perp)
    if norm_perp < 1e-12:
        raise ValueError(
            "trial excited state is (near-)parallel to ground state; "
            "choose a linearly independent vector"
        )
    phi_perp /= norm_perp

    H11 = float(psi0 @ (ham @ psi0))
    H12 = float(psi0 @ (ham @ phi_perp))
    H22 = float(phi_perp @ (ham @ phi_perp))
    trace = H11 + H22
    disc = math.sqrt(max(0.0, ((H11 - H22) * 0.5) ** 2 + H12 ** 2))
    return trace * 0.5 + disc


def trial_energy_difference(
    ham: np.ndarray,
    state_gs: np.ndarray,
    state_es: np.ndarray,
) -> Certificate:
    """Arb-certified **trial energy difference** E1_var − E0_var.

    .. warning::
        This is **NOT** a certified upper bound on the spectral gap (P0-2).
        A true gap upper bound requires ``E1_upper − E0_lower``; this function
        computes ``E1_var − E0_var`` (both variational upper bounds), which can
        be *smaller* than the true gap.  The result is a discovery-tier estimate
        with Arb floating-point rounding certification only.

    The returned ``Certificate.result`` is the midpoint of the Arb interval;
    ``error_bound`` bounds floating-point rounding (not the gap itself).
    Bond-dimension and finite-size bias are ``[OUT]``.
    """
    try:
        from flint import arb, arb_mat
    except ImportError as exc:
        raise ImportError("trial_energy_difference requires python-flint") from exc

    psi0 = np.asarray(state_gs, dtype=float).reshape(-1)
    phi  = np.asarray(state_es, dtype=float).reshape(-1)

    norm0 = np.linalg.norm(psi0)
    if norm0 < 1e-15:
        raise ValueError("ground-state vector has zero norm")
    psi0 = psi0 / norm0

    phi_perp = phi - float(psi0 @ phi) * psi0
    norm_perp = np.linalg.norm(phi_perp)
    if norm_perp < 1e-12:
        raise ValueError("trial excited state too close to ground state")
    phi_perp /= norm_perp

    n = len(psi0)

    def _arb_energy(state: np.ndarray) -> arb:
        s_row = arb_mat([[arb(float(state[i])) for i in range(n)]])
        s_col = arb_mat([[arb(float(state[i]))] for i in range(n)])
        H_mat = arb_mat([[arb(float(ham[i, j])) for j in range(n)] for i in range(n)])
        num = s_row * (H_mat * s_col)
        den = s_row * s_col
        return num[0, 0] / den[0, 0]

    E0_arb  = _arb_energy(psi0)
    E1_arb  = _arb_energy(phi_perp)
    gap_arb = E1_arb - E0_arb

    return Certificate(
        result=float(gap_arb.mid()),
        mode="certified",
        error_bound=float(gap_arb.rad()),
        backend="flint-arb",
        notes=(
            "trial energy difference E1_var - E0_var (NOT a certified spectral-gap "
            "upper bound — P0-2); certifies floating-point rounding only; "
            "bond-dimension and finite-size bias are [OUT] (Phase 4 scope)"
        ),
    )


def certified_gap_upper(
    ham: np.ndarray,
    state_gs: np.ndarray,
    state_es: np.ndarray,
) -> Certificate:
    """Backward-compat alias for :func:`trial_energy_difference`.

    The old name ``certified_gap_upper`` implied a spectral-gap upper bound,
    which it is not (P0-2).  Prefer ``trial_energy_difference``.
    """
    return trial_energy_difference(ham, state_gs, state_es)


def gap_report(
    ham: np.ndarray,
    state_gs: np.ndarray,
    state_es: np.ndarray,
) -> dict:
    """Full gap report for a finite-lattice system.

    Returns a dict with:

    ``gap_exact``   — exact gap from full diagonalisation.
    ``E0_var``      — variational ground-state energy (upper bound on E_0).
    ``E1_var``      — variational first-excited energy (upper bound on E_1).
    ``gap_var``     — ``E1_var − E0_var`` (heuristic estimate; NOT a gap upper bound).
    ``temple_lb``   — Temple heuristic (NOT a rigorous lower bound — P0-1;
                      E1_var is an upper bound on E_1, not the required lower bound).
    ``gap_cert``    — :class:`~htf.certificate.Certificate` for trial energy diff
                      (NOT a certified gap upper bound — P0-2).
    """
    from .variational import energy_expectation

    E0_var   = energy_expectation(ham, state_gs)
    E1_var   = first_excited_upper(ham, state_gs, state_es)
    h2_exp   = h2_expectation(ham, state_gs)
    # NOTE: passing E1_var (an upper bound) violates Temple's E1_lower requirement.
    # The result is a heuristic diagnostic, not a rigorous lower bound (P0-1).
    t_lb     = temple_lower_bound(E0_var, h2_exp, E1_var)
    gap_cert = certified_gap_upper(ham, state_gs, state_es)

    return {
        "gap_exact": spectral_gap_exact(ham),
        "E0_var":    E0_var,
        "E1_var":    E1_var,
        "gap_var":   E1_var - E0_var,
        "temple_lb": t_lb,
        "gap_cert":  gap_cert,
    }
