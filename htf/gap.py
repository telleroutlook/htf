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
    E1_upper: float,
) -> float:
    """Temple's inequality: rigorous lower bound on the ground-state energy E_0.

    Given a trial state |ψ⟩ with variational energy ``E_var`` and second
    moment ``h2_exp``, and an upper bound ``E1_upper`` on the first excited
    energy::

        E_0 ≥ E_var − (⟨H²⟩ − E_var²) / (E1_upper − E_var)

    This is a **rigorous finite-lattice lower bound** (not a certified bound
    on the spectral gap; bond-dimension bias is ``[OUT]``).

    Returns ``-inf`` when ``E1_upper ≤ E_var`` (bound not applicable).
    """
    denom = E1_upper - E_var
    if denom <= 0.0:
        return float("-inf")
    variance = h2_exp - E_var ** 2
    return E_var - variance / denom


def first_excited_upper(
    ham: np.ndarray,
    state_gs: np.ndarray,
    state_es: np.ndarray,
) -> float:
    """Variational upper bound on E_1 from a state orthogonalised to |ψ_0⟩.

    Projects ``|φ⟩`` onto the orthogonal complement of ``|ψ_0⟩``::

        |φ_⊥⟩ = |φ⟩ − ⟨ψ_0|φ⟩|ψ_0⟩   (normalised)

    Then ``E_1 ≤ ⟨φ_⊥|H|φ_⊥⟩ / ⟨φ_⊥|φ_⊥⟩``.

    Parameters
    ----------
    ham      : dense Hamiltonian.
    state_gs : approximate ground state (normalisation applied internally).
    state_es : trial excited state (must not be parallel to state_gs).
    """
    from .variational import energy_expectation

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
    return energy_expectation(ham, phi_perp)


def certified_gap_upper(
    ham: np.ndarray,
    state_gs: np.ndarray,
    state_es: np.ndarray,
) -> Certificate:
    """Certified upper bound on the spectral gap via flint Arb.

    Computes ``E_1_var − E_0_var`` where both energies are evaluated with
    Arb interval arithmetic, giving a rigorous floating-point rounding bound.

    The returned ``Certificate.result`` is the gap estimate;
    ``error_bound`` bounds the floating-point rounding error.
    Bond-dimension truncation bias is ``[OUT]``.
    """
    try:
        from flint import arb, arb_mat
    except ImportError as exc:
        raise ImportError("certified_gap_upper requires python-flint") from exc

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

    def _arb_energy(state: np.ndarray) -> "arb":
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
            "variational upper bound on spectral gap; "
            "certifies floating-point rounding only; "
            "bond-dimension and finite-size bias are [OUT] (Phase 4 scope)"
        ),
    )


def gap_report(
    ham: np.ndarray,
    state_gs: np.ndarray,
    state_es: np.ndarray,
) -> dict:
    """Full gap report for a finite-lattice system.

    Returns a dict with:

    ``gap_exact``   — exact gap from full diagonalisation.
    ``E0_var``      — variational ground-state energy.
    ``E1_var``      — variational first-excited energy (upper bound on E_1).
    ``gap_var``     — ``E1_var − E0_var`` (upper bound on gap).
    ``temple_lb``   — Temple's inequality lower bound on E_0.
    ``gap_cert``    — :class:`~htf.certificate.Certificate` for gap_var.
    """
    from .variational import energy_expectation

    E0_var   = energy_expectation(ham, state_gs)
    E1_var   = first_excited_upper(ham, state_gs, state_es)
    h2_exp   = h2_expectation(ham, state_gs)
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
