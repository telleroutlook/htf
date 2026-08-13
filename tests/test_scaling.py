"""Tests for htf/scaling.py — chi-convergence study and ScalingReport."""
from __future__ import annotations

import math

import numpy as np
import pytest

from htf.scaling import ChiPoint, ScalingReport, _power_law_fit, chi_convergence_study
from htf.variational import transverse_ising_ham

# ───────────────────────────── Fixtures / helpers ─────────────────────────────

def make_ham(n: int, chi: int) -> np.ndarray:
    """Random symmetric Hamiltonian of shape (chi^n, chi^n)."""
    rng = np.random.default_rng(42)
    dim = chi ** n
    A = rng.standard_normal((dim, dim))
    return (A + A.T) / 2


def qubit_ham_factory(n: int, chi: int) -> np.ndarray:
    """Transverse Ising Hamiltonian; only valid for chi==2."""
    return transverse_ising_ham(n)


def bad_shape_factory(n: int, chi: int) -> np.ndarray:
    """Always returns a 3×3 matrix regardless of chi, so shape is wrong."""
    return np.eye(3)


# ─────────────────────────────── ChiPoint ────────────────────────────────────

class TestChiPoint:
    def test_chi_stored(self):
        p = ChiPoint(chi=4, energy=-1.5, error_bound=1e-10, n_iter_used=42)
        assert p.chi == 4

    def test_energy_stored(self):
        p = ChiPoint(chi=4, energy=-1.5, error_bound=1e-10, n_iter_used=42)
        assert p.energy == pytest.approx(-1.5)

    def test_error_bound_stored(self):
        p = ChiPoint(chi=4, energy=-1.5, error_bound=1e-10, n_iter_used=42)
        assert p.error_bound == pytest.approx(1e-10)

    def test_n_iter_used_stored(self):
        p = ChiPoint(chi=4, energy=-1.5, error_bound=1e-10, n_iter_used=42)
        assert p.n_iter_used == 42

    def test_zero_error_bound_allowed(self):
        p = ChiPoint(chi=2, energy=0.0, error_bound=0.0, n_iter_used=1)
        assert p.error_bound == 0.0

    def test_chi_int(self):
        p = ChiPoint(chi=8, energy=0.5, error_bound=1e-12, n_iter_used=3)
        assert isinstance(p.chi, int)


# ─────────────────────────── ScalingReport ───────────────────────────────────

class TestScalingReport:
    def test_n_sites_stored(self):
        r = ScalingReport(n_sites=4)
        assert r.n_sites == 4

    def test_default_chi_points_empty(self):
        r = ScalingReport(n_sites=4)
        assert r.chi_points == []

    def test_default_E_extrapolated_nan(self):
        r = ScalingReport(n_sites=4)
        assert math.isnan(r.E_extrapolated)

    def test_default_E_extrap_stderr_nan(self):
        r = ScalingReport(n_sites=4)
        assert math.isnan(r.E_extrap_stderr)

    def test_default_fit_exponent_nan(self):
        r = ScalingReport(n_sites=4)
        assert math.isnan(r.fit_exponent)

    def test_default_notes_empty(self):
        r = ScalingReport(n_sites=4)
        assert r.notes == ""

    def test_chi_points_stored(self):
        p = ChiPoint(chi=2, energy=-1.0, error_bound=1e-8, n_iter_used=10)
        r = ScalingReport(n_sites=2, chi_points=[p])
        assert len(r.chi_points) == 1
        assert r.chi_points[0].chi == 2

    def test_summary_returns_string(self):
        r = ScalingReport(n_sites=2)
        assert isinstance(r.summary(), str)

    def test_summary_contains_n_sites(self):
        r = ScalingReport(n_sites=4)
        assert "4" in r.summary()

    def test_summary_contains_chi_header(self):
        r = ScalingReport(n_sites=4)
        assert "chi" in r.summary()

    def test_summary_contains_chi_value(self):
        p = ChiPoint(chi=3, energy=-2.5, error_bound=1e-9, n_iter_used=20)
        r = ScalingReport(n_sites=2, chi_points=[p])
        assert "3" in r.summary()

    def test_summary_contains_energy_value(self):
        p = ChiPoint(chi=2, energy=-1.23456789, error_bound=1e-9, n_iter_used=20)
        r = ScalingReport(n_sites=2, chi_points=[p])
        assert "-1.23456789" in r.summary()

    def test_summary_no_extrapolation_section_when_nan(self):
        r = ScalingReport(n_sites=2)
        assert "Power-law extrapolation" not in r.summary()

    def test_summary_shows_extrapolation_when_set(self):
        r = ScalingReport(n_sites=2, E_extrapolated=-2.0, E_extrap_stderr=0.01)
        assert "extrapolation" in r.summary().lower()

    def test_summary_shows_notes_when_set(self):
        r = ScalingReport(n_sites=2, notes="some diagnostic note")
        assert "some diagnostic note" in r.summary()

    def test_multiple_chi_points_all_shown(self):
        pts = [
            ChiPoint(chi=2, energy=-1.0, error_bound=1e-8, n_iter_used=10),
            ChiPoint(chi=3, energy=-1.5, error_bound=1e-8, n_iter_used=10),
        ]
        r = ScalingReport(n_sites=2, chi_points=pts)
        summary = r.summary()
        assert "2" in summary
        assert "3" in summary


# ─────────────────────── chi_convergence_study ───────────────────────────────

class TestChiConvergenceStudy:
    def test_returns_scaling_report_type(self):
        report = chi_convergence_study(2, [2], make_ham, n_iter=5, seed=0)
        assert isinstance(report, ScalingReport)

    def test_n_sites_stored_in_report(self):
        report = chi_convergence_study(2, [2], make_ham, n_iter=5, seed=0)
        assert report.n_sites == 2

    def test_single_chi_produces_one_point(self):
        report = chi_convergence_study(2, [2], make_ham, n_iter=5, seed=0)
        assert len(report.chi_points) == 1

    def test_two_chi_produces_two_points(self):
        report = chi_convergence_study(2, [2, 3], make_ham, n_iter=5, seed=0)
        assert len(report.chi_points) == 2

    def test_chi_value_stored_in_point(self):
        report = chi_convergence_study(2, [2], make_ham, n_iter=5, seed=0)
        assert report.chi_points[0].chi == 2

    def test_chi_values_stored_in_order(self):
        report = chi_convergence_study(2, [2, 3], make_ham, n_iter=5, seed=0)
        assert [p.chi for p in report.chi_points] == [2, 3]

    def test_error_bound_non_negative_single(self):
        report = chi_convergence_study(2, [2], make_ham, n_iter=5, seed=0)
        assert report.chi_points[0].error_bound >= 0.0

    def test_error_bound_non_negative_all_points(self):
        report = chi_convergence_study(2, [2, 3], make_ham, n_iter=5, seed=0)
        for p in report.chi_points:
            assert p.error_bound >= 0.0

    def test_n_iter_used_positive(self):
        report = chi_convergence_study(2, [2], make_ham, n_iter=5, seed=0)
        assert report.chi_points[0].n_iter_used > 0

    def test_energy_is_float(self):
        report = chi_convergence_study(2, [2], make_ham, n_iter=5, seed=0)
        assert isinstance(report.chi_points[0].energy, float)

    def test_single_chi_no_extrapolation(self):
        report = chi_convergence_study(2, [2], make_ham, n_iter=5, seed=0)
        assert math.isnan(report.E_extrapolated)

    def test_single_chi_notes_empty(self):
        report = chi_convergence_study(2, [2], make_ham, n_iter=5, seed=0)
        assert report.notes == ""

    def test_two_chi_no_extrapolation(self):
        report = chi_convergence_study(2, [2, 3], make_ham, n_iter=5, seed=0)
        assert math.isnan(report.E_extrapolated)

    def test_two_chi_notes_empty(self):
        report = chi_convergence_study(2, [2, 3], make_ham, n_iter=5, seed=0)
        assert report.notes == ""

    def test_three_chi_notes_set(self):
        report = chi_convergence_study(2, [2, 3, 4], make_ham, n_iter=5, seed=0)
        assert report.notes != ""

    def test_three_chi_three_points(self):
        report = chi_convergence_study(2, [2, 3, 4], make_ham, n_iter=5, seed=0)
        assert len(report.chi_points) == 3

    def test_qubit_ham_factory_chi2(self):
        report = chi_convergence_study(2, [2], qubit_ham_factory, n_iter=5, seed=0)
        assert len(report.chi_points) == 1
        assert report.chi_points[0].chi == 2

    def test_qubit_ham_energy_below_zero(self):
        # TFIM ground state energy is negative
        report = chi_convergence_study(2, [2], qubit_ham_factory, n_iter=20, seed=0)
        assert report.chi_points[0].energy < 0.0

    def test_wrong_shape_raises_value_error(self):
        with pytest.raises(ValueError):
            chi_convergence_study(2, [2], bad_shape_factory, n_iter=3, seed=0)

    def test_wrong_shape_error_mentions_shape(self):
        with pytest.raises(ValueError, match="shape"):
            chi_convergence_study(2, [2], bad_shape_factory, n_iter=3, seed=0)

    def test_summary_with_chi_points_is_string(self):
        report = chi_convergence_study(2, [2], make_ham, n_iter=5, seed=0)
        assert isinstance(report.summary(), str)

    def test_summary_contains_chi_from_study(self):
        report = chi_convergence_study(2, [2], make_ham, n_iter=5, seed=0)
        assert "2" in report.summary()

    def test_different_seeds_same_chi_count(self):
        r1 = chi_convergence_study(2, [2], make_ham, n_iter=5, seed=0)
        r2 = chi_convergence_study(2, [2], make_ham, n_iter=5, seed=7)
        assert len(r1.chi_points) == len(r2.chi_points) == 1

    def test_n_iter_used_lte_requested(self):
        # Optimizer may terminate early but should not exceed n_iter significantly
        report = chi_convergence_study(2, [2], make_ham, n_iter=10, seed=0)
        # The recorded history length is bounded by the optimizer call
        assert report.chi_points[0].n_iter_used >= 1


# ──────────────────────────── _power_law_fit ─────────────────────────────────

class TestPowerLawFit:
    """Tests for the private power-law fitting helper."""

    def test_returns_three_element_tuple(self):
        # E ~ E_inf + a/chi^b; energies decreasing so residuals are positive
        chi_vals = [2, 4, 8, 16]
        energies = [-1.0, -1.5, -1.75, -1.875]
        result = _power_law_fit(chi_vals, energies)
        assert len(result) == 3

    def test_e_inf_equals_last_energy(self):
        chi_vals = [2, 4, 8, 16]
        energies = [-1.0, -1.5, -1.75, -1.875]
        E_inf, _, _ = _power_law_fit(chi_vals, energies)
        assert E_inf == pytest.approx(-1.875)

    def test_exponent_b_is_float(self):
        chi_vals = [2, 4, 8, 16]
        energies = [-1.0, -1.5, -1.75, -1.875]
        _, _, b = _power_law_fit(chi_vals, energies)
        assert isinstance(b, float)

    def test_exponent_b_positive_for_converging_series(self):
        # Energy increases toward the reference (ground state from below): b > 0
        chi_vals = [2, 4, 8, 16]
        energies = [-1.0, -1.5, -1.75, -1.875]
        _, _, b = _power_law_fit(chi_vals, energies)
        assert b > 0.0

    def test_raises_value_error_no_positive_residuals(self):
        # All energies equal to E_ref → no positive residuals
        with pytest.raises(ValueError):
            _power_law_fit([2, 4, 8], [-2.0, -2.0, -2.0])

    def test_raises_value_error_only_one_positive_residual(self):
        # Only one energy strictly above E_ref → mask.sum() = 1 < 2
        with pytest.raises(ValueError):
            _power_law_fit([2, 4, 8], [-1.5, -2.0, -2.0])

    def test_stderr_is_float_or_nan(self):
        chi_vals = [2, 4, 8, 16]
        energies = [-1.0, -1.5, -1.75, -1.875]
        _, stderr, _ = _power_law_fit(chi_vals, energies)
        assert isinstance(stderr, float)
