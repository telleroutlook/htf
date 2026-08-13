"""Tests for htf/gap.py — Spectral gap estimation and certified bounds."""
from __future__ import annotations

import math

import numpy as np
import pytest

from htf.certificate import Certificate
from htf.gap import (
    certified_gap_upper,
    first_excited_upper,
    gap_report,
    h2_expectation,
    spectral_gap_exact,
    temple_lower_bound,
)
from htf.variational import energy_expectation, transverse_ising_ham, xx_model_ham

# ─────────────────────────── Helpers ──────────────────────────────

def _eigh(ham: np.ndarray):
    """Return (eigenvalues, eigenvectors) via np.linalg.eigh."""
    return np.linalg.eigh(ham)


def _gs(ham: np.ndarray) -> np.ndarray:
    """Exact normalised ground-state eigenvector."""
    return _eigh(ham)[1][:, 0]


def _es(ham: np.ndarray) -> np.ndarray:
    """Exact normalised first-excited-state eigenvector."""
    return _eigh(ham)[1][:, 1]


# ─────────────────────── spectral_gap_exact ───────────────────────

class TestSpectralGapExact:
    def test_returns_float_n2(self):
        H = transverse_ising_ham(2)
        result = spectral_gap_exact(H)
        assert isinstance(result, float)

    def test_nonnegative_gap_n2(self):
        H = transverse_ising_ham(2)
        assert spectral_gap_exact(H) >= 0.0

    def test_nonnegative_gap_n4(self):
        H = transverse_ising_ham(4)
        assert spectral_gap_exact(H) >= 0.0

    def test_nonnegative_gap_xx_model(self):
        H = xx_model_ham(3)
        assert spectral_gap_exact(H) >= 0.0

    def test_agrees_with_direct_eigh_n2(self):
        H = transverse_ising_ham(2)
        evals = np.linalg.eigvalsh(H)
        expected = float(evals[1] - evals[0])
        np.testing.assert_allclose(spectral_gap_exact(H), expected, rtol=1e-12)

    def test_agrees_with_direct_eigh_n4(self):
        H = transverse_ising_ham(4, J=1.0, h=0.5)
        evals = np.linalg.eigvalsh(H)
        expected = float(evals[1] - evals[0])
        np.testing.assert_allclose(spectral_gap_exact(H), expected, rtol=1e-12)

    def test_physics_n4_gap_approx(self):
        # For n=4, J=1, h=0.5: gap ≈ 0.0948
        H = transverse_ising_ham(4, J=1.0, h=0.5)
        gap = spectral_gap_exact(H)
        np.testing.assert_allclose(gap, 0.0948, atol=0.005)

    def test_identity_ham_zero_gap(self):
        # All eigenvalues equal → gap == 0
        H = np.eye(4)
        assert spectral_gap_exact(H) == 0.0

    def test_diagonal_ham_gap_is_second_minus_first(self):
        diag = np.array([3.0, 1.0, 5.0, 2.0])
        H = np.diag(diag)
        gap = spectral_gap_exact(H)
        sorted_d = np.sort(diag)
        np.testing.assert_allclose(gap, sorted_d[1] - sorted_d[0], rtol=1e-12)

    def test_gap_n2_xx_model(self):
        H = xx_model_ham(2)
        gap = spectral_gap_exact(H)
        evals = np.linalg.eigvalsh(H)
        np.testing.assert_allclose(gap, evals[1] - evals[0], rtol=1e-12)

    def test_strong_field_gap_larger(self):
        # Strong transverse field opens the gap
        H_weak = transverse_ising_ham(4, J=1.0, h=0.1)
        H_strong = transverse_ising_ham(4, J=1.0, h=2.0)
        assert spectral_gap_exact(H_strong) > spectral_gap_exact(H_weak)

    def test_n3_gap_nonnegative(self):
        H = transverse_ising_ham(3, J=1.0, h=1.0)
        assert spectral_gap_exact(H) >= 0.0


# ───────────────────────── h2_expectation ─────────────────────────

class TestH2Expectation:
    def setup_method(self):
        self.H2 = transverse_ising_ham(2, J=1.0, h=0.5)
        self.evals, self.evecs = _eigh(self.H2)

    def test_returns_float(self):
        gs = self.evecs[:, 0]
        result = h2_expectation(self.H2, gs)
        assert isinstance(result, float)

    def test_zero_norm_raises(self):
        with pytest.raises(ValueError, match="zero norm"):
            h2_expectation(self.H2, np.zeros(4))

    def test_near_zero_norm_raises(self):
        with pytest.raises(ValueError):
            h2_expectation(self.H2, np.ones(4) * 1e-9)

    def test_eigenstate_gives_eigenvalue_squared(self):
        # H|psi_i> = E_i|psi_i> => <H^2> = E_i^2
        for i in range(4):
            vec = self.evecs[:, i]
            h2 = h2_expectation(self.H2, vec)
            np.testing.assert_allclose(h2, self.evals[i] ** 2, rtol=1e-8)

    def test_variance_nonnegative_for_gs(self):
        # <H^2> >= <H>^2 (Cauchy-Schwarz / Jensen)
        gs = self.evecs[:, 0]
        h2 = h2_expectation(self.H2, gs)
        e_var = energy_expectation(self.H2, gs)
        assert h2 >= e_var ** 2 - 1e-12

    def test_variance_nonnegative_random_state(self):
        rng = np.random.default_rng(0)
        for _ in range(8):
            state = rng.standard_normal(4)
            h2 = h2_expectation(self.H2, state)
            e_var = energy_expectation(self.H2, state)
            assert h2 >= e_var ** 2 - 1e-10

    def test_scale_invariant(self):
        gs = self.evecs[:, 0]
        h2_norm = h2_expectation(self.H2, gs)
        h2_scaled = h2_expectation(self.H2, gs * 7.5)
        np.testing.assert_allclose(h2_norm, h2_scaled, rtol=1e-10)

    def test_accepts_2d_input(self):
        gs = self.evecs[:, 0].reshape(4, 1)
        result = h2_expectation(self.H2, gs)
        assert isinstance(result, float)

    def test_n4_eigenstate_gives_eigenvalue_squared(self):
        H = transverse_ising_ham(4, J=1.0, h=0.5)
        evals, evecs = _eigh(H)
        vec = evecs[:, 0]
        h2 = h2_expectation(H, vec)
        np.testing.assert_allclose(h2, evals[0] ** 2, rtol=1e-8)

    def test_ground_state_variance_is_zero(self):
        # Exact eigenstate: variance = <H^2> - <H>^2 = 0
        gs = self.evecs[:, 0]
        h2 = h2_expectation(self.H2, gs)
        e_var = energy_expectation(self.H2, gs)
        variance = h2 - e_var ** 2
        np.testing.assert_allclose(variance, 0.0, atol=1e-10)


# ──────────────────────── temple_lower_bound ──────────────────────

class TestTempleLowerBound:
    def setup_method(self):
        self.H = transverse_ising_ham(2, J=1.0, h=0.5)
        self.evals, self.evecs = _eigh(self.H)
        self.E0 = float(self.evals[0])
        self.E1 = float(self.evals[1])

    def test_returns_float_valid(self):
        gs = self.evecs[:, 0]
        E_var = energy_expectation(self.H, gs)
        h2 = h2_expectation(self.H, gs)
        result = temple_lower_bound(E_var, h2, self.E1 + 0.1)
        assert isinstance(result, float)

    def test_returns_neg_inf_when_E1_upper_leq_E_var(self):
        # E1_upper <= E_var → bound not applicable
        E_var = -1.0
        h2 = 1.0
        E1_upper = -1.5  # less than E_var
        result = temple_lower_bound(E_var, h2, E1_upper)
        assert result == float("-inf")

    def test_returns_neg_inf_when_E1_upper_equals_E_var(self):
        E_var = -2.0
        E1_upper = -2.0  # exactly equal
        result = temple_lower_bound(E_var, 4.0, E1_upper)
        assert result == float("-inf")

    def test_formula_correctness(self):
        E_var = -1.5
        h2 = 2.5
        E1_upper = 0.5
        variance = h2 - E_var ** 2
        expected = E_var - variance / (E1_upper - E_var)
        result = temple_lower_bound(E_var, h2, E1_upper)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_exact_gs_gives_temple_lb_equals_E0(self):
        # For exact eigenstate: variance=0, so temple_lb = E_var = E0
        gs = self.evecs[:, 0]
        E_var = energy_expectation(self.H, gs)
        h2 = h2_expectation(self.H, gs)
        E1_upper = self.E1 + 0.5
        t_lb = temple_lower_bound(E_var, h2, E1_upper)
        np.testing.assert_allclose(t_lb, self.E0, atol=1e-10)

    def test_lower_bound_leq_E0_for_random_state(self):
        # Temple bound is a lower bound: temple_lb <= E0
        rng = np.random.default_rng(1)
        state = rng.standard_normal(4)
        state /= np.linalg.norm(state)
        E_var = energy_expectation(self.H, state)
        h2 = h2_expectation(self.H, state)
        # Only meaningful when E_var < E1_exact
        if E_var < self.E1:
            t_lb = temple_lower_bound(E_var, h2, self.E1)
            assert t_lb <= self.E0 + 1e-10

    def test_zero_variance_gives_exact_E_var(self):
        # When variance=0, temple_lb = E_var regardless of E1_upper
        E_var = -3.0
        h2 = E_var ** 2  # zero variance
        E1_upper = 0.0
        result = temple_lower_bound(E_var, h2, E1_upper)
        np.testing.assert_allclose(result, E_var, rtol=1e-12)

    def test_tight_E1_upper_gives_tighter_bound(self):
        # Tighter E1_upper (closer to E_var from above) tightens denominator
        gs = self.evecs[:, 0]
        E_var = energy_expectation(self.H, gs)
        h2 = h2_expectation(self.H, gs)
        # Both should equal E_var since variance=0
        t1 = temple_lower_bound(E_var, h2, self.E1)
        t2 = temple_lower_bound(E_var, h2, self.E1 + 10.0)
        np.testing.assert_allclose(t1, t2, atol=1e-10)

    def test_large_variance_state_gives_weaker_bound(self):
        # A superposition state has nonzero variance → temple_lb < E0
        # Mix GS and first ES 50/50
        mixed = (self.evecs[:, 0] + self.evecs[:, 1]) / math.sqrt(2.0)
        E_var = energy_expectation(self.H, mixed)
        h2 = h2_expectation(self.H, mixed)
        if E_var < self.E1:
            t_lb = temple_lower_bound(E_var, h2, self.E1)
            # variance > 0, so t_lb < E_var
            variance = h2 - E_var ** 2
            assert variance > 1e-12  # confirm nonzero variance
            assert t_lb < E_var

    def test_p0_1_regression_upper_bound_can_exceed_E0(self):
        # Regression for P0-1: passing an E1 UPPER bound (larger than true E1)
        # produces a "lower bound" that may exceed the true E0.
        # E.g. H=diag(0,1,10): true E0=0, true E1=1.
        # Passing E1_upper=9.99 (≈ second Ritz value, an upper bound on E2 not E1)
        # gives a result > 0, demonstrating the formula is not rigorous here.
        H = np.diag([0.0, 1.0, 10.0])
        psi = np.array([math.sqrt(0.9), math.sqrt(0.1), 0.0])
        E_var = energy_expectation(H, psi)
        h2 = h2_expectation(H, psi)
        # Correct rigorous lower bound requires E1_lower <= E1_exact = 1.
        # Using E1_lower = 1.0 (exact) should give t_lb <= E0 = 0.
        t_lb_correct = temple_lower_bound(E_var, h2, 1.0)
        assert t_lb_correct <= 0.0 + 1e-10, "Temple with exact E1 must bound E0"
        # Using an upper bound on E1 (e.g. 9.9) can produce t_lb > E0 = 0.
        t_lb_wrong = temple_lower_bound(E_var, h2, 9.9)
        # This asserts the known-bad behaviour rather than accidentally passing:
        assert t_lb_wrong > 0.0, (
            "P0-1 regression: using E1_upper in Temple denominator "
            "should yield a value > true E0=0"
        )



class TestFirstExcitedUpper:
    def setup_method(self):
        self.H = transverse_ising_ham(2, J=1.0, h=0.5)
        self.evals, self.evecs = _eigh(self.H)
        self.E0 = float(self.evals[0])
        self.E1 = float(self.evals[1])
        self.gs = self.evecs[:, 0]
        self.es1 = self.evecs[:, 1]

    def test_returns_float(self):
        result = first_excited_upper(self.H, self.gs, self.es1)
        assert isinstance(result, float)

    def test_exact_eigenstates_gives_E1(self):
        # Using exact eigenstates: result should be very close to E1
        result = first_excited_upper(self.H, self.gs, self.es1)
        np.testing.assert_allclose(result, self.E1, rtol=1e-8)

    def test_variational_upper_bound_on_E1(self):
        # For any trial state, first_excited_upper >= E1_exact
        rng = np.random.default_rng(5)
        trial_es = rng.standard_normal(4)
        result = first_excited_upper(self.H, self.gs, trial_es)
        assert result >= self.E1 - 1e-10

    def test_parallel_state_raises(self):
        with pytest.raises(ValueError, match="parallel"):
            first_excited_upper(self.H, self.gs, self.gs)

    def test_scaled_parallel_state_raises(self):
        # A scaled version of GS is also parallel
        with pytest.raises(ValueError):
            first_excited_upper(self.H, self.gs, self.gs * 3.14)

    def test_negated_parallel_state_raises(self):
        # Negation is also parallel
        with pytest.raises(ValueError):
            first_excited_upper(self.H, self.gs, -self.gs)

    def test_zero_norm_gs_raises(self):
        with pytest.raises(ValueError, match="zero norm"):
            first_excited_upper(self.H, np.zeros(4), self.es1)

    def test_unnormalized_gs_gives_same_result(self):
        # Normalisation is applied internally
        result_norm = first_excited_upper(self.H, self.gs, self.es1)
        result_scaled = first_excited_upper(self.H, self.gs * 5.0, self.es1)
        np.testing.assert_allclose(result_norm, result_scaled, rtol=1e-10)

    def test_n4_variational_upper_bound(self):
        H4 = transverse_ising_ham(4, J=1.0, h=0.5)
        evals, evecs = _eigh(H4)
        E1_exact = float(evals[1])
        result = first_excited_upper(H4, evecs[:, 0], evecs[:, 1])
        np.testing.assert_allclose(result, E1_exact, rtol=1e-8)

    def test_result_geq_E0_exact(self):
        # Variational principle: energy_expectation >= E0_exact for any state
        rng = np.random.default_rng(99)
        trial_gs = rng.standard_normal(4)
        trial_es = rng.standard_normal(4)
        E0_exact = float(self.evals[0])
        E1_var = first_excited_upper(self.H, trial_gs, trial_es)
        assert E1_var >= E0_exact - 1e-10

    def test_accepts_2d_state_vectors(self):
        gs_2d = self.gs.reshape(4, 1)
        es_2d = self.es1.reshape(4, 1)
        result = first_excited_upper(self.H, gs_2d, es_2d)
        assert isinstance(result, float)

    def test_second_excited_trial_still_upper_bound(self):
        # Using the second excited eigenstate as trial still yields E1_var >= E1
        es2 = self.evecs[:, 2]
        result = first_excited_upper(self.H, self.gs, es2)
        assert result >= self.E1 - 1e-10


# ──────────────────────── certified_gap_upper ─────────────────────

class TestCertifiedGapUpper:
    def setup_method(self):
        self.H = transverse_ising_ham(2, J=1.0, h=0.5)
        self.evals, self.evecs = _eigh(self.H)
        self.gs = self.evecs[:, 0]
        self.es1 = self.evecs[:, 1]

    def test_returns_certificate(self):
        cert = certified_gap_upper(self.H, self.gs, self.es1)
        assert isinstance(cert, Certificate)

    def test_mode_is_certified(self):
        cert = certified_gap_upper(self.H, self.gs, self.es1)
        assert cert.mode == "certified"

    def test_error_bound_nonnegative(self):
        cert = certified_gap_upper(self.H, self.gs, self.es1)
        assert cert.error_bound is not None
        assert cert.error_bound >= 0.0

    def test_result_is_float(self):
        cert = certified_gap_upper(self.H, self.gs, self.es1)
        assert isinstance(cert.result, float)

    def test_backend_contains_flint(self):
        cert = certified_gap_upper(self.H, self.gs, self.es1)
        assert "flint" in cert.backend.lower()

    def test_notes_mention_certified(self):
        cert = certified_gap_upper(self.H, self.gs, self.es1)
        assert "certif" in cert.notes.lower()

    def test_result_close_to_exact_gap(self):
        # Using exact eigenstates: certified gap ≈ exact gap
        exact_gap = spectral_gap_exact(self.H)
        cert = certified_gap_upper(self.H, self.gs, self.es1)
        np.testing.assert_allclose(cert.result, exact_gap, rtol=1e-6)

    def test_certified_interval_covers_exact_gap(self):
        # cert.result ± cert.error_bound should cover exact gap
        exact_gap = spectral_gap_exact(self.H)
        cert = certified_gap_upper(self.H, self.gs, self.es1)
        low = cert.result - cert.error_bound
        high = cert.result + cert.error_bound
        assert low <= exact_gap + 1e-10
        assert high >= exact_gap - 1e-10

    def test_parallel_state_raises(self):
        with pytest.raises(ValueError):
            certified_gap_upper(self.H, self.gs, self.gs)

    def test_zero_norm_gs_raises(self):
        with pytest.raises(ValueError):
            certified_gap_upper(self.H, np.zeros(4), self.es1)

    def test_n4_returns_certificate(self):
        H4 = transverse_ising_ham(4, J=1.0, h=0.5)
        _evals, evecs = _eigh(H4)
        cert = certified_gap_upper(H4, evecs[:, 0], evecs[:, 1])
        assert isinstance(cert, Certificate)
        assert cert.mode == "certified"
        assert cert.error_bound >= 0.0

    def test_n4_result_close_to_exact(self):
        H4 = transverse_ising_ham(4, J=1.0, h=0.5)
        evals, evecs = _eigh(H4)
        exact_gap = float(evals[1] - evals[0])
        cert = certified_gap_upper(H4, evecs[:, 0], evecs[:, 1])
        np.testing.assert_allclose(cert.result, exact_gap, rtol=1e-5)

    def test_serializable_to_json(self):
        cert = certified_gap_upper(self.H, self.gs, self.es1)
        json_str = cert.to_json()
        assert "certified" in json_str

    def test_xx_model_returns_certificate(self):
        H = xx_model_ham(2)
        _evals, evecs = _eigh(H)
        cert = certified_gap_upper(H, evecs[:, 0], evecs[:, 1])
        assert isinstance(cert, Certificate)
        assert cert.mode == "certified"

    def test_p0_2_regression_not_a_gap_upper_bound(self):
        # Regression for P0-2: certified_gap_upper can return a value LESS THAN
        # the true spectral gap, so it is NOT an upper bound on the gap.
        # H=diag(0,1), trial states at angle 0.3: exact gap=1.
        # Using approximate (non-exact) states demonstrates the issue.
        theta = 0.3
        H = np.diag([0.0, 1.0])
        psi_gs = np.array([math.cos(theta), math.sin(theta)])
        psi_es = np.array([-math.sin(theta), math.cos(theta)])
        exact_gap = 1.0
        cert = certified_gap_upper(H, psi_gs, psi_es)
        # The trial energy difference < exact gap when states are not exact eigenstates.
        assert cert.result < exact_gap, (
            "P0-2 regression: trial energy difference must be < true gap "
            "when using approximate states, confirming it is not an upper bound"
        )
        assert "NOT a certified spectral-gap" in cert.notes


# ───────────────────────────── gap_report ─────────────────────────

class TestGapReport:
    def setup_method(self):
        self.H = transverse_ising_ham(2, J=1.0, h=0.5)
        self.evals, self.evecs = _eigh(self.H)
        self.gs = self.evecs[:, 0]
        self.es1 = self.evecs[:, 1]

    def test_returns_dict(self):
        report = gap_report(self.H, self.gs, self.es1)
        assert isinstance(report, dict)

    def test_all_keys_present(self):
        expected_keys = {"gap_exact", "E0_var", "E1_var", "gap_var", "temple_lb", "gap_cert"}
        report = gap_report(self.H, self.gs, self.es1)
        assert set(report.keys()) == expected_keys

    def test_gap_exact_nonnegative(self):
        report = gap_report(self.H, self.gs, self.es1)
        assert report["gap_exact"] >= 0.0

    def test_gap_exact_is_float(self):
        report = gap_report(self.H, self.gs, self.es1)
        assert isinstance(report["gap_exact"], float)

    def test_E0_var_is_float(self):
        report = gap_report(self.H, self.gs, self.es1)
        assert isinstance(report["E0_var"], float)

    def test_E1_var_is_float(self):
        report = gap_report(self.H, self.gs, self.es1)
        assert isinstance(report["E1_var"], float)

    def test_gap_var_equals_E1_minus_E0(self):
        report = gap_report(self.H, self.gs, self.es1)
        np.testing.assert_allclose(
            report["gap_var"],
            report["E1_var"] - report["E0_var"],
            rtol=1e-12,
        )

    def test_gap_cert_is_certificate(self):
        report = gap_report(self.H, self.gs, self.es1)
        assert isinstance(report["gap_cert"], Certificate)

    def test_gap_cert_mode_certified(self):
        report = gap_report(self.H, self.gs, self.es1)
        assert report["gap_cert"].mode == "certified"

    def test_gap_cert_error_bound_nonnegative(self):
        report = gap_report(self.H, self.gs, self.es1)
        assert report["gap_cert"].error_bound >= 0.0

    def test_E0_var_geq_E0_exact(self):
        # Variational principle: E0_var >= E0
        E0_exact = float(self.evals[0])
        report = gap_report(self.H, self.gs, self.es1)
        assert report["E0_var"] >= E0_exact - 1e-10

    def test_E1_var_geq_E1_exact(self):
        # first_excited_upper is a variational upper bound on E1
        E1_exact = float(self.evals[1])
        report = gap_report(self.H, self.gs, self.es1)
        assert report["E1_var"] >= E1_exact - 1e-10

    def test_exact_eigenstates_E0_var_close_to_E0(self):
        report = gap_report(self.H, self.gs, self.es1)
        np.testing.assert_allclose(report["E0_var"], self.evals[0], rtol=1e-8)

    def test_exact_eigenstates_gap_var_close_to_gap_exact(self):
        report = gap_report(self.H, self.gs, self.es1)
        np.testing.assert_allclose(report["gap_var"], report["gap_exact"], rtol=1e-6)

    def test_temple_lb_is_float(self):
        report = gap_report(self.H, self.gs, self.es1)
        assert isinstance(report["temple_lb"], float)

    def test_temple_lb_value_for_exact_gs(self):
        # Exact GS: variance=0 → temple_lb = E0_var
        report = gap_report(self.H, self.gs, self.es1)
        np.testing.assert_allclose(report["temple_lb"], report["E0_var"], atol=1e-10)

    def test_n4_report_all_keys(self):
        H4 = transverse_ising_ham(4, J=1.0, h=0.5)
        _evals, evecs = _eigh(H4)
        report = gap_report(H4, evecs[:, 0], evecs[:, 1])
        expected_keys = {"gap_exact", "E0_var", "E1_var", "gap_var", "temple_lb", "gap_cert"}
        assert set(report.keys()) == expected_keys

    def test_n4_gap_exact_approx_known(self):
        H4 = transverse_ising_ham(4, J=1.0, h=0.5)
        _evals, evecs = _eigh(H4)
        report = gap_report(H4, evecs[:, 0], evecs[:, 1])
        np.testing.assert_allclose(report["gap_exact"], 0.0948, atol=0.005)

    def test_gap_var_nonnegative(self):
        report = gap_report(self.H, self.gs, self.es1)
        assert report["gap_var"] >= 0.0
