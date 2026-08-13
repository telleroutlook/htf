"""Tests for htf/difficulty.py — entanglement diagnostics and difficulty map."""
from __future__ import annotations

import math

import numpy as np
import pytest

from htf.difficulty import (
    DifficultyReport,
    bipartite_entanglement_profile,
    difficulty_report,
    entanglement_entropy,
    entanglement_spectrum,
)
from htf.variational import transverse_ising_ham


# ───────────────────────────── State helpers ─────────────────────────────────

def product_state(n_sites: int) -> np.ndarray:
    """All-zeros product state |00…0>: index 0 has amplitude 1."""
    psi = np.zeros(2 ** n_sites)
    psi[0] = 1.0
    return psi


def bell_state() -> np.ndarray:
    """Two-qubit Bell state (|00> + |11>) / sqrt(2)."""
    psi = np.zeros(4)
    psi[0] = 1.0 / math.sqrt(2)
    psi[3] = 1.0 / math.sqrt(2)
    return psi


def ghz4_state() -> np.ndarray:
    """Four-qubit GHZ state (|0000> + |1111>) / sqrt(2)."""
    psi = np.zeros(16)
    psi[0] = 1.0 / math.sqrt(2)
    psi[15] = 1.0 / math.sqrt(2)
    return psi


# ─────────────────────────── entanglement_entropy ────────────────────────────

class TestEntanglementEntropy:
    def test_product_state_entropy_zero(self):
        psi = product_state(2)
        S = entanglement_entropy(psi, n_sites=2, cut=1)
        assert S == pytest.approx(0.0, abs=1e-12)

    def test_product_state_entropy_zero_n4(self):
        psi = product_state(4)
        for cut in [1, 2, 3]:
            S = entanglement_entropy(psi, n_sites=4, cut=cut)
            assert S == pytest.approx(0.0, abs=1e-12), f"cut={cut}"

    def test_bell_state_entropy_ln2(self):
        psi = bell_state()
        S = entanglement_entropy(psi, n_sites=2, cut=1)
        assert S == pytest.approx(math.log(2), abs=1e-12)

    def test_entropy_symmetric_for_reflection_symmetric_state(self):
        # For the GHZ state |0000>+|1111>/sqrt(2), the state is invariant under
        # site-index reversal, so S(cut=1) == S(cut=3).
        psi = ghz4_state()
        S1 = entanglement_entropy(psi, n_sites=4, cut=1)
        S3 = entanglement_entropy(psi, n_sites=4, cut=3)
        assert S1 == pytest.approx(S3, abs=1e-12)

    def test_entropy_symmetric_cut2(self):
        rng = np.random.default_rng(13)
        psi = rng.standard_normal(16)
        psi /= np.linalg.norm(psi)
        S2 = entanglement_entropy(psi, n_sites=4, cut=2)
        # cut=2 bipartition: S(2) == S(4-2)=S(2), trivially true; more useful:
        # cross-check that it's non-negative
        assert S2 >= 0.0

    def test_entropy_non_negative(self):
        rng = np.random.default_rng(99)
        psi = rng.standard_normal(8)
        psi /= np.linalg.norm(psi)
        for cut in [1, 2]:
            S = entanglement_entropy(psi, n_sites=3, cut=cut)
            assert S >= 0.0

    def test_entropy_upper_bounded_by_log_dim(self):
        # S(cut) <= log(min(d^cut, d^(n-cut))) = log(d) for cut=1, d=2, n=2
        psi = bell_state()
        S = entanglement_entropy(psi, n_sites=2, cut=1)
        upper = math.log(min(2 ** 1, 2 ** 1))  # = log(2)
        assert S <= upper + 1e-12

    def test_raises_value_error_cut_zero(self):
        psi = product_state(2)
        with pytest.raises(ValueError):
            entanglement_entropy(psi, n_sites=2, cut=0)

    def test_raises_value_error_cut_equals_n_sites(self):
        psi = product_state(2)
        with pytest.raises(ValueError):
            entanglement_entropy(psi, n_sites=2, cut=2)

    def test_raises_value_error_cut_negative(self):
        psi = product_state(2)
        with pytest.raises(ValueError):
            entanglement_entropy(psi, n_sites=2, cut=-1)

    def test_raises_value_error_zero_norm(self):
        psi = np.zeros(4)
        with pytest.raises(ValueError):
            entanglement_entropy(psi, n_sites=2, cut=1)

    def test_raises_value_error_non_integer_d(self):
        # Length 5 is not d^2 for any integer d (sqrt(5) ≈ 2.24)
        psi = np.ones(5) / math.sqrt(5)
        with pytest.raises(ValueError):
            entanglement_entropy(psi, n_sites=2, cut=1)

    def test_accepts_unnormalised_state(self):
        # Function normalises internally; result should match normalised version
        psi_norm = bell_state()
        psi_scaled = psi_norm * 3.0
        S_norm = entanglement_entropy(psi_norm, n_sites=2, cut=1)
        S_scaled = entanglement_entropy(psi_scaled, n_sites=2, cut=1)
        assert S_norm == pytest.approx(S_scaled, abs=1e-12)

    def test_returns_float(self):
        psi = bell_state()
        S = entanglement_entropy(psi, n_sites=2, cut=1)
        assert isinstance(S, float)


# ─────────────────────────── entanglement_spectrum ───────────────────────────

class TestEntanglementSpectrum:
    def test_shape_n2_cut1(self):
        # min(d^1, d^1) = min(2, 2) = 2
        psi = bell_state()
        sv = entanglement_spectrum(psi, n_sites=2, cut=1)
        assert sv.shape == (2,)

    def test_shape_n4_cut1(self):
        # min(d^1, d^3) = min(2, 8) = 2
        psi = ghz4_state()
        sv = entanglement_spectrum(psi, n_sites=4, cut=1)
        assert sv.shape == (2,)

    def test_shape_n4_cut2(self):
        # min(d^2, d^2) = min(4, 4) = 4
        psi = ghz4_state()
        sv = entanglement_spectrum(psi, n_sites=4, cut=2)
        assert sv.shape == (4,)

    def test_sorted_descending(self):
        rng = np.random.default_rng(5)
        psi = rng.standard_normal(8)
        psi /= np.linalg.norm(psi)
        sv = entanglement_spectrum(psi, n_sites=3, cut=1)
        for i in range(len(sv) - 1):
            assert sv[i] >= sv[i + 1]

    def test_product_state_largest_sv_is_one(self):
        psi = product_state(2)
        sv = entanglement_spectrum(psi, n_sites=2, cut=1)
        assert sv[0] == pytest.approx(1.0, abs=1e-12)

    def test_bell_state_sv_equal_half_sqrt2(self):
        psi = bell_state()
        sv = entanglement_spectrum(psi, n_sites=2, cut=1)
        expected = 1.0 / math.sqrt(2)
        assert sv[0] == pytest.approx(expected, abs=1e-12)
        assert sv[1] == pytest.approx(expected, abs=1e-12)

    def test_sv_non_negative(self):
        rng = np.random.default_rng(3)
        psi = rng.standard_normal(8)
        psi /= np.linalg.norm(psi)
        sv = entanglement_spectrum(psi, n_sites=3, cut=1)
        assert np.all(sv >= 0.0)

    def test_raises_value_error_cut_zero(self):
        psi = product_state(2)
        with pytest.raises(ValueError):
            entanglement_spectrum(psi, n_sites=2, cut=0)

    def test_raises_value_error_zero_norm(self):
        psi = np.zeros(4)
        with pytest.raises(ValueError):
            entanglement_spectrum(psi, n_sites=2, cut=1)


# ─────────────────────── bipartite_entanglement_profile ───────────────────────

class TestBipartiteEntanglementProfile:
    def test_length_n2(self):
        psi = product_state(2)
        profile = bipartite_entanglement_profile(psi, n_sites=2)
        assert len(profile) == 1  # n_sites - 1 = 1

    def test_length_n4(self):
        psi = product_state(4)
        profile = bipartite_entanglement_profile(psi, n_sites=4)
        assert len(profile) == 3  # n_sites - 1 = 3

    def test_product_state_all_zeros_n2(self):
        psi = product_state(2)
        profile = bipartite_entanglement_profile(psi, n_sites=2)
        np.testing.assert_allclose(profile, 0.0, atol=1e-12)

    def test_product_state_all_zeros_n4(self):
        psi = product_state(4)
        profile = bipartite_entanglement_profile(psi, n_sites=4)
        np.testing.assert_allclose(profile, 0.0, atol=1e-12)

    def test_bell_state_profile_contains_ln2(self):
        psi = bell_state()
        profile = bipartite_entanglement_profile(psi, n_sites=2)
        assert profile[0] == pytest.approx(math.log(2), abs=1e-12)

    def test_returns_ndarray(self):
        psi = product_state(2)
        profile = bipartite_entanglement_profile(psi, n_sites=2)
        assert isinstance(profile, np.ndarray)

    def test_non_negative_values(self):
        rng = np.random.default_rng(21)
        psi = rng.standard_normal(8)
        psi /= np.linalg.norm(psi)
        profile = bipartite_entanglement_profile(psi, n_sites=3)
        assert np.all(profile >= 0.0)

    def test_ghz4_all_cuts_have_equal_entropy(self):
        # GHZ |0000>+|1111> has S = log(2) for all cuts
        psi = ghz4_state()
        profile = bipartite_entanglement_profile(psi, n_sites=4)
        for S in profile:
            assert S == pytest.approx(math.log(2), abs=1e-12)


# ─────────────────────── DifficultyReport dataclass ──────────────────────────

class TestDifficultyReportDataclass:
    def _make_report(self) -> DifficultyReport:
        profile = np.array([0.5, 0.3])
        return DifficultyReport(
            n_sites=4,
            chi_used=2,
            energy=-1.5,
            entanglement_profile=profile,
            max_entropy=0.5,
            area_law_limit=math.log(2),
            likely_area_law=True,
            notes="test note",
        )

    def test_n_sites_stored(self):
        r = self._make_report()
        assert r.n_sites == 4

    def test_chi_used_stored(self):
        r = self._make_report()
        assert r.chi_used == 2

    def test_energy_stored(self):
        r = self._make_report()
        assert r.energy == pytest.approx(-1.5)

    def test_entanglement_profile_stored(self):
        r = self._make_report()
        np.testing.assert_allclose(r.entanglement_profile, [0.5, 0.3])

    def test_max_entropy_stored(self):
        r = self._make_report()
        assert r.max_entropy == pytest.approx(0.5)

    def test_area_law_limit_stored(self):
        r = self._make_report()
        assert r.area_law_limit == pytest.approx(math.log(2))

    def test_likely_area_law_is_bool(self):
        r = self._make_report()
        assert isinstance(r.likely_area_law, bool)

    def test_likely_area_law_true_when_s_below_limit(self):
        profile = np.array([0.1])
        r = DifficultyReport(
            n_sites=2, chi_used=2, energy=-1.0,
            entanglement_profile=profile,
            max_entropy=0.1,
            area_law_limit=math.log(2),
            likely_area_law=0.1 < math.log(2),
        )
        assert r.likely_area_law is True

    def test_notes_stored(self):
        r = self._make_report()
        assert "test note" in r.notes

    def test_summary_returns_string(self):
        r = self._make_report()
        assert isinstance(r.summary(), str)

    def test_summary_contains_n_sites(self):
        r = self._make_report()
        assert "4" in r.summary()

    def test_summary_contains_chi(self):
        r = self._make_report()
        assert "2" in r.summary()

    def test_summary_contains_energy(self):
        r = self._make_report()
        assert "-1.5" in r.summary()

    def test_summary_contains_regime_label_area_law(self):
        r = self._make_report()
        assert "area-law" in r.summary()

    def test_summary_heuristic_label(self):
        r = self._make_report()
        assert "heuristic" in r.summary()


# ─────────────────────────── difficulty_report ───────────────────────────────

class TestDifficultyReportFunction:
    def test_returns_difficulty_report_type(self):
        ham = transverse_ising_ham(2)
        result = difficulty_report(ham, n_sites=2, n_iter=5, seed=0)
        assert isinstance(result, DifficultyReport)

    def test_chi_used_is_2_for_qubits(self):
        # transverse_ising_ham(2) has shape (4, 4) → dim=4=2^2 → chi=2
        ham = transverse_ising_ham(2)
        result = difficulty_report(ham, n_sites=2, n_iter=5, seed=0)
        assert result.chi_used == 2

    def test_n_sites_stored(self):
        ham = transverse_ising_ham(2)
        result = difficulty_report(ham, n_sites=2, n_iter=5, seed=0)
        assert result.n_sites == 2

    def test_entanglement_profile_length(self):
        # n_sites=2 → profile length = 1
        ham = transverse_ising_ham(2)
        result = difficulty_report(ham, n_sites=2, n_iter=5, seed=0)
        assert len(result.entanglement_profile) == 1

    def test_max_entropy_non_negative(self):
        ham = transverse_ising_ham(2)
        result = difficulty_report(ham, n_sites=2, n_iter=5, seed=0)
        assert result.max_entropy >= 0.0

    def test_area_law_limit_equals_log_chi(self):
        ham = transverse_ising_ham(2)
        result = difficulty_report(ham, n_sites=2, n_iter=5, seed=0)
        assert result.area_law_limit == pytest.approx(math.log(2), abs=1e-12)

    def test_likely_area_law_is_bool(self):
        ham = transverse_ising_ham(2)
        result = difficulty_report(ham, n_sites=2, n_iter=5, seed=0)
        assert isinstance(result.likely_area_law, bool)

    def test_likely_area_law_consistent_with_max_entropy(self):
        ham = transverse_ising_ham(2)
        result = difficulty_report(ham, n_sites=2, n_iter=5, seed=0)
        expected = result.max_entropy < result.area_law_limit
        assert result.likely_area_law == expected

    def test_energy_is_float(self):
        ham = transverse_ising_ham(2)
        result = difficulty_report(ham, n_sites=2, n_iter=5, seed=0)
        assert isinstance(result.energy, float)

    def test_notes_non_empty(self):
        ham = transverse_ising_ham(2)
        result = difficulty_report(ham, n_sites=2, n_iter=5, seed=0)
        assert result.notes != ""

    def test_raises_value_error_when_dims_dont_factor(self):
        # dim=6, n_sites=2: round(sqrt(6))=2, 2^2=4 ≠ 6 → ValueError
        ham = np.eye(6)
        with pytest.raises(ValueError):
            difficulty_report(ham, n_sites=2, n_iter=3, seed=0)

    def test_raises_value_error_dim5_n_sites2(self):
        # dim=5, n_sites=2: round(sqrt(5))=2, 2^2=4 ≠ 5 → ValueError
        ham = np.eye(5)
        with pytest.raises(ValueError):
            difficulty_report(ham, n_sites=2, n_iter=3, seed=0)

    def test_summary_method_callable_on_result(self):
        ham = transverse_ising_ham(2)
        result = difficulty_report(ham, n_sites=2, n_iter=5, seed=0)
        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_profile_values_non_negative(self):
        ham = transverse_ising_ham(2)
        result = difficulty_report(ham, n_sites=2, n_iter=5, seed=0)
        assert np.all(result.entanglement_profile >= 0.0)

    def test_max_entropy_equals_max_of_profile(self):
        ham = transverse_ising_ham(2)
        result = difficulty_report(ham, n_sites=2, n_iter=5, seed=0)
        assert result.max_entropy == pytest.approx(
            float(np.max(result.entanglement_profile)), abs=1e-12
        )
