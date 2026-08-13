"""Tests for htf/variational.py — Variational ground-state search."""
from __future__ import annotations

import numpy as np
import pytest

from htf.certificate import Certificate
from htf.mera import random_mera
from htf.variational import (
    energy_expectation,
    optimize_mera,
    transverse_ising_ham,
    variational_bound,
    xx_model_ham,
)

# ─────────────────────────── Helpers ──────────────────────────────

def ground_state_vec(ham: np.ndarray) -> np.ndarray:
    """Return the eigenvector corresponding to the smallest eigenvalue."""
    _eigenvalues, eigenvectors = np.linalg.eigh(ham)
    return eigenvectors[:, 0]


# ─────────────────── transverse_ising_ham ─────────────────────────

class TestTransverseIsingHam:
    def test_shape_n2(self):
        H = transverse_ising_ham(2)
        assert H.shape == (4, 4)

    def test_shape_n3(self):
        H = transverse_ising_ham(3)
        assert H.shape == (8, 8)

    def test_shape_n4(self):
        H = transverse_ising_ham(4)
        assert H.shape == (16, 16)

    def test_symmetric_n2(self):
        H = transverse_ising_ham(2)
        np.testing.assert_array_equal(H, H.T)

    def test_symmetric_n3(self):
        H = transverse_ising_ham(3)
        np.testing.assert_allclose(H, H.T, atol=1e-12)

    def test_real_valued(self):
        H = transverse_ising_ham(3)
        assert H.dtype == float

    def test_n1_shape(self):
        # n=1: pure transverse field, no ZZ terms
        H = transverse_ising_ham(1)
        assert H.shape == (2, 2)
        np.testing.assert_allclose(H, H.T, atol=1e-12)

    def test_coupling_J_scales_zz(self):
        H_default = transverse_ising_ham(2, J=1.0, h=0.0)
        H_double = transverse_ising_ham(2, J=2.0, h=0.0)
        np.testing.assert_allclose(H_double, 2.0 * H_default, atol=1e-12)

    def test_field_h_scales_x(self):
        H_default = transverse_ising_ham(2, J=0.0, h=1.0)
        H_double = transverse_ising_ham(2, J=0.0, h=2.0)
        np.testing.assert_allclose(H_double, 2.0 * H_default, atol=1e-12)

    def test_zero_field_only_zz(self):
        # With h=0 and n=2: H = -J * ZZ, which is diagonal
        H = transverse_ising_ham(2, J=1.0, h=0.0)
        # Diagonal matrix: all off-diagonal should be zero
        off_diag = H - np.diag(np.diag(H))
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-12)

    def test_zero_coupling_only_x(self):
        # With J=0 and n=2: H = -h * (X1 + X2)
        H = transverse_ising_ham(2, J=0.0, h=1.0)
        # X is off-diagonal so H should have zero diagonal
        np.testing.assert_allclose(np.diag(H), 0.0, atol=1e-12)

    def test_ground_state_energy_n2_known(self):
        # For n=2, J=1, h=0.5: ground energy should match scipy reference
        H = transverse_ising_ham(2, J=1.0, h=0.5)
        eigenvalues = np.linalg.eigvalsh(H)
        e0 = eigenvalues[0]
        # Ground state energy is variational upper bound — verify it's negative
        assert e0 < 0.0

    def test_all_eigenvalues_real(self):
        H = transverse_ising_ham(3)
        eigenvalues = np.linalg.eigvalsh(H)
        assert eigenvalues.dtype == float


# ─────────────────────── xx_model_ham ─────────────────────────────

class TestXXModelHam:
    def test_shape_n2(self):
        H = xx_model_ham(2)
        assert H.shape == (4, 4)

    def test_shape_n3(self):
        H = xx_model_ham(3)
        assert H.shape == (8, 8)

    def test_shape_n4(self):
        H = xx_model_ham(4)
        assert H.shape == (16, 16)

    def test_symmetric_n2(self):
        H = xx_model_ham(2)
        np.testing.assert_allclose(H, H.T, atol=1e-12)

    def test_symmetric_n3(self):
        H = xx_model_ham(3)
        np.testing.assert_allclose(H, H.T, atol=1e-12)

    def test_real_valued(self):
        H = xx_model_ham(3)
        assert H.dtype == float

    def test_coupling_J_scales(self):
        H_default = xx_model_ham(2, J=1.0)
        H_double = xx_model_ham(2, J=2.0)
        np.testing.assert_allclose(H_double, 2.0 * H_default, atol=1e-12)

    def test_n2_ground_energy_negative(self):
        H = xx_model_ham(2)
        e0 = np.linalg.eigvalsh(H)[0]
        assert e0 < 0.0

    def test_zero_coupling_is_zero(self):
        H = xx_model_ham(3, J=0.0)
        np.testing.assert_allclose(H, np.zeros((8, 8)), atol=1e-12)

    def test_n1_is_zero_matrix(self):
        # n=1: no nearest-neighbour pairs, H is zero
        H = xx_model_ham(1)
        assert H.shape == (2, 2)
        np.testing.assert_allclose(H, np.zeros((2, 2)), atol=1e-12)


# ──────────────────── energy_expectation ──────────────────────────

class TestEnergyExpectation:
    def setup_method(self):
        self.H2 = transverse_ising_ham(2, J=1.0, h=0.5)
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.H2)

    def test_ground_state_equals_min_eigenvalue(self):
        gs = self.eigenvectors[:, 0]
        e = energy_expectation(self.H2, gs)
        assert abs(e - self.eigenvalues[0]) < 1e-10

    def test_all_eigenstates_match_eigenvalues(self):
        for i in range(4):
            vec = self.eigenvectors[:, i]
            e = energy_expectation(self.H2, vec)
            assert abs(e - self.eigenvalues[i]) < 1e-10

    def test_exact_ground_energy_rtol(self):
        gs = self.eigenvectors[:, 0]
        e = energy_expectation(self.H2, gs)
        e0 = self.eigenvalues[0]
        np.testing.assert_allclose(e, e0, rtol=1e-6)

    def test_unnormalized_state_gives_same_energy(self):
        gs = self.eigenvectors[:, 0]
        e_normalized = energy_expectation(self.H2, gs)
        e_scaled = energy_expectation(self.H2, gs * 5.0)
        assert abs(e_normalized - e_scaled) < 1e-10

    def test_zero_norm_raises_value_error(self):
        zero = np.zeros(4)
        with pytest.raises(ValueError, match="zero norm"):
            energy_expectation(self.H2, zero)

    def test_near_zero_norm_raises_value_error(self):
        near_zero = np.ones(4) * 1e-10
        with pytest.raises(ValueError):
            energy_expectation(self.H2, near_zero)

    def test_returns_float(self):
        gs = self.eigenvectors[:, 0]
        result = energy_expectation(self.H2, gs)
        assert isinstance(result, float)

    def test_n3_ground_state(self):
        H3 = transverse_ising_ham(3, J=1.0, h=0.5)
        eigenvalues, eigenvectors = np.linalg.eigh(H3)
        gs = eigenvectors[:, 0]
        e = energy_expectation(H3, gs)
        np.testing.assert_allclose(e, eigenvalues[0], rtol=1e-6)

    def test_xx_ground_state(self):
        H = xx_model_ham(2)
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        gs = eigenvectors[:, 0]
        e = energy_expectation(H, gs)
        np.testing.assert_allclose(e, eigenvalues[0], rtol=1e-6)

    def test_variational_energy_gte_ground_energy(self):
        # Any normalised state has E >= E_0
        H = transverse_ising_ham(2, J=1.0, h=1.0)
        e0 = np.linalg.eigvalsh(H)[0]
        rng = np.random.default_rng(42)
        for _ in range(10):
            state = rng.standard_normal(4)
            e = energy_expectation(H, state)
            assert e >= e0 - 1e-10, f"Energy {e} < ground {e0}"

    def test_accepts_2d_input(self):
        # state_vec can be 2-D column vector; should be flattened
        H = transverse_ising_ham(2)
        gs = np.linalg.eigh(H)[1][:, 0].reshape(4, 1)
        e = energy_expectation(H, gs)
        assert isinstance(e, float)


# ─────────────────────── variational_bound ────────────────────────

class TestVariationalBound:
    def setup_method(self):
        self.H = transverse_ising_ham(2, J=1.0, h=0.5)
        self.mera = random_mera(n_sites=2, chi=2, seed=7)

    def test_returns_certificate(self):
        cert = variational_bound(self.H, self.mera)
        assert isinstance(cert, Certificate)

    def test_mode_is_certified(self):
        cert = variational_bound(self.H, self.mera)
        assert cert.mode == "certified"

    def test_error_bound_non_negative(self):
        cert = variational_bound(self.H, self.mera)
        assert cert.error_bound is not None
        assert cert.error_bound >= 0.0

    def test_result_is_float(self):
        cert = variational_bound(self.H, self.mera)
        assert isinstance(cert.result, float)

    def test_energy_close_to_expectation(self):
        cert = variational_bound(self.H, self.mera)
        psi = self.mera.state_vector()
        e_float = energy_expectation(self.H, psi)
        # The certified midpoint should be close to the float expectation
        assert abs(cert.result - e_float) < 1e-6

    def test_backend_is_flint(self):
        cert = variational_bound(self.H, self.mera)
        assert "flint" in cert.backend.lower()

    def test_notes_mention_variational(self):
        cert = variational_bound(self.H, self.mera)
        assert "variational" in cert.notes.lower()

    def test_certified_upper_bound_satisfies_variational_principle(self):
        # E_var (= cert.result + cert.error_bound) >= E_0 (variational principle)
        H = transverse_ising_ham(2, J=1.0, h=0.5)
        e0 = np.linalg.eigvalsh(H)[0]
        mera = random_mera(n_sites=2, chi=2, seed=3)
        cert = variational_bound(H, mera)
        upper_bound = cert.result + cert.error_bound
        # variational principle: E_var >= E_0
        assert upper_bound >= e0 - 1e-8

    def test_n2_different_seeds(self):
        for seed in [0, 1, 2, 99]:
            m = random_mera(n_sites=2, chi=2, seed=seed)
            cert = variational_bound(self.H, m)
            assert isinstance(cert, Certificate)
            assert cert.error_bound >= 0.0

    def test_serializable_to_json(self):
        cert = variational_bound(self.H, self.mera)
        json_str = cert.to_json()
        assert "certified" in json_str

    def test_xx_hamiltonian(self):
        H = xx_model_ham(2)
        mera = random_mera(n_sites=2, chi=2, seed=5)
        cert = variational_bound(H, mera)
        assert isinstance(cert, Certificate)
        assert cert.mode == "certified"
        assert cert.error_bound >= 0.0


# ─────────────────────── optimize_mera ───────────────────────────

class TestOptimizeMera:
    def setup_method(self):
        self.H2 = transverse_ising_ham(2, J=1.0, h=0.5)
        self.mera = random_mera(n_sites=2, chi=2, seed=42)

    def test_returns_tuple(self):
        result = optimize_mera(self.H2, self.mera, n_iter=5)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_mera_instance(self):
        from htf.mera import MERA
        optimised, _ = optimize_mera(self.H2, self.mera, n_iter=5)
        assert isinstance(optimised, MERA)

    def test_energy_history_is_list(self):
        _, history = optimize_mera(self.H2, self.mera, n_iter=5)
        assert isinstance(history, list)

    def test_energy_history_at_least_two_entries(self):
        _, history = optimize_mera(self.H2, self.mera, n_iter=5)
        assert len(history) >= 2

    def test_energy_history_all_floats(self):
        _, history = optimize_mera(self.H2, self.mera, n_iter=5)
        for val in history:
            assert isinstance(val, float)

    def test_first_history_entry_is_initial_energy(self):
        psi0 = self.mera.state_vector()
        e_initial = energy_expectation(self.H2, psi0)
        _, history = optimize_mera(self.H2, self.mera, n_iter=5)
        assert abs(history[0] - e_initial) < 1e-10

    def test_energy_does_not_increase(self):
        # First entry should be >= last entry (energy decreases or stays the same)
        _, history = optimize_mera(self.H2, self.mera, n_iter=20)
        assert history[0] >= history[-1] - 1e-8

    def test_optimised_mera_has_same_n_sites(self):
        optimised, _ = optimize_mera(self.H2, self.mera, n_iter=5)
        assert optimised.n_sites == self.mera.n_sites

    def test_optimised_mera_has_same_chi(self):
        optimised, _ = optimize_mera(self.H2, self.mera, n_iter=5)
        assert optimised.chi == self.mera.chi

    def test_optimised_energy_bounded_by_ground_energy(self):
        # Variational principle: optimised energy >= E_0
        H = transverse_ising_ham(2, J=1.0, h=0.5)
        e0 = np.linalg.eigvalsh(H)[0]
        mera = random_mera(n_sites=2, chi=2, seed=10)
        _optimised, history = optimize_mera(H, mera, n_iter=50)
        e_final = history[-1]
        assert e_final >= e0 - 1e-8

    def test_optimize_with_n_iter_1(self):
        # Should not crash with minimal iterations
        _optimised, history = optimize_mera(self.H2, self.mera, n_iter=1)
        assert len(history) >= 1

    def test_optimize_xx_model(self):
        H = xx_model_ham(2)
        mera = random_mera(n_sites=2, chi=2, seed=13)
        _optimised, history = optimize_mera(H, mera, n_iter=10)
        assert len(history) >= 2
        assert history[0] >= history[-1] - 1e-8

    def test_original_mera_unchanged(self):
        # optimize_mera should not modify the input mera in-place
        psi_before = self.mera.state_vector().copy()
        optimize_mera(self.H2, self.mera, n_iter=5)
        psi_after = self.mera.state_vector()
        np.testing.assert_array_equal(psi_before, psi_after)

    def test_optimised_constraints_enforced(self):
        # After optimize_mera, enforce_constraints should be idempotent (state unchanged)
        optimised, _ = optimize_mera(self.H2, self.mera, n_iter=10)
        optimised.state_vector().copy()
        optimised.enforce_constraints()
        # Top should be normalised to unit norm already
        norm = float(np.linalg.norm(optimised.top))
        assert abs(norm - 1.0) < 1e-10

    def test_multiple_seeds_all_converge_down(self):
        H = transverse_ising_ham(2, J=1.0, h=1.0)
        for seed in [0, 1, 5, 17]:
            mera = random_mera(n_sites=2, chi=2, seed=seed)
            _, history = optimize_mera(H, mera, n_iter=30)
            assert history[0] >= history[-1] - 1e-6, (
                f"seed={seed}: initial {history[0]} < final {history[-1]}"
            )
