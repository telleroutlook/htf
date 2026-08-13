"""Tests for htf/inverse.py — differentiable inverse design and Hamiltonian learning."""
import numpy as np
import pytest

from htf.inverse import (
    InverseDesignResult,
    LearningResult,
    ParametricHam,
    energy_gradient,
    hamiltonian_learning,
    inverse_design,
)
from htf.variational import transverse_ising_ham, xx_model_ham

# ─────────────────────── TestParametricHam ───────────────────────────────

class TestParametricHam:

    def test_ising_default_param_names(self):
        ph = ParametricHam("ising", 4)
        assert ph.param_names == ["J", "h"]

    def test_xx_default_param_names(self):
        ph = ParametricHam("xx", 4)
        assert ph.param_names == ["J"]

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            ParametricHam("heisenberg", 4)

    def test_ising_ham_matches_variational_module(self):
        ph = ParametricHam("ising", 4)
        H_ph = ph.ham([1.0, 0.5])
        H_ref = transverse_ising_ham(4, J=1.0, h=0.5)
        assert np.allclose(H_ph, H_ref, atol=1e-12)

    def test_xx_ham_matches_variational_module(self):
        ph = ParametricHam("xx", 4)
        H_ph = ph.ham([2.0])
        H_ref = xx_model_ham(4, J=2.0)
        assert np.allclose(H_ph, H_ref, atol=1e-12)

    def test_ground_energy_matches_eigvalsh(self):
        ph = ParametricHam("ising", 4)
        e0 = ph.ground_energy([1.0, 0.5])
        exact = float(np.linalg.eigvalsh(transverse_ising_ham(4, J=1.0, h=0.5))[0])
        assert abs(e0 - exact) < 1e-12

    def test_spectrum_length(self):
        ph = ParametricHam("ising", 4)
        spec = ph.spectrum([1.0, 0.5], k=3)
        assert len(spec) == 3

    def test_spectrum_sorted_ascending(self):
        ph = ParametricHam("ising", 4)
        spec = ph.spectrum([1.0, 0.5], k=4)
        assert np.all(spec[:-1] <= spec[1:] + 1e-15)

    def test_n_params_ising(self):
        assert ParametricHam("ising", 4).n_params() == 2

    def test_n_params_xx(self):
        assert ParametricHam("xx", 4).n_params() == 1


# ─────────────────────── TestInverseDesign ───────────────────────────────

class TestInverseDesign:

    @classmethod
    def setup_class(cls):
        # Pre-compute a target E0 from known params so we have a ground truth
        cls.true_J, cls.true_h = 1.5, 0.8
        cls.true_e0 = float(
            np.linalg.eigvalsh(transverse_ising_ham(4, J=cls.true_J, h=cls.true_h))[0]
        )
        cls.result = inverse_design(
            cls.true_e0, model="ising", n_sites=4,
            n_restarts=3, seed=7, tol=1e-8,
        )

    def test_returns_result_type(self):
        assert isinstance(self.result, InverseDesignResult)

    def test_residual_small(self):
        assert self.result.residual < 1e-5

    def test_e0_achieved_close_to_target(self):
        assert abs(self.result.E0_achieved - self.result.E0_target) < 1e-5

    def test_param_names_ising(self):
        assert self.result.param_names == ["J", "h"]

    def test_n_restarts_recorded(self):
        assert self.result.n_restarts >= 3

    def test_notes_not_empty(self):
        assert len(self.result.notes) > 0

    def test_params_within_bounds(self):
        p = self.result.params_opt
        assert len(p) == 2
        assert all(v > 0 for v in p)

    def test_xx_model(self):
        true_e0 = float(np.linalg.eigvalsh(xx_model_ham(4, J=1.0))[0])
        r = inverse_design(true_e0, model="xx", n_sites=4, n_restarts=3, seed=1)
        assert r.residual < 1e-4

    def test_x0_kwarg_used(self):
        r = inverse_design(
            self.true_e0, model="ising", n_sites=4,
            x0=np.array([self.true_J, self.true_h]), n_restarts=1, seed=0,
        )
        assert r.residual < 1e-6

    def test_achieves_target_energy_via_parametric_ham(self):
        ph = ParametricHam("ising", 4)
        e_check = ph.ground_energy(self.result.params_opt)
        assert abs(e_check - self.true_e0) < 1e-5


# ─────────────────────── TestHamiltonianLearning ─────────────────────────

class TestHamiltonianLearning:

    @classmethod
    def setup_class(cls):
        cls.true_J, cls.true_h = 1.2, 0.6
        H_true = transverse_ising_ham(4, J=cls.true_J, h=cls.true_h)
        cls.target = np.linalg.eigvalsh(H_true)[:3]
        cls.result = hamiltonian_learning(
            cls.target, model="ising", n_sites=4,
            n_restarts=3, seed=42, tol=1e-8,
        )

    def test_returns_result_type(self):
        assert isinstance(self.result, LearningResult)

    def test_loss_final_small(self):
        assert self.result.loss_final < 1e-6

    def test_achieved_energies_close_to_target(self):
        assert np.allclose(self.result.achieved_energies, self.target, atol=1e-4)

    def test_target_energies_stored(self):
        assert np.allclose(self.result.target_energies, self.target)

    def test_param_names_ising(self):
        assert self.result.param_names == ["J", "h"]

    def test_n_restarts_recorded(self):
        assert self.result.n_restarts >= 3

    def test_notes_not_empty(self):
        assert len(self.result.notes) > 0

    def test_single_energy_level(self):
        # Learning from ground state only — E0 constraint
        true_e0 = float(np.linalg.eigvalsh(xx_model_ham(4, J=2.0))[0])
        r = hamiltonian_learning([true_e0], model="xx", n_sites=4, n_restarts=2, seed=5)
        assert r.loss_final < 1e-4

    def test_xx_learning(self):
        H_true = xx_model_ham(4, J=1.5)
        tgt    = np.linalg.eigvalsh(H_true)[:2]
        r = hamiltonian_learning(tgt, model="xx", n_sites=4, n_restarts=3, seed=3)
        assert r.loss_final < 1e-6


# ─────────────────────── TestEnergyGradient ──────────────────────────────

class TestEnergyGradient:

    def test_gradient_shape(self):
        ph = ParametricHam("ising", 4)
        g  = energy_gradient([1.0, 0.5], ph)
        assert g.shape == (2,)

    def test_gradient_sign_wrt_J(self):
        # Ground energy of TFIM decreases as J increases (deeper FM well)
        ph = ParametricHam("ising", 4)
        g  = energy_gradient([1.0, 0.5], ph)
        assert g[0] < 0  # dE0/dJ < 0 for TFIM with h < J

    def test_gradient_is_finite(self):
        ph = ParametricHam("ising", 4)
        g  = energy_gradient([1.0, 0.5], ph)
        assert np.all(np.isfinite(g))

    def test_gradient_xx_single_param(self):
        ph = ParametricHam("xx", 4)
        g  = energy_gradient([1.0], ph)
        assert g.shape == (1,)
        assert np.isfinite(g[0])

    def test_gradient_consistent_with_finite_diff(self):
        ph  = ParametricHam("ising", 4)
        p   = np.array([1.0, 0.5])
        eps = 1e-5
        g   = energy_gradient(p, ph, eps=eps)
        # Manual FD check for first component
        gJ  = (ph.ground_energy(p + np.array([eps, 0])) -
               ph.ground_energy(p - np.array([eps, 0]))) / (2 * eps)
        assert abs(g[0] - gJ) < 1e-8

    def test_gradient_zero_at_sym_point(self):
        # At J=0 TFIM is pure transverse field, E0 = -n*h/2... gradient in h still finite
        ph = ParametricHam("ising", 4)
        g  = energy_gradient([0.01, 1.0], ph)
        assert np.all(np.isfinite(g))
