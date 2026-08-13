"""Tests for htf/lanczos.py — Lanczos algorithm and two-sided spectral bounds."""
import numpy as np

from htf.lanczos import (
    TwoSidedBounds,
    lanczos,
    lanczos_eigs,
    lanczos_ground_state,
    temple_lanczos,
    two_sided_bounds,
)
from htf.variational import transverse_ising_ham, xx_model_ham

# ─────── test helpers ─────────────────────────────────────────────────────

def _diag_ham(n: int) -> np.ndarray:
    """Simple diagonal Hamiltonian: H = diag(0, 1, 2, ..., n-1)."""
    return np.diag(np.arange(n, dtype=float))


def _random_sym(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return (A + A.T) / 2


# ─────────────────── TestLanczos ──────────────────────────────────────────

class TestLanczos:

    def test_returns_three_arrays(self):
        H = _diag_ham(4)
        v0 = np.ones(4) / 2
        alpha, beta, V = lanczos(H, v0, k=3)
        assert alpha.ndim == 1
        assert beta.ndim == 1
        assert V.ndim == 2

    def test_alpha_length_equals_k(self):
        H = _diag_ham(6)
        v0 = np.random.default_rng(0).standard_normal(6)
        alpha, _beta, _V = lanczos(H, v0, k=4)
        assert len(alpha) == 4

    def test_beta_length_is_k_minus_1(self):
        H = _diag_ham(6)
        v0 = np.random.default_rng(0).standard_normal(6)
        _alpha, beta, _V = lanczos(H, v0, k=4)
        assert len(beta) == 3

    def test_V_columns_orthonormal(self):
        H = _diag_ham(8)
        v0 = np.random.default_rng(1).standard_normal(8)
        _alpha, _beta, V = lanczos(H, v0, k=5)
        gram = V.T @ V
        assert np.allclose(gram, np.eye(gram.shape[0]), atol=1e-10)

    def test_k_capped_at_n(self):
        H = _diag_ham(3)
        v0 = np.ones(3) / np.sqrt(3)
        alpha, _beta, _V = lanczos(H, v0, k=100)
        assert len(alpha) <= 3

    def test_v0_normalised_internally(self):
        H = _diag_ham(4)
        v0_scaled = 5.0 * np.array([1.0, 0.0, 0.0, 0.0])
        v0_unit   = np.array([1.0, 0.0, 0.0, 0.0])
        a1, _b1, _V1 = lanczos(H, v0_scaled, k=3)
        a2, _b2, _V2 = lanczos(H, v0_unit,   k=3)
        assert np.allclose(a1, a2, atol=1e-12)

    def test_tridiagonal_reconstructs_projection(self):
        H  = _random_sym(6)
        v0 = np.random.default_rng(7).standard_normal(6)
        k  = 4
        alpha, beta, V = lanczos(H, v0, k)
        len(alpha)
        T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
        # V^T A V should equal T
        assert np.allclose(V.T @ H @ V, T, atol=1e-8)

    def test_invariant_subspace_early_stop(self):
        # H diagonal, v0 = e_0 — subspace is invariant after 1 step
        H = np.diag([0.0, 1.0, 2.0, 3.0])
        v0 = np.array([1.0, 0.0, 0.0, 0.0])
        alpha, _beta, _V = lanczos(H, v0, k=4)
        # Should stop early (only 1 effective step for eigenvector v0)
        assert len(alpha) <= 2


# ─────────────────── TestLanczosEigs ─────────────────────────────────────

class TestLanczosEigs:

    def test_ritz_values_are_variational_upper_bounds(self):
        H = _diag_ham(8)
        exact_evals = np.sort(np.linalg.eigvalsh(H))
        ritz_vals, _ = lanczos_eigs(H, k=5, seed=42)
        # Lowest Ritz value ≥ lowest exact eigenvalue
        assert ritz_vals[0] >= exact_evals[0] - 1e-10

    def test_ritz_values_converge_to_exact(self):
        # With k=n, Lanczos is exact (up to floating point)
        H = _random_sym(6)
        exact_evals = np.sort(np.linalg.eigvalsh(H))
        ritz_vals, _ = lanczos_eigs(H, k=6, seed=0)
        assert np.allclose(np.sort(ritz_vals), exact_evals, atol=1e-8)

    def test_returns_sorted_ritz_values(self):
        H = _random_sym(8)
        ritz_vals, _ = lanczos_eigs(H, k=5, seed=0)
        assert np.all(ritz_vals[:-1] <= ritz_vals[1:] + 1e-15)

    def test_ritz_vecs_approximately_normalised(self):
        H = _random_sym(6)
        _, ritz_vecs = lanczos_eigs(H, k=4, seed=0)
        norms = np.linalg.norm(ritz_vecs, axis=0)
        assert np.allclose(norms, 1.0, atol=1e-8)

    def test_v0_kwarg_accepted(self):
        H = _diag_ham(6)
        v0 = np.ones(6) / np.sqrt(6)
        ritz_vals, _ = lanczos_eigs(H, v0=v0, k=4)
        assert ritz_vals[0] >= 0.0 - 1e-10

    def test_smallest_ritz_close_to_ground_state(self):
        H = transverse_ising_ham(4)
        exact_e0 = np.linalg.eigvalsh(H)[0]
        ritz_vals, _ = lanczos_eigs(H, k=8, seed=0)
        assert abs(ritz_vals[0] - exact_e0) < 0.1


# ─────────────────── TestLanczosGroundState ──────────────────────────────

class TestLanczosGroundState:

    def test_normalised(self):
        H = transverse_ising_ham(4)
        psi = lanczos_ground_state(H, k=10)
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-12

    def test_energy_close_to_exact(self):
        H = transverse_ising_ham(4)
        exact_e0 = np.linalg.eigvalsh(H)[0]
        psi = lanczos_ground_state(H, k=10, seed=0)
        E_ritz = float(psi @ H @ psi)
        assert E_ritz >= exact_e0 - 1e-10   # variational
        assert abs(E_ritz - exact_e0) < 0.05

    def test_ising_critical(self):
        H = transverse_ising_ham(4, J=1.0, h=1.0)
        exact_e0 = np.linalg.eigvalsh(H)[0]
        psi = lanczos_ground_state(H, k=12, seed=1)
        E_ritz = float(psi @ H @ psi)
        assert E_ritz >= exact_e0 - 1e-9

    def test_xx_model(self):
        H = xx_model_ham(4)
        exact_e0 = np.linalg.eigvalsh(H)[0]
        psi = lanczos_ground_state(H, k=10, seed=2)
        E_ritz = float(psi @ H @ psi)
        assert E_ritz >= exact_e0 - 1e-9


# ─────────────────── TestTwoSidedBounds ──────────────────────────────────

class TestTwoSidedBounds:

    @classmethod
    def setup_class(cls):
        cls.H = transverse_ising_ham(4)
        cls.exact = np.linalg.eigvalsh(cls.H)
        cls.bounds = temple_lanczos(cls.H, k=12, seed=0)

    def test_returns_two_sided_bounds(self):
        assert isinstance(self.bounds, TwoSidedBounds)

    def test_upper_bound_above_exact_e0(self):
        assert self.bounds.E0_upper >= self.exact[0] - 1e-10

    def test_lower_bound_below_exact_e0(self):
        if self.bounds.temple_condition_met:
            assert self.bounds.E0_lower <= self.exact[0] + 1e-10

    def test_lower_bound_below_upper_bound(self):
        if self.bounds.temple_condition_met:
            assert self.bounds.E0_lower <= self.bounds.E0_upper + 1e-12

    def test_width_positive_or_inf(self):
        w = self.bounds.width
        assert w >= 0 or w == float("inf")

    def test_e1_ritz_above_exact_e0(self):
        # Ritz value for E1 is variational upper bound
        assert self.bounds.E1_ritz >= self.exact[0] - 1e-10

    def test_k_lanczos_reported(self):
        assert self.bounds.k_lanczos >= 1

    def test_notes_not_empty(self):
        assert len(self.bounds.notes) > 0

    def test_alias_two_sided_bounds(self):
        b = two_sided_bounds(self.H, k=8, seed=0)
        assert isinstance(b, TwoSidedBounds)

    def test_xx_model_gives_valid_bounds(self):
        H = xx_model_ham(4)
        exact_e0 = np.linalg.eigvalsh(H)[0]
        b = temple_lanczos(H, k=10, seed=3)
        assert b.E0_upper >= exact_e0 - 1e-9
        if b.temple_condition_met:
            assert b.E0_lower <= exact_e0 + 1e-9

    def test_width_shrinks_with_more_steps(self):
        H = transverse_ising_ham(4)
        b_small = temple_lanczos(H, k=4,  seed=0)
        b_large = temple_lanczos(H, k=12, seed=0)
        # More steps → tighter bounds (width should not increase)
        if b_small.temple_condition_met and b_large.temple_condition_met:
            assert b_large.width <= b_small.width + 1e-10

    def test_diagonal_ham_exact_at_k_n(self):
        n  = 4
        H  = _diag_ham(n)
        b  = temple_lanczos(H, k=n, seed=0)
        assert abs(b.E0_upper - 0.0) < 1e-8   # E_0 = 0
