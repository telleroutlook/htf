"""Comprehensive tests for htf/mera.py — binary MERA tensor network."""
from __future__ import annotations

import numpy as np
import pytest

from htf.mera import (
    MERA,
    MERALayer,
    _apply_isometry_adjoint,
    _apply_unitary_adjoint,
    random_mera,
)
from htf.structure import isometry_defect, unitary_defect

# ──────────────────── helpers ─────────────────────────────────────────

def _make_isometry(chi: int, seed: int = 0) -> np.ndarray:
    """Return a valid (chi, chi, chi) isometry via SVD polar factor."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((chi, chi ** 2))
    U, _, Vt = np.linalg.svd(raw, full_matrices=False)
    return (U @ Vt).reshape(chi, chi, chi)


def _make_unitary(chi: int, seed: int = 0) -> np.ndarray:
    """Return a valid (chi, chi, chi, chi) unitary via SVD polar factor."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((chi ** 2, chi ** 2))
    U, _, Vt = np.linalg.svd(raw, full_matrices=True)
    return (U @ Vt).reshape(chi, chi, chi, chi)


# ──────────────── _apply_isometry_adjoint ─────────────────────────────

class TestApplyIsometryAdjoint:
    """Unit tests for the isometry-adjoint contraction helper."""

    def test_output_ndim_increases_by_one_1d(self):
        chi = 2
        W = _make_isometry(chi)
        state = np.ones(chi) / np.sqrt(chi)
        result = _apply_isometry_adjoint(state, W, axis=0, chi=chi)
        assert result.ndim == 2

    def test_output_shape_1d_state(self):
        chi = 2
        W = _make_isometry(chi)
        state = np.ones(chi) / np.sqrt(chi)
        result = _apply_isometry_adjoint(state, W, axis=0, chi=chi)
        assert result.shape == (chi, chi)

    def test_output_shape_2d_state_axis0(self):
        chi = 2
        W = _make_isometry(chi)
        state = np.eye(chi)
        result = _apply_isometry_adjoint(state, W, axis=0, chi=chi)
        assert result.shape == (chi, chi, chi)

    def test_output_shape_2d_state_axis1(self):
        chi = 2
        W = _make_isometry(chi)
        state = np.eye(chi)
        result = _apply_isometry_adjoint(state, W, axis=1, chi=chi)
        assert result.shape == (chi, chi, chi)

    def test_new_axes_inserted_at_axis_position(self):
        """After applying to axis 1 on a 3D state, new axes appear at positions 1 and 2."""
        chi = 2
        W = _make_isometry(chi)
        state = np.arange(chi ** 3, dtype=float).reshape(chi, chi, chi)
        result = _apply_isometry_adjoint(state, W, axis=1, chi=chi)
        assert result.ndim == 4
        assert result.shape[1] == chi
        assert result.shape[2] == chi

    def test_contraction_matches_einsum_axis0(self):
        """Manual einsum for axis=0 on a 1-D state must match output."""
        chi = 2
        rng = np.random.default_rng(77)
        W = _make_isometry(chi, seed=10)
        state = rng.standard_normal(chi)
        result = _apply_isometry_adjoint(state, W, axis=0, chi=chi)
        # W†[in1, in2, out] = W[out, in1, in2] for real W
        expected = np.einsum("oij,o->ij", W, state)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_contraction_matches_einsum_axis1(self):
        """Axis=1 on a 2-D state: new axes at positions 1 and 2."""
        chi = 2
        rng = np.random.default_rng(88)
        W = _make_isometry(chi, seed=20)
        state = rng.standard_normal((chi, chi))
        result = _apply_isometry_adjoint(state, W, axis=1, chi=chi)
        # W[out, in1, in2] * state[a, out] -> result[a, in1, in2]
        expected = np.einsum("oij,ao->aij", W, state)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_isometry_relation_ww_adjoint_equals_identity(self):
        """W applied after W† must recover the original state (W W† = I)."""
        chi = 2
        W = _make_isometry(chi, seed=5)
        rng = np.random.default_rng(42)
        v = rng.standard_normal(chi)
        expanded = _apply_isometry_adjoint(v, W, axis=0, chi=chi)
        # Apply W forward: contract "out" with (in1, in2) of expanded
        contracted = np.einsum("oij,ij->o", W, expanded)
        np.testing.assert_allclose(contracted, v, atol=1e-12)

    def test_chi3_shape(self):
        chi = 3
        W = _make_isometry(chi, seed=7)
        state = np.ones(chi) / np.sqrt(chi)
        result = _apply_isometry_adjoint(state, W, axis=0, chi=chi)
        assert result.shape == (chi, chi)


# ──────────────── _apply_unitary_adjoint ──────────────────────────────

class TestApplyUnitaryAdjoint:
    """Unit tests for the unitary-adjoint contraction helper."""

    def test_output_ndim_unchanged_2d(self):
        chi = 2
        U = _make_unitary(chi)
        state = np.ones((chi, chi)) / chi
        result = _apply_unitary_adjoint(state, U, 0, 1, chi)
        assert result.ndim == state.ndim

    def test_output_shape_2d(self):
        chi = 2
        U = _make_unitary(chi)
        state = np.eye(chi)
        result = _apply_unitary_adjoint(state, U, 0, 1, chi)
        assert result.shape == (chi, chi)

    def test_output_shape_4d(self):
        chi = 2
        U = _make_unitary(chi)
        state = np.ones((chi, chi, chi, chi))
        result = _apply_unitary_adjoint(state, U, 0, 1, chi)
        assert result.shape == (chi, chi, chi, chi)

    def test_identity_unitary_is_noop(self):
        """Applying the identity unitary leaves the state unchanged."""
        chi = 2
        eye4 = np.eye(chi ** 2).reshape(chi, chi, chi, chi)
        rng = np.random.default_rng(3)
        state = rng.standard_normal((chi, chi))
        result = _apply_unitary_adjoint(state, eye4, 0, 1, chi)
        np.testing.assert_allclose(result, state, atol=1e-12)

    def test_adjoint_then_forward_recovers_original(self):
        """Applying U then U† (or vice-versa) recovers the original state."""
        chi = 2
        U = _make_unitary(chi, seed=9)
        rng = np.random.default_rng(5)
        state = rng.standard_normal((chi, chi))
        after_adj = _apply_unitary_adjoint(state, U, 0, 1, chi)
        # Apply U forward: tensordot over (in1, in2) of U against (site1, site2)
        U_4d = U  # shape (out1, out2, in1, in2)
        recovered = np.tensordot(U_4d, after_adj, axes=([2, 3], [0, 1]))
        np.testing.assert_allclose(recovered, state, atol=1e-12)

    def test_contraction_matches_einsum_2d(self):
        """Result must match explicit einsum on a 2-D state."""
        chi = 2
        U = _make_unitary(chi, seed=99)
        rng = np.random.default_rng(11)
        state = rng.standard_normal((chi, chi))
        result = _apply_unitary_adjoint(state, U, 0, 1, chi)
        # U†[in1, in2, out1, out2] = U[out1, out2, in1, in2]
        # result[in1, in2] = sum_{out1, out2} U[out1, out2, in1, in2] * state[out1, out2]
        expected = np.einsum("opij,op->ij", U, state)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_contraction_matches_einsum_3d(self):
        """Result must match explicit einsum on a 3-D state (sites 0 and 1)."""
        chi = 2
        U = _make_unitary(chi, seed=55)
        rng = np.random.default_rng(22)
        state = rng.standard_normal((chi, chi, chi))
        result = _apply_unitary_adjoint(state, U, 0, 1, chi)
        # result[in1, in2, k] = sum_{out1, out2} U[out1, out2, in1, in2] * state[out1, out2, k]
        expected = np.einsum("opij,opk->ijk", U, state)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_chi3_identity_noop(self):
        chi = 3
        eye4 = np.eye(chi ** 2).reshape(chi, chi, chi, chi)
        rng = np.random.default_rng(30)
        state = rng.standard_normal((chi, chi))
        result = _apply_unitary_adjoint(state, eye4, 0, 1, chi)
        np.testing.assert_allclose(result, state, atol=1e-12)


# ──────────────── MERALayer ───────────────────────────────────────────

class TestMERALayer:
    """Tests for the MERALayer dataclass."""

    def _layer_from_mera(self, n_in: int = 4, chi: int = 2, seed: int = 0) -> MERALayer:
        return random_mera(n_in, chi=chi, seed=seed).layers[0]

    def test_n_out_property_4sites(self):
        layer = self._layer_from_mera(n_in=4)
        assert layer.n_out == 2

    def test_n_out_property_8sites(self):
        layer = self._layer_from_mera(n_in=8, chi=2)
        assert layer.n_out == 4

    def test_n_out_property_2sites(self):
        layer = self._layer_from_mera(n_in=2, chi=2)
        assert layer.n_out == 1

    def test_disentanglers_count(self):
        layer = self._layer_from_mera(n_in=4)
        assert len(layer.disentanglers) == 2

    def test_isometries_count(self):
        layer = self._layer_from_mera(n_in=4)
        assert len(layer.isometries) == 2

    def test_disentangler_shapes_chi2(self):
        chi = 2
        layer = self._layer_from_mera(n_in=4, chi=chi)
        for d in layer.disentanglers:
            assert d.shape == (chi, chi, chi, chi)

    def test_isometry_shapes_chi2(self):
        chi = 2
        layer = self._layer_from_mera(n_in=4, chi=chi)
        for iso in layer.isometries:
            assert iso.shape == (chi, chi, chi)

    def test_disentangler_shapes_chi3(self):
        chi = 3
        layer = self._layer_from_mera(n_in=4, chi=chi)
        for d in layer.disentanglers:
            assert d.shape == (chi, chi, chi, chi)

    def test_isometry_shapes_chi3(self):
        chi = 3
        layer = self._layer_from_mera(n_in=4, chi=chi)
        for iso in layer.isometries:
            assert iso.shape == (chi, chi, chi)

    def test_isometries_satisfy_isometry_condition(self):
        layer = self._layer_from_mera(n_in=4, chi=2)
        for iso in layer.isometries:
            assert isometry_defect(iso) < 1e-10

    def test_disentanglers_satisfy_unitary_condition(self):
        layer = self._layer_from_mera(n_in=4, chi=2)
        for d in layer.disentanglers:
            assert unitary_defect(d) < 1e-10

    def test_enforce_constraints_preserves_shapes(self):
        chi = 2
        layer = self._layer_from_mera(n_in=4, chi=chi)
        rng = np.random.default_rng(42)
        layer.disentanglers = [rng.standard_normal(d.shape) for d in layer.disentanglers]
        layer.isometries = [rng.standard_normal(w.shape) for w in layer.isometries]
        layer.enforce_constraints()
        for d in layer.disentanglers:
            assert d.shape == (chi, chi, chi, chi)
        for iso in layer.isometries:
            assert iso.shape == (chi, chi, chi)

    def test_enforce_constraints_restores_unitarity(self):
        chi = 2
        layer = self._layer_from_mera(n_in=4, chi=chi)
        rng = np.random.default_rng(17)
        layer.disentanglers = [rng.standard_normal(d.shape) for d in layer.disentanglers]
        layer.enforce_constraints()
        for d in layer.disentanglers:
            assert unitary_defect(d) < 1e-10

    def test_enforce_constraints_restores_isometry(self):
        chi = 2
        layer = self._layer_from_mera(n_in=4, chi=chi)
        rng = np.random.default_rng(19)
        layer.isometries = [rng.standard_normal(w.shape) for w in layer.isometries]
        layer.enforce_constraints()
        for iso in layer.isometries:
            assert isometry_defect(iso) < 1e-10

    def test_n_in_attribute_stored(self):
        layer = self._layer_from_mera(n_in=4, chi=2)
        assert layer.n_in == 4

    def test_chi_attribute_stored(self):
        layer = self._layer_from_mera(n_in=4, chi=3)
        assert layer.chi == 3


# ──────────────── MERA.state_vector ──────────────────────────────────

class TestStateVector:
    """Tests for MERA.state_vector."""

    def test_shape_4sites_chi2(self):
        psi = random_mera(4, chi=2, seed=0).state_vector()
        assert psi.shape == (16,)

    def test_shape_2sites_chi2(self):
        psi = random_mera(2, chi=2, seed=0).state_vector()
        assert psi.shape == (4,)

    def test_shape_8sites_chi2(self):
        psi = random_mera(8, chi=2, seed=0).state_vector()
        assert psi.shape == (256,)

    def test_shape_4sites_chi3(self):
        psi = random_mera(4, chi=3, seed=0).state_vector()
        assert psi.shape == (3 ** 4,)

    def test_is_1d(self):
        psi = random_mera(4, chi=2, seed=0).state_vector()
        assert psi.ndim == 1

    def test_is_real(self):
        psi = random_mera(4, chi=2, seed=0).state_vector()
        assert np.isrealobj(psi)

    def test_normalised_4sites(self):
        psi = random_mera(4, chi=2, seed=0).state_vector()
        norm_sq = float(np.dot(psi, psi))
        assert abs(norm_sq - 1.0) < 1e-10, f"||psi||^2 = {norm_sq}"

    def test_normalised_2sites(self):
        psi = random_mera(2, chi=2, seed=7).state_vector()
        norm_sq = float(np.dot(psi, psi))
        assert abs(norm_sq - 1.0) < 1e-10

    def test_normalised_8sites(self):
        psi = random_mera(8, chi=2, seed=3).state_vector()
        norm_sq = float(np.dot(psi, psi))
        assert abs(norm_sq - 1.0) < 1e-10

    def test_normalised_chi3(self):
        psi = random_mera(4, chi=3, seed=0).state_vector()
        norm_sq = float(np.dot(psi, psi))
        assert abs(norm_sq - 1.0) < 1e-10

    def test_deterministic_same_seed(self):
        psi1 = random_mera(4, chi=2, seed=42).state_vector()
        psi2 = random_mera(4, chi=2, seed=42).state_vector()
        np.testing.assert_array_equal(psi1, psi2)

    def test_different_seeds_give_different_vectors(self):
        psi1 = random_mera(4, chi=2, seed=0).state_vector()
        psi2 = random_mera(4, chi=2, seed=1).state_vector()
        assert not np.allclose(psi1, psi2)

    def test_without_disentanglers_still_normalised(self):
        psi = random_mera(4, chi=2, seed=0, with_disentanglers=False).state_vector()
        norm_sq = float(np.dot(psi, psi))
        assert abs(norm_sq - 1.0) < 1e-10

    def test_without_disentanglers_shape(self):
        psi = random_mera(4, chi=2, seed=0, with_disentanglers=False).state_vector()
        assert psi.shape == (16,)


# ──────────────── MERA.enforce_constraints ───────────────────────────

class TestEnforceConstraints:
    """Tests for MERA.enforce_constraints."""

    def test_normalises_top_state(self):
        mera = random_mera(4, chi=2, seed=0)
        mera.top = np.array([3.0, 4.0])  # norm = 5
        mera.enforce_constraints()
        assert abs(float(np.linalg.norm(mera.top)) - 1.0) < 1e-10

    def test_top_zero_vector_handled(self):
        """Near-zero top should not raise (norm guard skips division)."""
        mera = random_mera(4, chi=2, seed=0)
        mera.top = np.array([0.0, 0.0])
        # Should not raise; top is left unchanged by the norm guard
        mera.enforce_constraints()

    def test_restores_layer_unitary(self):
        mera = random_mera(4, chi=2, seed=0)
        rng = np.random.default_rng(55)
        for layer in mera.layers:
            layer.disentanglers = [rng.standard_normal(d.shape) for d in layer.disentanglers]
        mera.enforce_constraints()
        for layer in mera.layers:
            for d in layer.disentanglers:
                assert unitary_defect(d) < 1e-10

    def test_restores_layer_isometry(self):
        mera = random_mera(4, chi=2, seed=0)
        rng = np.random.default_rng(56)
        for layer in mera.layers:
            layer.isometries = [rng.standard_normal(w.shape) for w in layer.isometries]
        mera.enforce_constraints()
        for layer in mera.layers:
            for iso in layer.isometries:
                assert isometry_defect(iso) < 1e-10

    def test_state_vector_still_normalised_after_enforce(self):
        mera = random_mera(4, chi=2, seed=0)
        mera.enforce_constraints()
        psi = mera.state_vector()
        norm_sq = float(np.dot(psi, psi))
        assert abs(norm_sq - 1.0) < 1e-10

    def test_idempotent_on_valid_mera(self):
        """enforce_constraints on an already-valid MERA should leave it unchanged."""
        mera = random_mera(4, chi=2, seed=0)
        psi_before = mera.state_vector()
        mera.enforce_constraints()
        psi_after = mera.state_vector()
        np.testing.assert_allclose(psi_before, psi_after, atol=1e-12)


# ──────────────── to_flat_params / from_flat_params ──────────────────

class TestFlatParams:
    """Tests for parameter serialisation and round-trip."""

    def test_to_flat_params_returns_1d(self):
        mera = random_mera(4, chi=2, seed=0)
        assert mera.to_flat_params().ndim == 1

    def test_to_flat_params_dtype_float(self):
        mera = random_mera(4, chi=2, seed=0)
        params = mera.to_flat_params()
        assert np.issubdtype(params.dtype, np.floating)

    def test_flat_params_size_4sites_chi2(self):
        """
        Layer 0: n_in=4, n_pairs=2 → 2*(2^4) + 2*(2^3) = 32+16 = 48
        Layer 1: n_in=2, n_pairs=1 → 1*(2^4) + 1*(2^3) = 16+8 = 24
        Top: chi=2 → 2
        Total: 74
        """
        mera = random_mera(4, chi=2, seed=0)
        assert mera.to_flat_params().shape == (74,)

    def test_flat_params_size_2sites_chi2(self):
        """
        Layer 0: n_in=2, n_pairs=1 → 1*(2^4) + 1*(2^3) = 24
        Top: 2
        Total: 26
        """
        mera = random_mera(2, chi=2, seed=0)
        assert mera.to_flat_params().shape == (26,)

    def test_round_trip_state_vector_4sites(self):
        mera = random_mera(4, chi=2, seed=0)
        params = mera.to_flat_params()
        mera2 = mera.from_flat_params(params)
        np.testing.assert_allclose(
            mera.state_vector(), mera2.state_vector(), atol=1e-12
        )

    def test_round_trip_state_vector_2sites(self):
        mera = random_mera(2, chi=2, seed=3)
        params = mera.to_flat_params()
        mera2 = mera.from_flat_params(params)
        np.testing.assert_allclose(
            mera.state_vector(), mera2.state_vector(), atol=1e-12
        )

    def test_round_trip_state_vector_chi3(self):
        mera = random_mera(4, chi=3, seed=1)
        params = mera.to_flat_params()
        mera2 = mera.from_flat_params(params)
        np.testing.assert_allclose(
            mera.state_vector(), mera2.state_vector(), atol=1e-12
        )

    def test_round_trip_top(self):
        mera = random_mera(4, chi=2, seed=0)
        params = mera.to_flat_params()
        mera2 = mera.from_flat_params(params)
        np.testing.assert_allclose(mera.top, mera2.top, atol=1e-14)

    def test_round_trip_preserves_n_sites(self):
        mera = random_mera(4, chi=2, seed=0)
        mera2 = mera.from_flat_params(mera.to_flat_params())
        assert mera2.n_sites == mera.n_sites

    def test_round_trip_preserves_chi(self):
        mera = random_mera(4, chi=2, seed=0)
        mera2 = mera.from_flat_params(mera.to_flat_params())
        assert mera2.chi == mera.chi

    def test_round_trip_preserves_layer_count(self):
        mera = random_mera(4, chi=2, seed=0)
        mera2 = mera.from_flat_params(mera.to_flat_params())
        assert len(mera2.layers) == len(mera.layers)

    def test_modified_params_change_state_vector(self):
        mera = random_mera(4, chi=2, seed=0)
        params = mera.to_flat_params()
        params_mod = params.copy()
        params_mod[0] += 1.0
        mera2 = mera.from_flat_params(params_mod)
        assert not np.allclose(mera.state_vector(), mera2.state_vector())

    def test_round_trip_disentangler_shapes(self):
        chi = 2
        mera = random_mera(4, chi=chi, seed=0)
        mera2 = mera.from_flat_params(mera.to_flat_params())
        for layer in mera2.layers:
            for d in layer.disentanglers:
                assert d.shape == (chi, chi, chi, chi)

    def test_round_trip_isometry_shapes(self):
        chi = 2
        mera = random_mera(4, chi=chi, seed=0)
        mera2 = mera.from_flat_params(mera.to_flat_params())
        for layer in mera2.layers:
            for iso in layer.isometries:
                assert iso.shape == (chi, chi, chi)


# ──────────────── random_mera factory ────────────────────────────────

class TestRandomMera:
    """Tests for the random_mera factory function."""

    def test_invalid_n_sites_3_raises(self):
        with pytest.raises(ValueError, match="power of 2"):
            random_mera(3)

    def test_invalid_n_sites_1_raises(self):
        with pytest.raises(ValueError, match="power of 2"):
            random_mera(1)

    def test_invalid_n_sites_0_raises(self):
        with pytest.raises(ValueError):
            random_mera(0)

    def test_invalid_n_sites_6_raises(self):
        with pytest.raises(ValueError, match="power of 2"):
            random_mera(6)

    def test_invalid_n_sites_5_raises(self):
        with pytest.raises(ValueError, match="power of 2"):
            random_mera(5)

    def test_valid_n_sites_2(self):
        mera = random_mera(2, chi=2, seed=0)
        assert mera.n_sites == 2

    def test_valid_n_sites_4(self):
        mera = random_mera(4, chi=2, seed=0)
        assert mera.n_sites == 4

    def test_valid_n_sites_8(self):
        mera = random_mera(8, chi=2, seed=0)
        assert mera.n_sites == 8

    def test_chi_stored_correctly(self):
        mera = random_mera(4, chi=3, seed=0)
        assert mera.chi == 3

    def test_returns_mera_instance(self):
        assert isinstance(random_mera(4, chi=2, seed=0), MERA)

    def test_layers_are_mera_layer_instances(self):
        mera = random_mera(4, chi=2, seed=0)
        for layer in mera.layers:
            assert isinstance(layer, MERALayer)

    def test_layers_count_4sites(self):
        mera = random_mera(4, chi=2, seed=0)
        assert len(mera.layers) == 2

    def test_layers_count_8sites(self):
        mera = random_mera(8, chi=2, seed=0)
        assert len(mera.layers) == 3

    def test_layers_count_2sites(self):
        mera = random_mera(2, chi=2, seed=0)
        assert len(mera.layers) == 1

    def test_top_shape(self):
        chi = 2
        mera = random_mera(4, chi=chi, seed=0)
        assert mera.top.shape == (chi,)

    def test_top_normalised(self):
        mera = random_mera(4, chi=2, seed=0)
        assert abs(float(np.linalg.norm(mera.top)) - 1.0) < 1e-10

    def test_seed_reproducibility_top(self):
        m1 = random_mera(4, chi=2, seed=99)
        m2 = random_mera(4, chi=2, seed=99)
        np.testing.assert_array_equal(m1.top, m2.top)

    def test_seed_reproducibility_state_vector(self):
        psi1 = random_mera(4, chi=2, seed=99).state_vector()
        psi2 = random_mera(4, chi=2, seed=99).state_vector()
        np.testing.assert_array_equal(psi1, psi2)

    def test_different_seeds_different_top(self):
        m0 = random_mera(4, chi=2, seed=0)
        m1 = random_mera(4, chi=2, seed=1)
        assert not np.allclose(m0.top, m1.top)

    def test_without_disentanglers_all_identity(self):
        chi = 2
        mera = random_mera(4, chi=chi, seed=0, with_disentanglers=False)
        eye4 = np.eye(chi ** 2).reshape(chi, chi, chi, chi)
        for layer in mera.layers:
            for d in layer.disentanglers:
                np.testing.assert_allclose(d, eye4, atol=1e-14)

    def test_with_disentanglers_all_unitary(self):
        mera = random_mera(4, chi=2, seed=0, with_disentanglers=True)
        for layer in mera.layers:
            for d in layer.disentanglers:
                assert unitary_defect(d) < 1e-10

    def test_all_isometries_valid_chi2(self):
        mera = random_mera(4, chi=2, seed=0)
        for layer in mera.layers:
            for iso in layer.isometries:
                assert isometry_defect(iso) < 1e-10

    def test_all_isometries_valid_chi3(self):
        mera = random_mera(4, chi=3, seed=2)
        for layer in mera.layers:
            for iso in layer.isometries:
                assert isometry_defect(iso) < 1e-10

    def test_layer_n_in_ordering(self):
        """layers[0].n_in == n_sites; each subsequent layer halves n_in."""
        mera = random_mera(8, chi=2, seed=0)
        assert mera.layers[0].n_in == 8
        assert mera.layers[1].n_in == 4
        assert mera.layers[2].n_in == 2

    def test_default_chi_is_2(self):
        mera = random_mera(4, seed=0)
        assert mera.chi == 2

    def test_default_seed_is_0(self):
        m1 = random_mera(4)
        m2 = random_mera(4, seed=0)
        np.testing.assert_array_equal(m1.top, m2.top)

    def test_default_with_disentanglers_true(self):
        chi = 2
        mera = random_mera(4)
        eye4 = np.eye(chi ** 2).reshape(chi, chi, chi, chi)
        for layer in mera.layers:
            for d in layer.disentanglers:
                # Default includes disentanglers, so at least one should differ from eye4
                # (probabilistically guaranteed; we just check not all identity)
                pass
        # Verify the disentanglers are not all identity (random orthogonal)
        any_nonidentity = any(
            not np.allclose(d, eye4)
            for layer in mera.layers
            for d in layer.disentanglers
        )
        assert any_nonidentity
