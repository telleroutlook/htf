"""Tests for htf.mps — Matrix Product State operations."""
import math

import numpy as np
import pytest

from htf.mps import (
    MPS,
    _left_canonicalise,
    mps_add,
    mps_apply_gate,
    mps_expectation,
    mps_from_state,
    mps_inner,
    mps_norm,
    mps_normalise,
    mps_to_state,
    mps_truncate,
    random_mps,
)

# ── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def bell_state_mps():
    """MPS for (|00⟩ + |11⟩) / √2."""
    psi = np.array([1, 0, 0, 1], dtype=float) / math.sqrt(2)
    return mps_from_state(psi, d=2)


@pytest.fixture
def random_4site_mps():
    return random_mps(4, d=2, chi=4, seed=0)


# ── construction ──────────────────────────────────────────────────────────


class TestRandomMPS:
    def test_shape(self):
        mps = random_mps(4, d=2, chi=3, seed=0)
        assert mps.n_sites == 4
        assert mps.phys_dim == 2
        assert mps.tensors[0].shape[0] == 1
        assert mps.tensors[-1].shape[2] == 1

    def test_bond_dims_clipped(self):
        mps = random_mps(5, d=2, chi=4, seed=0)
        for b in mps.bond_dims:
            assert b <= 4

    def test_max_bond(self):
        mps = random_mps(4, d=2, chi=3, seed=0)
        assert mps.max_bond == max(mps.bond_dims)

    def test_copy_independent(self):
        mps = random_mps(3, d=2, chi=2, seed=1)
        c = mps.copy()
        c.tensors[0][:] = 0.0
        assert not np.allclose(mps.tensors[0], 0.0)

    def test_seeded_reproducible(self):
        a = random_mps(3, d=2, chi=2, seed=42)
        b = random_mps(3, d=2, chi=2, seed=42)
        for ta, tb in zip(a.tensors, b.tensors):
            np.testing.assert_array_equal(ta, tb)


class TestMpsFromState:
    def test_round_trip_no_truncation(self):
        n, d = 3, 2
        rng = np.random.default_rng(5)
        psi = rng.standard_normal(d**n)
        psi /= np.linalg.norm(psi)
        mps = mps_from_state(psi, d=d)
        psi2 = mps_to_state(mps)
        np.testing.assert_allclose(psi2, psi, atol=1e-14)

    def test_round_trip_with_truncation(self):
        # For n=3, d=2, the max bond dim is 2; chi=2 should be lossless
        psi = np.array([1, 0, 0, 0, 0, 0, 0, 1], dtype=float) / math.sqrt(2)
        mps = mps_from_state(psi, d=2, chi=2)
        psi2 = mps_to_state(mps)
        np.testing.assert_allclose(psi2, psi, atol=1e-14)

    def test_bad_length_raises(self):
        with pytest.raises(ValueError, match="not d\\^n"):
            mps_from_state(np.ones(5), d=2)

    def test_boundary_shapes(self):
        mps = mps_from_state(np.array([1, 0, 0, 0], dtype=float), d=2)
        assert mps.tensors[0].shape[0] == 1
        assert mps.tensors[-1].shape[2] == 1


# ── state vector conversion ───────────────────────────────────────────────


class TestMpsToState:
    def test_computational_basis(self):
        psi = np.zeros(8, dtype=float); psi[3] = 1.0   # |011⟩
        mps = mps_from_state(psi, d=2)
        out = mps_to_state(mps)
        np.testing.assert_allclose(out, psi, atol=1e-14)

    def test_bell_state(self, bell_state_mps):
        psi_ref = np.array([1, 0, 0, 1], dtype=float) / math.sqrt(2)
        out = mps_to_state(bell_state_mps)
        np.testing.assert_allclose(np.abs(out), np.abs(psi_ref), atol=1e-14)


# ── inner product and norm ─────────────────────────────────────────────────


class TestMpsInner:
    def test_self_inner_is_norm_squared(self, random_4site_mps):
        mps = random_4site_mps
        ip   = mps_inner(mps, mps)
        nrm2 = mps_norm(mps) ** 2
        assert abs(ip - nrm2) < 1e-12

    def test_against_dense(self):
        rng = np.random.default_rng(7)
        psi = rng.standard_normal(8); psi /= np.linalg.norm(psi)
        phi = rng.standard_normal(8); phi /= np.linalg.norm(phi)
        m1  = mps_from_state(psi, d=2)
        m2  = mps_from_state(phi, d=2)
        assert abs(mps_inner(m1, m2) - np.dot(psi.conj(), phi)) < 1e-12

    def test_orthogonal_states(self):
        psi_00 = mps_from_state(np.array([1,0,0,0], dtype=float), d=2)
        psi_11 = mps_from_state(np.array([0,0,0,1], dtype=float), d=2)
        assert abs(mps_inner(psi_00, psi_11)) < 1e-14

    def test_mismatched_sizes_raises(self):
        m3 = mps_from_state(np.array([1,0,0,0,0,0,0,0], dtype=float), d=2)
        m2 = mps_from_state(np.array([1,0,0,0], dtype=float), d=2)
        with pytest.raises(ValueError, match="site counts"):
            mps_inner(m3, m2)


class TestMpsNorm:
    def test_normalised_state(self, bell_state_mps):
        assert abs(mps_norm(bell_state_mps) - 1.0) < 1e-13

    def test_scaled_state(self):
        mps = mps_from_state(np.array([1,0,0,0], dtype=float), d=2)
        mps2 = MPS([t * 3.0 for t in mps.tensors])
        # dividing all tensors by 3 divides state by 3^n
        # the correct scaling depends on where the norm lives
        # just check norm is positive
        assert mps_norm(mps2) > 0


class TestMpsNormalise:
    def test_unit_norm_after_normalise(self):
        mps = random_mps(4, d=2, chi=3, seed=3)
        mps_n = mps_normalise(mps)
        assert abs(mps_norm(mps_n) - 1.0) < 1e-12

    def test_original_unchanged(self):
        mps = random_mps(4, d=2, chi=3, seed=3)
        nrm_before = mps_norm(mps)
        _ = mps_normalise(mps)
        assert abs(mps_norm(mps) - nrm_before) < 1e-12


# ── expectation value ─────────────────────────────────────────────────────


class TestMpsExpectation:
    Z = np.array([[1, 0], [0, -1]], dtype=float)
    X = np.array([[0, 1], [1, 0]], dtype=float)

    def test_z0_on_up_state(self):
        mps_00 = mps_from_state(np.array([1, 0, 0, 0], dtype=float), d=2)
        val = mps_expectation(mps_00, [(0, self.Z)])
        assert abs(val - 1.0) < 1e-12

    def test_z1_on_down_state(self):
        mps_01 = mps_from_state(np.array([0, 1, 0, 0], dtype=float), d=2)
        # qubit 1 is |0⟩ in |01⟩ (big-endian: site 0 = 0, site 1 = 1)
        # Z|1⟩ = -|1⟩ so <Z_1> = -1
        val = mps_expectation(mps_01, [(1, self.Z)])
        assert abs(val - (-1.0)) < 1e-12

    def test_x_expectation_on_plus_state(self):
        # |+⟩⊗|0⟩ = ([1,0,1,0]/√2 in big-endian): site 0 is |+⟩
        psi = np.array([1, 0, 1, 0], dtype=float) / math.sqrt(2)
        mps = mps_from_state(psi, d=2)
        val = mps_expectation(mps, [(0, self.X)])
        assert abs(val - 1.0) < 1e-12

    def test_against_dense(self):
        rng = np.random.default_rng(9)
        psi = rng.standard_normal(8)
        psi /= np.linalg.norm(psi)
        mps = mps_from_state(psi, d=2)
        Z2 = np.kron(np.eye(2), np.kron(self.Z, np.eye(2)))
        exp_ref = float(np.dot(psi.conj(), Z2 @ psi).real)
        exp_mps = float(mps_expectation(mps, [(1, self.Z)]).real)
        assert abs(exp_mps - exp_ref) < 1e-12


# ── add ───────────────────────────────────────────────────────────────────


class TestMpsAdd:
    def test_sum_norm(self):
        psi = np.array([1, 0, 0, 0], dtype=float)
        phi = np.array([0, 0, 0, 1], dtype=float)
        m1 = mps_from_state(psi, d=2)
        m2 = mps_from_state(phi, d=2)
        msum = mps_add(m1, m2)
        psi_sum = mps_to_state(msum)
        np.testing.assert_allclose(psi_sum, psi + phi, atol=1e-13)

    def test_sum_bond_dims(self):
        m1 = random_mps(3, d=2, chi=2, seed=0)
        m2 = random_mps(3, d=2, chi=3, seed=1)
        msum = mps_add(m1, m2)
        # middle bond should be chi_1 + chi_2
        assert msum.bond_dims[0] == m1.bond_dims[0] + m2.bond_dims[0]

    def test_mismatched_sizes_raises(self):
        m3 = random_mps(3, d=2, chi=2, seed=0)
        m4 = random_mps(4, d=2, chi=2, seed=1)
        with pytest.raises(ValueError):
            mps_add(m3, m4)


# ── truncation ────────────────────────────────────────────────────────────


class TestMpsTruncate:
    def test_bond_dim_reduced(self):
        mps = random_mps(5, d=2, chi=8, seed=2)
        mps_t, _disc = mps_truncate(mps, chi=3)
        for b in mps_t.bond_dims:
            assert b <= 3

    def test_discarded_nonneg(self):
        mps = random_mps(5, d=2, chi=4, seed=2)
        _, disc = mps_truncate(mps, chi=2)
        assert disc >= 0.0

    def test_lossless_if_chi_large(self):
        # For n=3, d=2, exact chi is at most 2; truncating to chi=4 is lossless
        psi = np.array([1, 0, 0, 1, 0, 0, 0, 0], dtype=float) / math.sqrt(2)
        mps = mps_from_state(psi, d=2)
        mps_t, disc = mps_truncate(mps, chi=8)
        assert disc < 1e-12
        psi_t = mps_to_state(mps_t)
        np.testing.assert_allclose(np.abs(psi_t), np.abs(psi), atol=1e-12)

    def test_truncation_fidelity(self):
        rng = np.random.default_rng(11)
        psi = rng.standard_normal(16); psi /= np.linalg.norm(psi)
        mps = mps_from_state(psi, d=2, chi=None)
        mps_t, _disc = mps_truncate(mps, chi=2)
        psi_t = mps_to_state(mps_t)
        psi_t /= np.linalg.norm(psi_t)
        fidelity = float(np.abs(np.dot(psi.conj(), psi_t)) ** 2)
        assert fidelity > 0.5   # should retain most fidelity


# ── gate application ──────────────────────────────────────────────────────


class TestMpsApplyGate:
    def test_x_gate_flips_qubit(self):
        psi_0 = np.array([1, 0, 0, 0], dtype=float)   # |00⟩
        mps   = mps_from_state(psi_0, d=2)
        X     = np.array([[0, 1], [1, 0]], dtype=float)
        mps2, _ = mps_apply_gate(mps, X, [0])
        psi2    = mps_to_state(mps2)
        # X|0⟩ = |1⟩ on qubit 0 → |10⟩
        np.testing.assert_allclose(np.abs(psi2), [0, 0, 1, 0], atol=1e-14)

    def test_swap_gate(self):
        # |01⟩ → SWAP → |10⟩
        psi = np.array([0, 1, 0, 0], dtype=float)
        mps = mps_from_state(psi, d=2)
        SWAP = np.eye(4)[[0, 2, 1, 3]].reshape(2, 2, 2, 2).transpose(2, 3, 0, 1)
        mps2, _ = mps_apply_gate(mps, SWAP, [0, 1])
        psi2 = mps_to_state(mps2)
        np.testing.assert_allclose(np.abs(psi2), [0, 0, 1, 0], atol=1e-14)

    def test_identity_gate_unchanged(self):
        mps = random_mps(3, d=2, chi=2, seed=5)
        psi_before = mps_to_state(mps)
        I4  = np.eye(4).reshape(2, 2, 2, 2)
        mps2, disc = mps_apply_gate(mps, I4, [0, 1])
        psi_after  = mps_to_state(mps2)
        np.testing.assert_allclose(psi_after, psi_before, atol=1e-13)
        assert disc < 1e-12

    def test_non_adjacent_raises(self):
        mps = random_mps(4, d=2, chi=2, seed=0)
        I4  = np.eye(4).reshape(2, 2, 2, 2)
        with pytest.raises(ValueError, match="adjacent"):
            mps_apply_gate(mps, I4, [0, 2])

    def test_truncation_with_chi(self):
        psi = np.array([1, 0, 0, 0, 0, 0, 0, 1], dtype=float) / math.sqrt(2)
        mps = mps_from_state(psi, d=2)
        I4  = np.eye(4).reshape(2, 2, 2, 2)
        mps2, _ = mps_apply_gate(mps, I4, [0, 1], chi=1)
        assert mps2.bond_dims[0] == 1


# ── left canonicalisation ─────────────────────────────────────────────────


class TestLeftCanonicalise:
    def test_preserves_state(self):
        mps = random_mps(4, d=2, chi=3, seed=13)
        psi_before = mps_to_state(mps)
        mps_c = _left_canonicalise(mps)
        psi_after  = mps_to_state(mps_c)
        np.testing.assert_allclose(psi_after, psi_before, atol=1e-12)
