"""Tests for htf.mpo — Matrix Product Operator data structure."""
import numpy as np
import pytest

from htf.mpo import (
    MPO,
    identity_mpo,
    mpo_apply_mps,
    mpo_expectation,
    mpo_from_matrix,
    mpo_hermitian_conjugate,
    mpo_to_matrix,
    nn_hamiltonian_mpo,
    random_mpo,
)
from htf.mps import MPS, mps_from_state, mps_inner, mps_norm
from htf.tebd import dmrg_sweep, heisenberg_bonds, nn_hamiltonian, tfim_bonds, xx_bonds


# ── helpers ────────────────────────────────────────────────────────────────


def _exact_ground_state(H: np.ndarray):
    vals, vecs = np.linalg.eigh(H)
    return vals[0], vecs[:, 0]


# ── MPO dataclass ──────────────────────────────────────────────────────────


class TestMPODataclass:
    def test_n_sites(self):
        mpo = identity_mpo(4, d=2)
        assert mpo.n_sites == 4

    def test_phys_dim(self):
        mpo = identity_mpo(3, d=3)
        assert mpo.phys_dim == 3

    def test_boundary_bond_dims(self):
        mpo = identity_mpo(4, d=2)
        assert mpo.tensors[0].shape[0] == 1
        assert mpo.tensors[-1].shape[3] == 1

    def test_copy_is_independent(self):
        mpo = random_mpo(3, d=2, chi=2, seed=0)
        mpo2 = mpo.copy()
        mpo2.tensors[0][:] = 0
        assert not np.allclose(mpo.tensors[0], 0)


# ── identity_mpo ───────────────────────────────────────────────────────────


class TestIdentityMpo:
    def test_to_matrix_is_identity(self):
        n, d = 3, 2
        I_mpo = identity_mpo(n, d)
        M = mpo_to_matrix(I_mpo)
        np.testing.assert_allclose(M, np.eye(d ** n), atol=1e-12)

    def test_apply_to_mps_preserves_state(self):
        n, d = 4, 2
        from htf.mps import mps_from_state
        psi = np.random.default_rng(1).standard_normal(d ** n)
        psi /= np.linalg.norm(psi)
        mps = mps_from_state(psi, d)
        I_mpo = identity_mpo(n, d)
        mps2 = mpo_apply_mps(I_mpo, mps)
        inner = mps_inner(mps, mps2)
        assert abs(inner - 1.0) < 1e-10

    def test_expectation_equals_norm_squared(self):
        n, d = 3, 2
        rng = np.random.default_rng(5)
        psi = rng.standard_normal(d ** n)
        psi /= np.linalg.norm(psi)
        mps = mps_from_state(psi, d)
        I_mpo = identity_mpo(n, d)
        exp_val = mpo_expectation(I_mpo, mps)
        assert abs(exp_val - 1.0) < 1e-10


# ── mpo_from_matrix ────────────────────────────────────────────────────────


class TestMpoFromMatrix:
    @pytest.fixture
    def tfim_data(self):
        n, d = 4, 2
        bonds = tfim_bonds(n, J=1.0, h=0.5)
        H = nn_hamiltonian(bonds, n, d)
        return H, n, d

    def test_round_trip_exact(self, tfim_data):
        H, n, d = tfim_data
        mpo = mpo_from_matrix(H, n, d)
        H_rec = mpo_to_matrix(mpo)
        np.testing.assert_allclose(H_rec, H, atol=1e-10)

    def test_chi_limits_bond_dim(self, tfim_data):
        H, n, d = tfim_data
        chi = 4
        mpo = mpo_from_matrix(H, n, d, chi=chi)
        for W in mpo.tensors:
            assert W.shape[0] <= chi or W.shape[0] == 1
            assert W.shape[3] <= chi or W.shape[3] == 1

    def test_identity_round_trip(self):
        n, d = 3, 2
        I = np.eye(d ** n)
        mpo = mpo_from_matrix(I, n, d)
        H_rec = mpo_to_matrix(mpo)
        np.testing.assert_allclose(H_rec, I, atol=1e-10)

    def test_complex_matrix(self):
        n, d = 3, 2
        bonds = heisenberg_bonds(n, J=1.0)
        H = nn_hamiltonian(bonds, n, d)
        assert np.iscomplexobj(H)
        mpo = mpo_from_matrix(H, n, d)
        H_rec = mpo_to_matrix(mpo)
        np.testing.assert_allclose(H_rec, H, atol=1e-10)


# ── nn_hamiltonian_mpo ─────────────────────────────────────────────────────


class TestNnHamiltonianMpo:
    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_matches_full_matrix_tfim(self, n):
        d = 2
        bonds = tfim_bonds(n, J=1.0, h=0.5)
        H_ref = nn_hamiltonian(bonds, n, d)
        mpo = nn_hamiltonian_mpo(bonds, n, d)
        H_rec = mpo_to_matrix(mpo)
        np.testing.assert_allclose(H_rec, H_ref, atol=1e-10)

    def test_matches_full_matrix_xx(self):
        n, d = 4, 2
        bonds = xx_bonds(n)
        H_ref = nn_hamiltonian(bonds, n, d)
        mpo = nn_hamiltonian_mpo(bonds, n, d)
        H_rec = mpo_to_matrix(mpo)
        np.testing.assert_allclose(H_rec, H_ref, atol=1e-10)

    def test_matches_full_matrix_heisenberg(self):
        n, d = 4, 2
        bonds = heisenberg_bonds(n, J=1.0)
        H_ref = nn_hamiltonian(bonds, n, d)
        mpo = nn_hamiltonian_mpo(bonds, n, d)
        H_rec = mpo_to_matrix(mpo)
        np.testing.assert_allclose(H_rec, H_ref, atol=1e-10)

    def test_wrong_term_count_raises(self):
        with pytest.raises(ValueError):
            nn_hamiltonian_mpo(tfim_bonds(4), 5)

    def test_bond_dim_efficient(self):
        n, d = 6, 2
        bonds = tfim_bonds(n, J=1.0, h=0.5)
        mpo = nn_hamiltonian_mpo(bonds, n, d)
        # Bond dim is 2 + rank(h_i); should be far below d^{2n} = 4096
        for W in mpo.tensors[1:-1]:
            assert W.shape[0] <= d ** 2 + 2 and W.shape[3] <= d ** 2 + 2


# ── mpo_apply_mps ──────────────────────────────────────────────────────────


class TestMpoApplyMps:
    def test_energy_eigenvalue(self):
        n, d = 4, 2
        bonds = tfim_bonds(n, J=1.0, h=0.5)
        H = nn_hamiltonian(bonds, n, d)
        E0, v0 = _exact_ground_state(H)
        mps = mps_from_state(v0, d)
        mpo = mpo_from_matrix(H, n, d)
        H_psi = mpo_apply_mps(mpo, mps)
        # ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩ = E0
        numer = mps_inner(mps, H_psi)
        denom = mps_inner(mps, mps)
        assert abs((numer / denom).real - E0) < 1e-8

    def test_output_shape_consistent(self):
        n, d = 3, 2
        mps = mps_from_state(np.ones(d ** n) / np.sqrt(d ** n), d)
        mpo = identity_mpo(n, d)
        out = mpo_apply_mps(mpo, mps)
        assert out.n_sites == n
        assert out.phys_dim == d


# ── mpo_expectation ────────────────────────────────────────────────────────


class TestMpoExpectation:
    def test_ground_state_energy(self):
        n, d = 4, 2
        bonds = tfim_bonds(n, J=1.0, h=0.5)
        H = nn_hamiltonian(bonds, n, d)
        E0, v0 = _exact_ground_state(H)
        mps = mps_from_state(v0, d)
        mpo = mpo_from_matrix(H, n, d)
        exp_val = mpo_expectation(mpo, mps)
        assert abs(exp_val.real - E0) < 1e-8

    def test_nn_mpo_energy_matches_matrix(self):
        n, d = 4, 2
        bonds = xx_bonds(n)
        H = nn_hamiltonian(bonds, n, d)
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(d ** n)
        psi /= np.linalg.norm(psi)
        mps = mps_from_state(psi, d)
        mpo_nn = nn_hamiltonian_mpo(bonds, n, d)
        mpo_full = mpo_from_matrix(H, n, d)
        e_nn = mpo_expectation(mpo_nn, mps).real
        e_full = mpo_expectation(mpo_full, mps).real
        assert abs(e_nn - e_full) < 1e-8


# ── mpo_hermitian_conjugate ────────────────────────────────────────────────


class TestMpoHermitianConjugate:
    def test_hermitian_hamiltonian_unchanged(self):
        n, d = 4, 2
        bonds = tfim_bonds(n, J=1.0, h=0.5)
        H = nn_hamiltonian(bonds, n, d)
        mpo = mpo_from_matrix(H, n, d)
        mpo_dag = mpo_hermitian_conjugate(mpo)
        H_dag = mpo_to_matrix(mpo_dag)
        np.testing.assert_allclose(H_dag, H, atol=1e-10)

    def test_double_conjugate_is_original(self):
        mpo = random_mpo(3, d=2, chi=3, seed=7)
        mpo_dagdag = mpo_hermitian_conjugate(mpo_hermitian_conjugate(mpo))
        for W, W2 in zip(mpo.tensors, mpo_dagdag.tensors):
            np.testing.assert_allclose(W, W2, atol=1e-14)

    def test_bra_ket_swap(self):
        mpo = random_mpo(2, d=2, chi=2, seed=3)
        mpo_dag = mpo_hermitian_conjugate(mpo)
        for W, W_dag in zip(mpo.tensors, mpo_dag.tensors):
            # dag swaps axes 1 and 2, conjugates
            np.testing.assert_allclose(W_dag, W.conj().transpose(0, 2, 1, 3), atol=1e-14)
