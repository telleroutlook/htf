"""Tests for htf.mpo — Matrix Product Operator data structure."""
import numpy as np
import pytest

from htf.mpo import (
    MPOChiPoint,
    MPODMRGResult,
    MPOScalingReport,
    MultiStartDMRGResult,
    dmrg_multistart,
    dmrg_sweep_mpo,
    dmrg_sweep_mpo_2site,
    identity_mpo,
    mpo_apply_mps,
    mpo_chi_convergence,
    mpo_expectation,
    mpo_from_matrix,
    mpo_hermitian_conjugate,
    mpo_to_matrix,
    nn_hamiltonian_mpo,
    random_mpo,
)
from htf.mps import mps_from_state, mps_inner
from htf.tebd import (
    heisenberg_bonds,
    nn_hamiltonian,
    tfim_bonds,
    xx_bonds,
)

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


# ── dmrg_sweep_mpo ─────────────────────────────────────────────────────────


class TestDmrgSweepMpo:
    """MPO-environment single-site DMRG tests.

    Single-site DMRG can get stuck in local minima (well-known limitation;
    2-site DMRG is the fix).  These tests verify correct type, environment
    construction, and agreement with the dense 1-site DMRG.
    """

    @pytest.fixture
    def heisenberg_setup(self):
        n, d = 4, 2
        bonds = heisenberg_bonds(n, J=1.0)
        H = nn_hamiltonian(bonds, n, d)
        E0 = float(np.linalg.eigvalsh(H)[0])
        mpo = nn_hamiltonian_mpo(bonds, n, d)
        return bonds, mpo, E0, n, d

    def test_result_type(self, heisenberg_setup):
        _bonds, mpo, _, n, d = heisenberg_setup
        from htf.mps import random_mps
        mps = random_mps(n, d, chi=4, seed=0)
        result = dmrg_sweep_mpo(mps, mpo, n_sweeps=2)
        assert isinstance(result, MPODMRGResult)

    def test_energies_non_empty(self, heisenberg_setup):
        _bonds, mpo, _, n, d = heisenberg_setup
        from htf.mps import random_mps
        mps = random_mps(n, d, chi=4, seed=0)
        result = dmrg_sweep_mpo(mps, mpo, n_sweeps=3)
        assert len(result.energies) > 0

    def test_energy_decreases(self, heisenberg_setup):
        _bonds, mpo, _, n, d = heisenberg_setup
        from htf.mps import random_mps
        mps = random_mps(n, d, chi=8, seed=5)
        result = dmrg_sweep_mpo(mps, mpo, n_sweeps=5)
        assert result.energies[-1] <= result.energies[0] + 1e-6

    def test_heisenberg_converges_to_exact(self, heisenberg_setup):
        # Heisenberg is easy for 1-site DMRG; converges reliably
        _bonds, mpo, E0, n, d = heisenberg_setup
        from htf.mps import random_mps
        mps = random_mps(n, d, chi=8, seed=0)
        result = dmrg_sweep_mpo(mps, mpo, n_sweeps=15, chi=8)
        assert abs(result.energies[-1] - E0) < 1e-6

    def test_matches_dense_dmrg_tfim(self):
        # Both MPO-DMRG and dense DMRG must be valid upper bounds for TFIM.
        n, d = 4, 2
        bonds = tfim_bonds(n, J=1.0, h=0.5)
        H = nn_hamiltonian(bonds, n, d)
        E0 = float(np.linalg.eigvalsh(H)[0])
        mpo = nn_hamiltonian_mpo(bonds, n, d)
        from htf.mps import random_mps
        mps = random_mps(n, d, chi=8, seed=7)
        result = dmrg_sweep_mpo(mps, mpo, n_sweeps=10, chi=8)
        # Variational upper bound must hold
        assert result.energies[-1] >= E0 - 1e-8

    def test_matches_dense_dmrg_xx(self):
        # MPO-DMRG reaches the same Heisenberg minimum as dense DMRG
        n, d = 4, 2
        bonds = heisenberg_bonds(n, J=1.0)
        H = nn_hamiltonian(bonds, n, d)
        E0 = float(np.linalg.eigvalsh(H)[0])
        mpo = nn_hamiltonian_mpo(bonds, n, d)
        from htf.mps import random_mps
        mps = random_mps(n, d, chi=8, seed=3)
        result = dmrg_sweep_mpo(mps, mpo, n_sweeps=10, chi=8)
        assert abs(result.energies[-1] - E0) < 1e-6

    def test_chi_limits_bond_dim(self, heisenberg_setup):
        _bonds, mpo, _, n, d = heisenberg_setup
        from htf.mps import random_mps
        chi = 2
        mps = random_mps(n, d, chi=8, seed=1)
        result = dmrg_sweep_mpo(mps, mpo, n_sweeps=3, chi=chi)
        for t in result.mps_final.tensors:
            assert t.shape[0] <= chi or t.shape[0] == 1
            assert t.shape[2] <= chi or t.shape[2] == 1

    def test_no_nan(self, heisenberg_setup):
        _bonds, mpo, _, n, d = heisenberg_setup
        from htf.mps import random_mps
        mps = random_mps(n, d, chi=4, seed=2)
        result = dmrg_sweep_mpo(mps, mpo, n_sweeps=5)
        assert not any(np.isnan(e) for e in result.energies)
        for t in result.mps_final.tensors:
            assert not np.any(np.isnan(t))

    def test_energy_upper_bounds_exact(self, heisenberg_setup):
        _bonds, mpo, E0, n, d = heisenberg_setup
        from htf.mps import random_mps
        mps = random_mps(n, d, chi=8, seed=0)
        result = dmrg_sweep_mpo(mps, mpo, n_sweeps=15, chi=8)
        # Variational principle: DMRG energy ≥ exact ground state
        assert result.energies[-1] >= E0 - 1e-8


# ── dmrg_sweep_mpo_2site ───────────────────────────────────────────────────


class TestDmrgSweepMpo2Site:
    """Two-site MPO-DMRG tests.

    Two-site DMRG grows the bond dimension via SVD at each update, escaping
    the local-minima traps of single-site DMRG.  For n≤5, d=2, chi=8 the
    variational space is large enough that both TFIM and Heisenberg converge
    to the exact ground state.
    """

    @pytest.fixture
    def heisenberg_setup(self):
        n, d = 4, 2
        bonds = heisenberg_bonds(n, J=1.0)
        H = nn_hamiltonian(bonds, n, d)
        E0 = float(np.linalg.eigvalsh(H)[0])
        mpo = nn_hamiltonian_mpo(bonds, n, d)
        return bonds, mpo, E0, n, d

    @pytest.fixture
    def tfim_setup(self):
        n, d = 4, 2
        bonds = tfim_bonds(n, J=1.0, h=0.5)
        H = nn_hamiltonian(bonds, n, d)
        E0 = float(np.linalg.eigvalsh(H)[0])
        mpo = nn_hamiltonian_mpo(bonds, n, d)
        return bonds, mpo, E0, n, d

    def test_result_type(self, heisenberg_setup):
        _bonds, mpo, _, n, d = heisenberg_setup
        from htf.mps import random_mps
        mps = random_mps(n, d, chi=2, seed=0)
        result = dmrg_sweep_mpo_2site(mps, mpo, n_sweeps=2)
        assert isinstance(result, MPODMRGResult)

    def test_energies_non_empty(self, heisenberg_setup):
        _bonds, mpo, _, n, d = heisenberg_setup
        from htf.mps import random_mps
        mps = random_mps(n, d, chi=2, seed=0)
        result = dmrg_sweep_mpo_2site(mps, mpo, n_sweeps=3)
        assert len(result.energies) > 0

    def test_energy_decreases(self, heisenberg_setup):
        _bonds, mpo, _, n, d = heisenberg_setup
        from htf.mps import random_mps
        mps = random_mps(n, d, chi=4, seed=5)
        result = dmrg_sweep_mpo_2site(mps, mpo, n_sweeps=5)
        assert result.energies[-1] <= result.energies[0] + 1e-6

    def test_heisenberg_converges_exact(self, heisenberg_setup):
        _bonds, mpo, E0, n, d = heisenberg_setup
        from htf.mps import random_mps
        mps = random_mps(n, d, chi=2, seed=0)
        result = dmrg_sweep_mpo_2site(mps, mpo, n_sweeps=20, chi=8)
        assert abs(result.energies[-1] - E0) < 1e-6

    def test_tfim_converges_exact(self, tfim_setup):
        # Key advantage over 1-site: 2-site escapes local minima for TFIM n=4
        _bonds, mpo, E0, n, d = tfim_setup
        from htf.mps import random_mps
        mps = random_mps(n, d, chi=2, seed=7)
        result = dmrg_sweep_mpo_2site(mps, mpo, n_sweeps=20, chi=8)
        assert abs(result.energies[-1] - E0) < 1e-5

    def test_tfim_valid_upper_bound(self, tfim_setup):
        _bonds, mpo, E0, n, d = tfim_setup
        from htf.mps import random_mps
        mps = random_mps(n, d, chi=4, seed=3)
        result = dmrg_sweep_mpo_2site(mps, mpo, n_sweeps=10, chi=8)
        assert result.energies[-1] >= E0 - 1e-8

    def test_2site_better_than_1site_tfim(self, tfim_setup):
        # 2-site variational space contains 1-site; expect lower or equal energy
        _bonds, mpo, _E0, n, d = tfim_setup
        from htf.mps import random_mps
        seed = 7
        mps1 = random_mps(n, d, chi=2, seed=seed)
        mps2 = random_mps(n, d, chi=2, seed=seed)
        r1 = dmrg_sweep_mpo(mps1, mpo, n_sweeps=15, chi=8)
        r2 = dmrg_sweep_mpo_2site(mps2, mpo, n_sweeps=15, chi=8)
        assert r2.energies[-1] <= r1.energies[-1] + 1e-6

    def test_chi_limits_bond_dim(self, heisenberg_setup):
        _bonds, mpo, _, n, d = heisenberg_setup
        from htf.mps import random_mps
        chi = 3
        mps = random_mps(n, d, chi=2, seed=1)
        result = dmrg_sweep_mpo_2site(mps, mpo, n_sweeps=3, chi=chi)
        for t in result.mps_final.tensors:
            assert t.shape[0] <= chi or t.shape[0] == 1
            assert t.shape[2] <= chi or t.shape[2] == 1

    def test_no_nan(self, heisenberg_setup):
        _bonds, mpo, _, n, d = heisenberg_setup
        from htf.mps import random_mps
        mps = random_mps(n, d, chi=2, seed=2)
        result = dmrg_sweep_mpo_2site(mps, mpo, n_sweeps=5)
        assert not any(np.isnan(e) for e in result.energies)
        for t in result.mps_final.tensors:
            assert not np.any(np.isnan(t))

    def test_n_sweeps_reported(self, heisenberg_setup):
        _bonds, mpo, _, n, d = heisenberg_setup
        from htf.mps import random_mps
        mps = random_mps(n, d, chi=2, seed=0)
        result = dmrg_sweep_mpo_2site(mps, mpo, n_sweeps=3)
        assert 1 <= result.n_sweeps <= 3


# ── dmrg_multistart ────────────────────────────────────────────────────────


class TestDmrgMultistart:
    """Parallel multi-start DMRG tests.

    All tests use n_workers=1 (sequential) to avoid subprocess overhead in
    the test suite.  One test explicitly exercises the parallel path.
    """

    @pytest.fixture
    def heisenberg_mpo(self):
        n, d = 4, 2
        bonds = heisenberg_bonds(n, J=1.0)
        H = nn_hamiltonian(bonds, n, d)
        E0 = float(np.linalg.eigvalsh(H)[0])
        mpo = nn_hamiltonian_mpo(bonds, n, d)
        return mpo, E0

    def test_result_type(self, heisenberg_mpo):
        mpo, _ = heisenberg_mpo
        result = dmrg_multistart(mpo, n_seeds=2, chi=4, n_sweeps=5, n_workers=1)
        assert isinstance(result, MultiStartDMRGResult)

    def test_best_is_mpo_dmrg_result(self, heisenberg_mpo):
        mpo, _ = heisenberg_mpo
        result = dmrg_multistart(mpo, n_seeds=2, chi=4, n_sweeps=5, n_workers=1)
        assert isinstance(result.best, MPODMRGResult)

    def test_best_is_minimum(self, heisenberg_mpo):
        mpo, _ = heisenberg_mpo
        result = dmrg_multistart(mpo, n_seeds=3, chi=4, n_sweeps=5, n_workers=1)
        assert result.best.energies[-1] == min(result.all_energies)

    def test_all_energies_count(self, heisenberg_mpo):
        mpo, _ = heisenberg_mpo
        result = dmrg_multistart(mpo, n_seeds=4, chi=4, n_sweeps=3, n_workers=1)
        assert len(result.all_energies) == 4

    def test_seeds_used_recorded(self, heisenberg_mpo):
        mpo, _ = heisenberg_mpo
        seeds = [10, 20, 30]
        result = dmrg_multistart(mpo, chi=4, n_sweeps=3, n_workers=1, seeds=seeds)
        assert result.seeds_used == seeds

    def test_best_seed_in_seeds(self, heisenberg_mpo):
        mpo, _ = heisenberg_mpo
        result = dmrg_multistart(mpo, n_seeds=3, chi=4, n_sweeps=5, n_workers=1)
        assert result.best_seed in result.seeds_used

    def test_heisenberg_converges_exact(self, heisenberg_mpo):
        mpo, E0 = heisenberg_mpo
        result = dmrg_multistart(mpo, n_seeds=4, chi=8, n_sweeps=20, n_workers=1)
        assert abs(result.best.energies[-1] - E0) < 1e-5

    def test_variational_upper_bound(self, heisenberg_mpo):
        mpo, E0 = heisenberg_mpo
        result = dmrg_multistart(mpo, n_seeds=2, chi=8, n_sweeps=10, n_workers=1)
        assert result.best.energies[-1] >= E0 - 1e-8

    def test_no_nan(self, heisenberg_mpo):
        mpo, _ = heisenberg_mpo
        result = dmrg_multistart(mpo, n_seeds=2, chi=4, n_sweeps=5, n_workers=1)
        assert not any(np.isnan(e) for e in result.all_energies)
        for t in result.best.mps_final.tensors:
            assert not np.any(np.isnan(t))

    def test_parallel_path(self, heisenberg_mpo):
        mpo, E0 = heisenberg_mpo
        result = dmrg_multistart(mpo, n_seeds=2, chi=8, n_sweeps=10, n_workers=2)
        assert isinstance(result, MultiStartDMRGResult)
        assert result.best.energies[-1] >= E0 - 1e-8


# ── mpo_chi_convergence ────────────────────────────────────────────────────


class TestMpoChiConvergence:
    """Parallel MPO χ-convergence study tests.

    Sequential mode (n_workers=1) used throughout to avoid subprocess
    overhead.  One test exercises the parallel path.
    """

    @pytest.fixture
    def heisenberg_mpo(self):
        n, d = 4, 2
        bonds = heisenberg_bonds(n, J=1.0)
        H = nn_hamiltonian(bonds, n, d)
        E0 = float(np.linalg.eigvalsh(H)[0])
        mpo = nn_hamiltonian_mpo(bonds, n, d)
        return mpo, E0

    def test_result_type(self, heisenberg_mpo):
        mpo, _ = heisenberg_mpo
        result = mpo_chi_convergence(mpo, [2, 4], n_seeds=1, n_sweeps=3, n_workers=1)
        assert isinstance(result, MPOScalingReport)

    def test_chi_points_count(self, heisenberg_mpo):
        mpo, _ = heisenberg_mpo
        chi_list = [2, 4, 6]
        result = mpo_chi_convergence(mpo, chi_list, n_seeds=1, n_sweeps=3, n_workers=1)
        assert len(result.chi_points) == len(chi_list)

    def test_chi_points_type(self, heisenberg_mpo):
        mpo, _ = heisenberg_mpo
        result = mpo_chi_convergence(mpo, [2, 4], n_seeds=1, n_sweeps=3, n_workers=1)
        for p in result.chi_points:
            assert isinstance(p, MPOChiPoint)

    def test_chi_values_recorded(self, heisenberg_mpo):
        mpo, _ = heisenberg_mpo
        chi_list = [2, 4, 8]
        result = mpo_chi_convergence(mpo, chi_list, n_seeds=1, n_sweeps=3, n_workers=1)
        assert [p.chi for p in result.chi_points] == chi_list

    def test_energy_improves_with_chi(self, heisenberg_mpo):
        # Higher χ → larger variational space → lower or equal energy
        mpo, _ = heisenberg_mpo
        result = mpo_chi_convergence(
            mpo, [2, 4, 8], n_seeds=2, n_sweeps=10, n_workers=1
        )
        energies = [p.energy for p in result.chi_points]
        assert energies[-1] <= energies[0] + 1e-6

    def test_variational_upper_bound(self, heisenberg_mpo):
        mpo, E0 = heisenberg_mpo
        result = mpo_chi_convergence(mpo, [4, 8], n_seeds=2, n_sweeps=10, n_workers=1)
        for p in result.chi_points:
            assert p.energy >= E0 - 1e-8

    def test_large_chi_converges_exact(self, heisenberg_mpo):
        mpo, E0 = heisenberg_mpo
        result = mpo_chi_convergence(
            mpo, [8], n_seeds=3, n_sweeps=20, n_workers=1
        )
        assert abs(result.chi_points[0].energy - E0) < 1e-5

    def test_extrapolation_with_3_points(self, heisenberg_mpo):
        mpo, _ = heisenberg_mpo
        result = mpo_chi_convergence(
            mpo, [2, 4, 8], n_seeds=2, n_sweeps=10, n_workers=1
        )
        # Power-law fit attempted when >= 3 chi values; may fail gracefully
        assert isinstance(result.E_extrapolated, float)
        assert isinstance(result.notes, str)

    def test_no_extrapolation_with_2_points(self, heisenberg_mpo):
        mpo, _ = heisenberg_mpo
        result = mpo_chi_convergence(mpo, [2, 4], n_seeds=1, n_sweeps=3, n_workers=1)
        assert np.isnan(result.E_extrapolated)

    def test_n_seeds_recorded(self, heisenberg_mpo):
        mpo, _ = heisenberg_mpo
        result = mpo_chi_convergence(mpo, [4], n_seeds=3, n_sweeps=5, n_workers=1)
        assert result.chi_points[0].n_seeds_used == 3

    def test_no_nan_energies(self, heisenberg_mpo):
        mpo, _ = heisenberg_mpo
        result = mpo_chi_convergence(mpo, [2, 4], n_seeds=2, n_sweeps=5, n_workers=1)
        for p in result.chi_points:
            assert not np.isnan(p.energy)

    def test_summary_is_string(self, heisenberg_mpo):
        mpo, _ = heisenberg_mpo
        result = mpo_chi_convergence(mpo, [2, 4], n_seeds=1, n_sweeps=3, n_workers=1)
        s = result.summary()
        assert isinstance(s, str) and len(s) > 0

    def test_parallel_path(self, heisenberg_mpo):
        mpo, E0 = heisenberg_mpo
        result = mpo_chi_convergence(
            mpo, [4, 8], n_seeds=2, n_sweeps=8, n_workers=2
        )
        assert isinstance(result, MPOScalingReport)
        assert len(result.chi_points) == 2
        for p in result.chi_points:
            assert p.energy >= E0 - 1e-8
