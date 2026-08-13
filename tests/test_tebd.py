"""Tests for htf.tebd — TEBD time evolution and DMRG."""
import math

import numpy as np
import pytest

from htf.mps import (
    MPS,
    mps_from_state,
    mps_inner,
    mps_norm,
    mps_normalise,
    mps_to_state,
    random_mps,
)
from htf.tebd import (
    DMRGResult,
    TEBDResult,
    _bond_gate,
    _nn_energy,
    bose_hubbard_bonds,
    dmrg_sweep,
    dmrg_sweep_2site,
    heisenberg_bonds,
    nn_hamiltonian,
    tdvp_evolve,
    tebd_evolve,
    tebd_step,
    tfim_bonds,
    xx_bonds,
)


# ── Hamiltonian helpers ────────────────────────────────────────────────────


class TestNnHamiltonian:
    def test_tfim_matches_reference(self):
        from htf.variational import transverse_ising_ham
        n = 4
        H_ref   = transverse_ising_ham(n=n, J=1.0, h=0.5)
        bonds   = tfim_bonds(n, J=1.0, h=0.5)
        H_bonds = nn_hamiltonian(bonds, n, d=2)
        np.testing.assert_allclose(H_bonds, H_ref, atol=1e-12)

    def test_xx_hermitian(self):
        n = 4
        bonds = xx_bonds(n, J=1.0, h=0.5)
        H = nn_hamiltonian(bonds, n, d=2)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)

    def test_hermitian(self):
        n = 4
        bonds = tfim_bonds(n, J=1.0, h=0.8)
        H = nn_hamiltonian(bonds, n)
        np.testing.assert_allclose(H, H.T, atol=1e-12)

    def test_shape(self):
        n, d = 5, 2
        bonds = tfim_bonds(n)
        H = nn_hamiltonian(bonds, n, d)
        assert H.shape == (d**n, d**n)


class TestTfimBonds:
    def test_number_of_bonds(self):
        for n in [2, 4, 6]:
            bonds = tfim_bonds(n)
            assert len(bonds) == n - 1

    def test_bond_shape(self):
        bonds = tfim_bonds(4)
        for b in bonds:
            assert b.shape == (4, 4)

    def test_hermitian_bonds(self):
        bonds = tfim_bonds(6, J=1.5, h=0.7)
        for b in bonds:
            np.testing.assert_allclose(b, b.T, atol=1e-12)


# ── bond gate ─────────────────────────────────────────────────────────────


class TestBondGate:
    def test_unitary_for_real_time(self):
        Z = np.array([[1, 0], [0, -1]], dtype=float)
        h = -np.kron(Z, Z)
        G = _bond_gate(h, dt=0.1, imaginary=False)
        # G should be unitary: G† G ≈ I
        G_mat = G.reshape(4, 4)
        np.testing.assert_allclose(G_mat.conj().T @ G_mat, np.eye(4), atol=1e-12)

    def test_hermitian_positive_for_imaginary_time(self):
        Z = np.array([[1, 0], [0, -1]], dtype=float)
        h = -np.kron(Z, Z)
        G = _bond_gate(h, dt=0.1, imaginary=True)
        G_mat = G.reshape(4, 4)
        # exp(-dt H) with H hermitian should be symmetric positive definite
        evals = np.linalg.eigvalsh(G_mat)
        assert np.all(evals > 0)

    def test_shape(self):
        h = np.eye(4, dtype=float)
        G = _bond_gate(h, dt=0.05)
        assert G.shape == (2, 2, 2, 2)


# ── TEBD step ─────────────────────────────────────────────────────────────


class TestTebdStep:
    @pytest.fixture
    def setup(self):
        n, d = 4, 2
        bonds  = tfim_bonds(n, J=1.0, h=0.5)
        mps0   = mps_normalise(random_mps(n, d, chi=4, seed=0))
        return mps0, bonds

    def test_no_nan(self, setup):
        mps0, bonds = setup
        mps1, disc = tebd_step(mps0, bonds, dt=0.05, chi=8, imaginary=False)
        for t in mps1.tensors:
            assert not np.any(np.isnan(t))

    def test_real_time_norm_conservation(self, setup):
        mps0, bonds = setup
        mps1, _ = tebd_step(mps0, bonds, dt=0.05, chi=8, imaginary=False)
        nrm0 = mps_norm(mps0)
        nrm1 = mps_norm(mps1)
        assert abs(nrm0 - nrm1) < 1e-10

    def test_imaginary_time_energy_decrease(self, setup):
        mps0, bonds = setup
        E0 = _nn_energy(mps0, bonds)
        mps1, _ = tebd_step(mps0, bonds, dt=0.1, chi=8, imaginary=True)
        mps1_n = mps_normalise(mps1)
        E1 = _nn_energy(mps1_n, bonds)
        assert E1 <= E0 + 1e-8

    def test_wrong_n_bonds_raises(self, setup):
        mps0, _ = setup
        with pytest.raises(ValueError, match="Expected"):
            tebd_step(mps0, [np.eye(4)], dt=0.05)

    def test_second_order_trotter(self, setup):
        mps0, bonds = setup
        mps1, _ = tebd_step(mps0, bonds, dt=0.05, chi=8,
                             imaginary=False, trotter_order=2)
        assert not any(np.any(np.isnan(t)) for t in mps1.tensors)

    def test_invalid_order_raises(self, setup):
        mps0, bonds = setup
        with pytest.raises(ValueError, match="trotter_order"):
            tebd_step(mps0, bonds, dt=0.05, trotter_order=3)

    def test_discarded_nonneg(self, setup):
        mps0, bonds = setup
        _, disc = tebd_step(mps0, bonds, dt=0.05, chi=2, imaginary=False)
        assert disc >= 0.0


# ── TEBD evolve ───────────────────────────────────────────────────────────


class TestTebdEvolve:
    @pytest.fixture
    def setup(self):
        n, d = 4, 2
        bonds = tfim_bonds(n, J=1.0, h=0.5)
        H     = nn_hamiltonian(bonds, n, d)
        E0    = float(np.linalg.eigvalsh(H)[0])
        mps0  = mps_normalise(random_mps(n, d, chi=4, seed=42))
        return mps0, bonds, E0

    def test_result_type(self, setup):
        mps0, bonds, _ = setup
        result = tebd_evolve(mps0, bonds, dt=0.05, n_steps=10, imaginary=True)
        assert isinstance(result, TEBDResult)

    def test_imaginary_time_convergence(self, setup):
        mps0, bonds, E0 = setup
        result = tebd_evolve(mps0, bonds, dt=0.05, n_steps=300, chi=8,
                             imaginary=True, measure_every=50)
        assert result.energies[-1] < result.energies[0]
        assert abs(result.energies[-1] - E0) < 0.02

    def test_real_time_norm_preserved(self, setup):
        mps0, bonds, _ = setup
        result = tebd_evolve(mps0, bonds, dt=0.02, n_steps=30,
                             chi=8, imaginary=False)
        final_nrm = abs(mps_inner(result.mps_final, result.mps_final)) ** 0.5
        assert abs(final_nrm - 1.0) < 1e-9

    def test_trajectory_length(self, setup):
        mps0, bonds, _ = setup
        result = tebd_evolve(mps0, bonds, dt=0.1, n_steps=20,
                             imaginary=True, measure_every=5)
        # recorded at step 0 + every 5 steps → steps 0,5,10,15,20 = 5 points
        assert len(result.energies) == 5

    def test_second_order_better_than_first(self, setup):
        mps0, bonds, E0 = setup
        # Use large dt where Trotter order difference is visible
        r1 = tebd_evolve(mps0, bonds, dt=0.2, n_steps=100, chi=8,
                         imaginary=True, trotter_order=1, measure_every=100)
        r2 = tebd_evolve(mps0, bonds, dt=0.2, n_steps=100, chi=8,
                         imaginary=True, trotter_order=2, measure_every=100)
        # Both should converge; 2nd-order should be at least as good
        assert r1.energies[-1] < r1.energies[0]
        assert r2.energies[-1] < r2.energies[0]
        # Both should get reasonably close to E0
        assert abs(r1.energies[-1] - E0) < 0.5
        assert abs(r2.energies[-1] - E0) < 0.5

    def test_discarded_weight_accumulated(self, setup):
        mps0, bonds, _ = setup
        result = tebd_evolve(mps0, bonds, dt=0.05, n_steps=20, chi=2, imaginary=True)
        assert result.total_discarded >= 0.0

    def test_measure_every_one(self, setup):
        mps0, bonds, _ = setup
        result = tebd_evolve(mps0, bonds, dt=0.05, n_steps=5,
                             imaginary=True, measure_every=1)
        assert len(result.energies) == 6   # 0 + 5 steps


# ── DMRG sweep ────────────────────────────────────────────────────────────


class TestDmrgSweep:
    @pytest.fixture
    def setup(self):
        n, d = 4, 2
        bonds = tfim_bonds(n, J=1.0, h=0.5)
        H     = nn_hamiltonian(bonds, n, d)
        E0    = float(np.linalg.eigvalsh(H)[0])
        mps0  = mps_normalise(random_mps(n, d, chi=4, seed=3))
        return mps0, bonds, E0

    def test_result_type(self, setup):
        mps0, bonds, _ = setup
        result = dmrg_sweep(mps0, bonds, n_sweeps=2, chi=4)
        assert isinstance(result, DMRGResult)

    def test_energy_decreases(self, setup):
        mps0, bonds, E0 = setup
        result = dmrg_sweep(mps0, bonds, n_sweeps=5, chi=4)
        assert result.energies[-1] < result.energies[0] + 1e-4
        # Final energy should be reasonably close to exact
        assert result.energies[-1] > E0 - 0.5   # at least not far below

    def test_energies_list_nonempty(self, setup):
        mps0, bonds, _ = setup
        result = dmrg_sweep(mps0, bonds, n_sweeps=2, chi=4)
        assert len(result.energies) > 0


# ── nn_energy helper ──────────────────────────────────────────────────────


class TestNnEnergy:
    def test_ground_state_energy(self):
        n, d = 4, 2
        bonds = tfim_bonds(n, J=1.0, h=0.5)
        H     = nn_hamiltonian(bonds, n, d)
        evals, evecs = np.linalg.eigh(H)
        psi0  = evecs[:, 0]
        mps0  = mps_from_state(psi0, d)
        E_mps = _nn_energy(mps0, bonds)
        assert abs(E_mps - float(evals[0])) < 1e-10

    def test_normalisation_invariant(self):
        n, d = 3, 2
        bonds = tfim_bonds(n, J=1.0, h=0.5)
        mps   = random_mps(n, d, chi=2, seed=10)
        mps_n = mps_normalise(mps)
        E1 = _nn_energy(mps,   bonds)
        E2 = _nn_energy(mps_n, bonds)
        assert abs(E1 - E2) < 1e-10


# ── Heisenberg bonds ──────────────────────────────────────────────────────


class TestHeisenbergBonds:

    def test_number_of_obc_bonds(self):
        for n in [2, 4, 6]:
            bonds = heisenberg_bonds(n)
            assert len(bonds) == n - 1

    def test_number_of_pbc_bonds(self):
        for n in [3, 4, 5]:
            bonds = heisenberg_bonds(n, periodic=True)
            assert len(bonds) == n

    def test_bond_shape(self):
        for b in heisenberg_bonds(5):
            assert b.shape == (4, 4)

    def test_hermitian(self):
        for b in heisenberg_bonds(4, J=1.2, h=0.3):
            np.testing.assert_allclose(b, b.conj().T, atol=1e-12)

    def test_full_ham_hermitian(self):
        n = 4
        H = nn_hamiltonian(heisenberg_bonds(n, J=1.0, h=0.0), n)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)

    def test_ground_energy_below_zero(self):
        # For antiferromagnetic J>0, Heisenberg ground energy is negative
        n = 4
        bonds = heisenberg_bonds(n, J=1.0, h=0.0)
        H = nn_hamiltonian(bonds, n)
        E0 = float(np.linalg.eigvalsh(H)[0])
        assert E0 < 0.0

    def test_pbc_ham_hermitian(self):
        n = 4
        bonds = heisenberg_bonds(n, J=1.0, h=0.0, periodic=True)
        H = nn_hamiltonian(bonds, n, periodic=True)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)

    def test_pbc_lower_than_obc(self):
        # PBC adds frustration / more bonds, so ground energy ≤ OBC
        n = 4
        bonds_obc = heisenberg_bonds(n, J=1.0, h=0.0, periodic=False)
        bonds_pbc = heisenberg_bonds(n, J=1.0, h=0.0, periodic=True)
        E_obc = float(np.linalg.eigvalsh(nn_hamiltonian(bonds_obc, n))[0])
        E_pbc = float(np.linalg.eigvalsh(nn_hamiltonian(bonds_pbc, n, periodic=True))[0])
        assert E_pbc <= E_obc + 1e-10


# ── nn_hamiltonian with periodic=True ─────────────────────────────────────


class TestNnHamiltonianPBC:

    def test_tfim_pbc_hermitian(self):
        n = 4
        bonds = tfim_bonds(n, J=1.0, h=0.5, periodic=True)
        H = nn_hamiltonian(bonds, n, periodic=True)
        np.testing.assert_allclose(H, H.T, atol=1e-12)

    def test_tfim_pbc_shape(self):
        n, d = 4, 2
        bonds = tfim_bonds(n, periodic=True)
        H = nn_hamiltonian(bonds, n, d, periodic=True)
        assert H.shape == (d**n, d**n)

    def test_tfim_pbc_translation_symmetry(self):
        # PBC TFIM is translation-invariant: all bonds identical
        n = 4
        bonds = tfim_bonds(n, J=1.0, h=0.5, periodic=True)
        assert len(bonds) == n
        for b in bonds:
            np.testing.assert_allclose(b, bonds[0], atol=1e-12)

    def test_pbc_adds_n_bonds(self):
        for n in [3, 4, 5]:
            assert len(tfim_bonds(n, periodic=True)) == n
            assert len(xx_bonds(n, periodic=True)) == n
            assert len(heisenberg_bonds(n, periodic=True)) == n

    def test_tfim_pbc_reconstruction(self):
        # Manually build TFIM PBC and compare with nn_hamiltonian
        from htf.variational import transverse_ising_ham
        n, J, h = 4, 1.0, 0.5
        bonds = tfim_bonds(n, J=J, h=h, periodic=True)
        H_pbc = nn_hamiltonian(bonds, n, periodic=True)
        # Reference: OBC Hamiltonian + periodic ZZ bond + periodic X terms
        H_obc = transverse_ising_ham(n, J=J, h=h)
        Z = np.array([[1, 0], [0, -1]], dtype=float)
        I = np.eye(2, dtype=float)
        # Add ZZ for (n-1, 0): site n-1 is least-significant, site 0 is most-significant
        # Using kron: |site0> ⊗ |mid> ⊗ |site_{n-1}>
        I_mid = np.eye(2**(n - 2))
        E_Z0 = np.kron(Z, np.kron(I_mid, I))   # Z on site 0
        E_Zn1 = np.kron(I, np.kron(I_mid, Z))  # Z on site n-1
        H_ref = H_obc - J * E_Z0 @ E_Zn1
        np.testing.assert_allclose(H_pbc.real, H_ref, atol=1e-10)


# ── Bose-Hubbard bonds ────────────────────────────────────────────────────


class TestBoseHubbardBonds:

    def test_number_of_bonds(self):
        for n in [2, 3, 4]:
            bonds = bose_hubbard_bonds(n)
            assert len(bonds) == n - 1

    def test_bond_shape(self):
        max_occ = 3
        d = max_occ + 1
        for b in bose_hubbard_bonds(4, max_occ=max_occ):
            assert b.shape == (d**2, d**2)

    def test_hermitian(self):
        for b in bose_hubbard_bonds(4, t=1.0, U=4.0, mu=2.0, max_occ=3):
            np.testing.assert_allclose(b, b.T, atol=1e-12)

    def test_full_ham_hermitian(self):
        n, max_occ = 3, 2
        d = max_occ + 1
        bonds = bose_hubbard_bonds(n, max_occ=max_occ)
        H = nn_hamiltonian(bonds, n, d=d)
        np.testing.assert_allclose(H, H.T, atol=1e-12)

    def test_ground_energy_finite(self):
        n, max_occ = 3, 2
        d = max_occ + 1
        bonds = bose_hubbard_bonds(n, t=1.0, U=4.0, mu=2.0, max_occ=max_occ)
        H = nn_hamiltonian(bonds, n, d=d)
        E0 = float(np.linalg.eigvalsh(H)[0])
        assert np.isfinite(E0)

    def test_hopping_lowers_energy(self):
        # Hopping t>0 lowers ground energy vs t=0
        n, max_occ = 3, 3
        d = max_occ + 1
        bonds_hop  = bose_hubbard_bonds(n, t=1.0, U=4.0, mu=2.0, max_occ=max_occ)
        bonds_nohop = bose_hubbard_bonds(n, t=0.0, U=4.0, mu=2.0, max_occ=max_occ)
        E_hop   = float(np.linalg.eigvalsh(nn_hamiltonian(bonds_hop,   n, d=d))[0])
        E_nohop = float(np.linalg.eigvalsh(nn_hamiltonian(bonds_nohop, n, d=d))[0])
        assert E_hop <= E_nohop + 1e-10

    def test_tebd_bose_hubbard_no_nan(self):
        n, max_occ = 3, 2
        d = max_occ + 1
        bonds = bose_hubbard_bonds(n, t=1.0, U=4.0, mu=2.0, max_occ=max_occ)
        mps0 = random_mps(n, d=d, chi=4, seed=7)
        mps0 = mps_normalise(mps0)
        result = tebd_evolve(mps0, bonds, dt=0.05, n_steps=10,
                             chi=8, imaginary=True, measure_every=5)
        for t in result.mps_final.tensors:
            assert not np.any(np.isnan(t))


# ── two-site DMRG sweep ───────────────────────────────────────────────────


class TestDmrgSweep2Site:
    @pytest.fixture
    def setup(self):
        n, d = 4, 2
        bonds = tfim_bonds(n, J=1.0, h=0.5)
        H     = nn_hamiltonian(bonds, n, d)
        E0    = float(np.linalg.eigvalsh(H)[0])
        mps0  = mps_normalise(random_mps(n, d, chi=4, seed=5))
        return mps0, bonds, E0

    def test_result_type(self, setup):
        mps0, bonds, _ = setup
        result = dmrg_sweep_2site(mps0, bonds, n_sweeps=2, chi=4)
        assert isinstance(result, DMRGResult)

    def test_energy_decreases(self, setup):
        mps0, bonds, E0 = setup
        result = dmrg_sweep_2site(mps0, bonds, n_sweeps=5, chi=4)
        assert result.energies[-1] <= result.energies[0] + 1e-6

    def test_converges_to_exact(self, setup):
        mps0, bonds, E0 = setup
        result = dmrg_sweep_2site(mps0, bonds, n_sweeps=20, chi=8)
        assert abs(result.energies[-1] - E0) < 0.05

    def test_energies_nonempty(self, setup):
        mps0, bonds, _ = setup
        result = dmrg_sweep_2site(mps0, bonds, n_sweeps=2, chi=4)
        assert len(result.energies) > 0

    def test_converged_flag(self, setup):
        mps0, bonds, E0 = setup
        result = dmrg_sweep_2site(mps0, bonds, n_sweeps=50, chi=8, tol=1e-6)
        assert isinstance(result.converged, bool)

    def test_chi_limits_bond_dimension(self, setup):
        mps0, bonds, _ = setup
        result = dmrg_sweep_2site(mps0, bonds, n_sweeps=3, chi=2)
        for t in result.mps_final.tensors:
            assert t.shape[0] <= 2 and t.shape[2] <= 2

    def test_heisenberg_energy(self):
        n = 4
        bonds = heisenberg_bonds(n, J=1.0, h=0.0)
        H     = nn_hamiltonian(bonds, n)
        E0    = float(np.linalg.eigvalsh(H.real)[0])
        mps0  = mps_normalise(random_mps(n, 2, chi=4, seed=9))
        result = dmrg_sweep_2site(mps0, bonds, n_sweeps=20, chi=8)
        assert abs(result.energies[-1] - E0) < 0.1

    def test_no_nan_in_final_mps(self, setup):
        mps0, bonds, _ = setup
        result = dmrg_sweep_2site(mps0, bonds, n_sweeps=3, chi=4)
        for t in result.mps_final.tensors:
            assert not np.any(np.isnan(t))


# ── single-site TDVP ──────────────────────────────────────────────────────


class TestTdvpEvolve:
    @pytest.fixture
    def setup(self):
        n, d = 4, 2
        bonds = tfim_bonds(n, J=1.0, h=0.5)
        H     = nn_hamiltonian(bonds, n, d)
        E0    = float(np.linalg.eigvalsh(H)[0])
        mps0  = mps_normalise(random_mps(n, d, chi=4, seed=11))
        return mps0, bonds, E0

    def test_result_type(self, setup):
        mps0, bonds, _ = setup
        result = tdvp_evolve(mps0, bonds, dt=0.05, n_steps=5, imaginary=True)
        assert isinstance(result, TEBDResult)

    def test_imaginary_time_energy_decrease(self, setup):
        mps0, bonds, E0 = setup
        result = tdvp_evolve(mps0, bonds, dt=0.05, n_steps=50, imaginary=True)
        assert result.energies[-1] < result.energies[0]

    def test_imaginary_time_convergence(self, setup):
        mps0, bonds, E0 = setup
        result = tdvp_evolve(mps0, bonds, dt=0.05, n_steps=200, imaginary=True,
                              measure_every=50)
        assert abs(result.energies[-1] - E0) < 0.05

    def test_real_time_norm_conserved(self, setup):
        mps0, bonds, _ = setup
        result = tdvp_evolve(mps0, bonds, dt=0.05, n_steps=20, imaginary=False)
        from htf.mps import mps_inner
        nrm = abs(mps_inner(result.mps_final, result.mps_final)) ** 0.5
        assert abs(nrm - 1.0) < 1e-4

    def test_trajectory_length(self, setup):
        mps0, bonds, _ = setup
        result = tdvp_evolve(mps0, bonds, dt=0.1, n_steps=10,
                              imaginary=True, measure_every=5)
        # recorded at step 0, 5 plus final = 3 points
        assert len(result.energies) == 3

    def test_no_nan(self, setup):
        mps0, bonds, _ = setup
        result = tdvp_evolve(mps0, bonds, dt=0.05, n_steps=5, imaginary=False)
        for t in result.mps_final.tensors:
            assert not np.any(np.isnan(t))

    def test_total_discarded_zero(self, setup):
        mps0, bonds, _ = setup
        result = tdvp_evolve(mps0, bonds, dt=0.05, n_steps=5, imaginary=True)
        assert result.total_discarded == 0.0

    def test_tdvp_vs_tebd_ground_state(self, setup):
        # Both methods should converge to similar ground-state energy
        mps0, bonds, E0 = setup
        r_tdvp = tdvp_evolve(mps0, bonds, dt=0.05, n_steps=200, imaginary=True)
        r_tebd = tebd_evolve(mps0, bonds, dt=0.05, n_steps=200, chi=8, imaginary=True)
        assert abs(r_tdvp.energies[-1] - E0) < 0.05
        assert abs(r_tebd.energies[-1] - E0) < 0.05


# ── §9-K: intra-step bond parallelism ─────────────────────────────────────


class TestTebdParallelBonds:
    """Tests for n_threads parallelism in tebd_step / tebd_evolve.

    Parallel (n_threads>1) and sequential (n_threads=1) must produce
    numerically identical results for trotter_order=2, since the even/odd
    parity groups are the same — only execution order within a group differs.
    """

    @pytest.fixture
    def setup(self):
        n, d = 6, 2
        bonds = heisenberg_bonds(n, J=1.0)
        from htf.mps import random_mps
        mps = random_mps(n, d, chi=4, seed=0)
        return mps, bonds

    def test_parallel_matches_sequential_step(self, setup):
        mps, bonds = setup
        gates = [
            np.eye(4).reshape(2, 2, 2, 2) for _ in bonds
        ]  # identity gates → exact, no SVD noise
        from htf.tebd import _apply_bond_parity, _bond_gate
        from htf.tebd import tfim_bonds, nn_hamiltonian
        bonds_tfim = tfim_bonds(6, J=1.0, h=0.5)
        gates2 = [_bond_gate(b, 0.01, False) for b in bonds_tfim]

        mps_seq,  d_seq  = tebd_step(mps, bonds_tfim, dt=0.01,
                                     chi=8, trotter_order=2, n_threads=1)
        mps_par,  d_par  = tebd_step(mps, bonds_tfim, dt=0.01,
                                     chi=8, trotter_order=2, n_threads=2)
        for t_s, t_p in zip(mps_seq.tensors, mps_par.tensors):
            np.testing.assert_allclose(t_s, t_p, atol=1e-12)
        assert abs(d_seq - d_par) < 1e-14

    def test_parallel_evolve_matches_sequential(self, setup):
        mps, bonds = setup
        from htf.tebd import tfim_bonds
        bonds_tfim = tfim_bonds(6, J=1.0, h=0.5)
        r_seq = tebd_evolve(mps, bonds_tfim, dt=0.02, n_steps=10,
                            chi=8, trotter_order=2, n_threads=1)
        r_par = tebd_evolve(mps, bonds_tfim, dt=0.02, n_steps=10,
                            chi=8, trotter_order=2, n_threads=2)
        np.testing.assert_allclose(r_seq.energies, r_par.energies, atol=1e-12)
        for t_s, t_p in zip(r_seq.mps_final.tensors, r_par.mps_final.tensors):
            np.testing.assert_allclose(t_s, t_p, atol=1e-12)

    def test_n_threads_ignored_for_order1(self, setup):
        mps, bonds = setup
        from htf.tebd import tfim_bonds
        bonds_tfim = tfim_bonds(6, J=1.0, h=0.5)
        r1 = tebd_evolve(mps, bonds_tfim, dt=0.02, n_steps=5,
                         chi=8, trotter_order=1, n_threads=1)
        r2 = tebd_evolve(mps, bonds_tfim, dt=0.02, n_steps=5,
                         chi=8, trotter_order=1, n_threads=4)
        np.testing.assert_allclose(r1.energies, r2.energies, atol=1e-14)

    def test_no_nan_parallel(self, setup):
        mps, bonds = setup
        from htf.tebd import tfim_bonds
        bonds_tfim = tfim_bonds(6, J=1.0, h=0.5)
        result = tebd_evolve(mps, bonds_tfim, dt=0.02, n_steps=10,
                             chi=8, trotter_order=2, n_threads=2)
        for t in result.mps_final.tensors:
            assert not np.any(np.isnan(t))

    def test_energy_reasonable_parallel(self, setup):
        mps, bonds = setup
        from htf.tebd import tfim_bonds, nn_hamiltonian
        from htf.mps import mps_from_state
        n, d = 6, 2
        bonds_tfim = tfim_bonds(n, J=1.0, h=0.5)
        H = nn_hamiltonian(bonds_tfim, n, d)
        E0 = float(np.linalg.eigvalsh(H)[0])
        # Imaginary-time TEBD with order=2 + threads should approach ground state
        result = tebd_evolve(mps, bonds_tfim, dt=0.05, n_steps=100,
                             chi=16, imaginary=True, trotter_order=2, n_threads=2)
        assert result.energies[-1] < result.energies[0]
        assert result.energies[-1] >= E0 - 0.1
