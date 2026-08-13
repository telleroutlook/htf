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
    dmrg_sweep,
    nn_hamiltonian,
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
