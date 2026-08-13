"""Tests for htf.thermal — finite-temperature thermal states."""
import numpy as np
import pytest

from htf.tebd import nn_hamiltonian, tfim_bonds
from htf.thermal import (
    ThermalResult,
    ThermalScanPoint,
    ThermalScanResult,
    purification_bonds,
    purified_initial_mps,
    thermal_expectation,
    thermal_scan,
    thermal_state,
)


class TestPurifiedInitialMps:
    def test_norm_is_one(self):
        from htf.mps import mps_inner
        mps = purified_initial_mps(4, d=2)
        nrm2 = abs(mps_inner(mps, mps))
        assert abs(nrm2 - 1.0) < 1e-12

    def test_super_site_dimension(self):
        d, n = 2, 4
        mps = purified_initial_mps(n, d=d)
        for t in mps.tensors:
            assert t.shape[1] == d * d

    def test_n_tensors(self):
        for n in [2, 3, 5]:
            assert len(purified_initial_mps(n).tensors) == n


class TestPurificationBonds:
    def test_output_count(self):
        bonds = tfim_bonds(4, J=1.0, h=0.5)
        ext = purification_bonds(bonds, d=2)
        assert len(ext) == len(bonds)

    def test_shape(self):
        d = 2
        bonds = tfim_bonds(4)
        for b in purification_bonds(bonds, d=d):
            assert b.shape == ((d * d) ** 2, (d * d) ** 2)

    def test_hermitian(self):
        bonds = tfim_bonds(4, J=1.0, h=0.5)
        for b in purification_bonds(bonds, d=2):
            np.testing.assert_allclose(b, b.T, atol=1e-12)


class TestThermalState:
    @pytest.fixture
    def setup(self):
        n, d = 3, 2
        bonds = tfim_bonds(n, J=1.0, h=0.5)
        H     = nn_hamiltonian(bonds, n, d)
        E0    = float(np.linalg.eigvalsh(H)[0])
        return n, d, bonds, E0

    def test_result_type(self, setup):
        n, d, bonds, _ = setup
        result = thermal_state(bonds, n, beta=1.0, chi=8, d=d, dt=0.1)
        assert isinstance(result, ThermalResult)

    def test_partition_function_infinite_T(self, setup):
        n, d, bonds, _ = setup
        # At β=0 (n_steps=0), Z = d^n exactly
        result = thermal_state(bonds, n, beta=0.0, chi=8, d=d, dt=0.1)
        assert abs(result.partition_function - d ** n) < 1e-10

    def test_energy_approaches_ground_state(self, setup):
        n, d, bonds, E0 = setup
        result = thermal_state(bonds, n, beta=6.0, chi=16, d=d, dt=0.05)
        assert abs(result.energies[-1] - E0) < 0.1

    def test_energy_decreases_with_cooling(self, setup):
        n, d, bonds, _ = setup
        result = thermal_state(bonds, n, beta=4.0, chi=8, d=d, dt=0.1,
                               measure_every=5)
        assert result.energies[-1] <= result.energies[0] + 1e-6

    def test_free_energy_le_energy(self, setup):
        # F = E - TS ≤ E at finite T
        n, d, bonds, _ = setup
        result = thermal_state(bonds, n, beta=2.0, chi=8, d=d, dt=0.1)
        assert result.free_energy_upper <= result.energies[-1] + 1e-8

    def test_no_nan(self, setup):
        n, d, bonds, _ = setup
        result = thermal_state(bonds, n, beta=1.0, chi=8, d=d, dt=0.1)
        for t in result.mps_purified.tensors:
            assert not np.any(np.isnan(t))

    def test_z_expectation_vanishes(self, setup):
        # TFIM has Z₂ symmetry (Z → -Z on all sites); ⟨Z_i⟩ = 0 for all β
        n, d, bonds, _ = setup
        result = thermal_state(bonds, n, beta=2.0, chi=8, d=d, dt=0.1)
        Z_op = np.array([[1.0, 0.0], [0.0, -1.0]])
        exp_Z = thermal_expectation(result.mps_purified, Z_op, 0, d)
        assert abs(exp_Z) < 1e-8

    def test_identity_expectation_is_one(self, setup):
        # ⟨I⟩_β = 1 (normalization check)
        n, d, bonds, _ = setup
        result = thermal_state(bonds, n, beta=2.0, chi=8, d=d, dt=0.1)
        I_op = np.eye(d)
        exp_I = thermal_expectation(result.mps_purified, I_op, 0, d)
        assert abs(exp_I - 1.0) < 1e-8


# ── thermal_scan ───────────────────────────────────────────────────────────


class TestThermalScan:
    """Parallel β-scan tests.  n_workers=1 (sequential) throughout except
    one test that explicitly exercises the parallel path."""

    @pytest.fixture
    def tfim_setup(self):
        n, d = 3, 2
        bonds = tfim_bonds(n, J=1.0, h=0.5)
        H = nn_hamiltonian(bonds, n, d)
        E0 = float(np.linalg.eigvalsh(H)[0])
        return n, d, bonds, E0

    def test_result_type(self, tfim_setup):
        n, d, bonds, _ = tfim_setup
        result = thermal_scan(bonds, n, [0.5, 1.0], chi=8, d=d, dt=0.1,
                              n_workers=1)
        assert isinstance(result, ThermalScanResult)

    def test_points_count(self, tfim_setup):
        n, d, bonds, _ = tfim_setup
        beta_list = [0.5, 1.0, 2.0]
        result = thermal_scan(bonds, n, beta_list, chi=8, d=d, dt=0.1,
                              n_workers=1)
        assert len(result.points) == len(beta_list)

    def test_points_type(self, tfim_setup):
        n, d, bonds, _ = tfim_setup
        result = thermal_scan(bonds, n, [1.0, 2.0], chi=8, d=d, dt=0.1,
                              n_workers=1)
        for p in result.points:
            assert isinstance(p, ThermalScanPoint)

    def test_sorted_by_beta(self, tfim_setup):
        n, d, bonds, _ = tfim_setup
        # Pass in reverse order — output must be sorted ascending
        result = thermal_scan(bonds, n, [3.0, 1.0, 0.5], chi=8, d=d, dt=0.1,
                              n_workers=1)
        betas = [p.beta for p in result.points]
        assert betas == sorted(betas)

    def test_energy_decreases_with_beta(self, tfim_setup):
        # Cooling (larger β) → lower thermal energy
        n, d, bonds, _ = tfim_setup
        result = thermal_scan(bonds, n, [0.5, 2.0, 5.0], chi=8, d=d, dt=0.05,
                              n_workers=1)
        energies = [p.energy for p in result.points]
        assert energies[-1] <= energies[0] + 0.1

    def test_large_beta_approaches_ground_state(self, tfim_setup):
        n, d, bonds, E0 = tfim_setup
        result = thermal_scan(bonds, n, [6.0], chi=16, d=d, dt=0.05,
                              n_workers=1)
        assert abs(result.points[0].energy - E0) < 0.1

    def test_no_nan(self, tfim_setup):
        n, d, bonds, _ = tfim_setup
        result = thermal_scan(bonds, n, [0.5, 1.0, 2.0], chi=8, d=d, dt=0.1,
                              n_workers=1)
        for p in result.points:
            assert not np.isnan(p.energy)

    def test_summary_is_string(self, tfim_setup):
        n, d, bonds, _ = tfim_setup
        result = thermal_scan(bonds, n, [1.0, 2.0], chi=8, d=d, dt=0.1,
                              n_workers=1)
        s = result.summary()
        assert isinstance(s, str) and "beta" in s

    def test_partition_function_positive(self, tfim_setup):
        n, d, bonds, _ = tfim_setup
        result = thermal_scan(bonds, n, [0.5, 1.0], chi=8, d=d, dt=0.1,
                              n_workers=1)
        for p in result.points:
            assert p.partition_function > 0

    def test_parallel_path(self, tfim_setup):
        n, d, bonds, _E0 = tfim_setup
        result = thermal_scan(bonds, n, [1.0, 2.0], chi=8, d=d, dt=0.1,
                              n_workers=2)
        assert isinstance(result, ThermalScanResult)
        assert len(result.points) == 2
        for p in result.points:
            assert not np.isnan(p.energy)
