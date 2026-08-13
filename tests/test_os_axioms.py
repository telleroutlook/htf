"""Tests for htf/os_axioms.py — transfer matrix and reflection diagnostics."""
import warnings

import numpy as np
import pytest

from htf.os_axioms import (
    check_reflection_symmetry,
    check_transfer_positivity,
    finite_lattice_reflection_diagnostics,
    os_positivity_report,
    reflection_operator,
    transfer_matrix,
)
from htf.variational import transverse_ising_ham, xx_model_ham

# ─────────────────── fixtures ────────────────────────────────────────────

@pytest.fixture
def tfim4():
    return transverse_ising_ham(4, J=1.0, h=0.5)


@pytest.fixture
def xx4():
    return xx_model_ham(4, J=1.0)


# ─────────────────── TestTransferMatrix ──────────────────────────────────

class TestTransferMatrix:

    def test_shape_matches_hamiltonian(self, tfim4):
        T = transfer_matrix(tfim4)
        assert T.shape == tfim4.shape

    def test_is_symmetric_for_symmetric_ham(self, tfim4):
        T = transfer_matrix(tfim4)
        assert np.allclose(T, T.T, atol=1e-12)

    def test_eigenvalues_all_positive(self, tfim4):
        T = transfer_matrix(tfim4)
        evals = np.linalg.eigvalsh(T)
        assert np.all(evals > 0)

    def test_beta_zero_gives_identity(self, tfim4):
        T = transfer_matrix(tfim4, beta=0.0)
        assert np.allclose(T, np.eye(16), atol=1e-12)

    def test_larger_beta_suppresses_excited_states(self, tfim4):
        evals_h = np.linalg.eigvalsh(transfer_matrix(tfim4, beta=1.0))
        evals_2h = np.linalg.eigvalsh(transfer_matrix(tfim4, beta=2.0))
        # largest eigenvalue (ground state) dominates more at larger beta
        ratio_1 = evals_h[-1] / evals_h[-2]
        ratio_2 = evals_2h[-1] / evals_2h[-2]
        assert ratio_2 > ratio_1

    def test_2x2_diagonal_ham(self):
        H = np.diag([-1.0, 1.0])
        T = transfer_matrix(H, beta=1.0)
        assert np.allclose(np.diag(T), [np.exp(1.0), np.exp(-1.0)], atol=1e-12)

    def test_returns_float_array(self, tfim4):
        T = transfer_matrix(tfim4)
        assert T.dtype == float

    def test_beta_scaling(self, tfim4):
        T1 = transfer_matrix(tfim4, beta=1.0)
        T2 = transfer_matrix(tfim4, beta=2.0)
        # T(2β) ≈ T(β)^2 for commuting matrices (diagonal basis)
        evals1 = np.sort(np.linalg.eigvalsh(T1))
        evals2 = np.sort(np.linalg.eigvalsh(T2))
        assert np.allclose(evals2, evals1 ** 2, atol=1e-10)


# ─────────────────── TestReflectionOperator ──────────────────────────────

class TestReflectionOperator:

    def test_shape_2_qubit(self):
        R = reflection_operator(2, d=2)
        assert R.shape == (4, 4)

    def test_shape_4_qubit(self):
        R = reflection_operator(4, d=2)
        assert R.shape == (16, 16)

    def test_shape_qutrit(self):
        R = reflection_operator(3, d=3)
        assert R.shape == (27, 27)

    def test_is_orthogonal(self):
        R = reflection_operator(3, d=2)
        assert np.allclose(R @ R.T, np.eye(8), atol=1e-12)

    def test_involution_R_squared_is_identity(self):
        R = reflection_operator(4, d=2)
        assert np.allclose(R @ R, np.eye(16), atol=1e-12)

    def test_1_site_is_identity(self):
        R = reflection_operator(1, d=2)
        assert np.allclose(R, np.eye(2), atol=1e-12)

    def test_2_qubit_explicit(self):
        # |01⟩ ↔ |10⟩, |00⟩ and |11⟩ fixed
        R = reflection_operator(2, d=2)
        # big-endian: |00⟩=0, |01⟩=1, |10⟩=2, |11⟩=3
        basis = np.eye(4)
        assert np.allclose(R @ basis[0], basis[0])   # |00⟩ → |00⟩
        assert np.allclose(R @ basis[1], basis[2])   # |01⟩ → |10⟩
        assert np.allclose(R @ basis[2], basis[1])   # |10⟩ → |01⟩
        assert np.allclose(R @ basis[3], basis[3])   # |11⟩ → |11⟩

    def test_3_qubit_middle_preserved(self):
        # |010⟩ should map to |010⟩ (palindrome)
        R = reflection_operator(3, d=2)
        # |010⟩ = index 0*4 + 1*2 + 0*1 = 2
        v = np.zeros(8)
        v[2] = 1.0
        assert np.allclose(R @ v, v, atol=1e-12)

    def test_determinant_is_pm1(self):
        R = reflection_operator(3, d=2)
        det = np.linalg.det(R)
        assert abs(abs(det) - 1.0) < 1e-10

    def test_sum_of_rows_is_one(self):
        R = reflection_operator(3, d=2)
        assert np.allclose(R.sum(axis=1), 1.0, atol=1e-12)

    def test_is_permutation_matrix(self):
        R = reflection_operator(3, d=2)
        assert np.all((R == 0) | (R == 1))
        assert np.allclose(R.sum(axis=0), 1.0)
        assert np.allclose(R.sum(axis=1), 1.0)


# ─────────────────── TestCheckTransferPositivity ─────────────────────────

class TestCheckTransferPositivity:

    def test_tfim_passes(self, tfim4):
        r = check_transfer_positivity(tfim4)
        assert r.passed

    def test_xx_model_passes(self, xx4):
        r = check_transfer_positivity(xx4)
        assert r.passed

    def test_defect_near_zero_for_hermitian(self, tfim4):
        r = check_transfer_positivity(tfim4)
        assert r.defect < 1e-10

    def test_property_name(self, tfim4):
        r = check_transfer_positivity(tfim4)
        assert r.property_name == "transfer_matrix_positivity"

    def test_notes_contains_beta(self, tfim4):
        r = check_transfer_positivity(tfim4, beta=2.5)
        assert "2.5" in r.notes

    def test_larger_beta_still_passes(self, tfim4):
        r = check_transfer_positivity(tfim4, beta=5.0)
        assert r.passed

    def test_returns_structure_report(self, tfim4):
        from htf.structure import StructureReport
        r = check_transfer_positivity(tfim4)
        assert isinstance(r, StructureReport)

    def test_diagonal_ham_passes(self):
        H = np.diag([0.0, 1.0, 2.0])
        r = check_transfer_positivity(H)
        assert r.passed

    def test_2_site_tfim(self):
        H = transverse_ising_ham(2, J=1.0, h=1.0)
        r = check_transfer_positivity(H)
        assert r.passed


# ─────────────────── TestCheckReflectionSymmetry ─────────────────────────

class TestCheckReflectionSymmetry:

    def test_tfim_is_reflection_symmetric(self, tfim4):
        r = check_reflection_symmetry(tfim4, 4, d=2)
        assert r.passed

    def test_xx_model_is_reflection_symmetric(self, xx4):
        r = check_reflection_symmetry(xx4, 4, d=2)
        assert r.passed

    def test_property_name(self, tfim4):
        r = check_reflection_symmetry(tfim4, 4)
        assert r.property_name == "reflection_symmetry"

    def test_defect_tiny_for_symmetric_model(self, tfim4):
        r = check_reflection_symmetry(tfim4, 4)
        assert r.defect < 1e-10

    def test_asymmetric_ham_fails(self):
        # Asymmetric on-site field: add a local Z only on site 0
        H = transverse_ising_ham(4, J=1.0, h=0.5)
        Z = np.diag([1.0, -1.0])
        np.random.default_rng(0)
        # Add random asymmetric on-site perturbation on site 0 only
        H_asym = H.copy()
        field = 0.5 * np.kron(Z, np.eye(8))
        H_asym += field
        r = check_reflection_symmetry(H_asym, 4, d=2)
        assert not r.passed

    def test_notes_contain_n_sites(self, tfim4):
        r = check_reflection_symmetry(tfim4, 4)
        assert "4" in r.notes

    def test_2_site_system(self):
        H = transverse_ising_ham(2, J=1.0, h=0.5)
        r = check_reflection_symmetry(H, 2)
        assert r.passed

    def test_returns_structure_report(self, tfim4):
        from htf.structure import StructureReport
        r = check_reflection_symmetry(tfim4, 4)
        assert isinstance(r, StructureReport)

    def test_3_site_tfim(self):
        H = transverse_ising_ham(4, J=1.0, h=1.0)
        r = check_reflection_symmetry(H, 4, d=2)
        assert r.passed


# ─────────────────── TestOsPositivityReport (now finite_lattice_reflection_diagnostics) ──

class TestOsPositivityReport:

    def test_tfim_all_passed(self, tfim4):
        rep = finite_lattice_reflection_diagnostics(tfim4, 4)
        assert rep["all_passed"] is True

    def test_xx_all_passed(self, xx4):
        rep = finite_lattice_reflection_diagnostics(xx4, 4)
        assert rep["all_passed"] is True

    def test_has_required_keys(self, tfim4):
        rep = finite_lattice_reflection_diagnostics(tfim4, 4)
        for key in (
            "transfer_positivity", "reflection_symmetry",
            "os_gram_positivity", "all_passed", "notes",
        ):
            assert key in rep

    def test_notes_is_string_with_out(self, tfim4):
        rep = finite_lattice_reflection_diagnostics(tfim4, 4)
        assert isinstance(rep["notes"], str)
        assert "[OUT]" in rep["notes"]

    def test_transfer_positivity_is_structure_report(self, tfim4):
        from htf.structure import StructureReport
        rep = finite_lattice_reflection_diagnostics(tfim4, 4)
        assert isinstance(rep["transfer_positivity"], StructureReport)

    def test_os_gram_property_name(self, tfim4):
        rep = finite_lattice_reflection_diagnostics(tfim4, 4)
        assert rep["os_gram_positivity"].property_name == "os_gram_positivity"

    def test_different_beta_still_passes(self, tfim4):
        rep = finite_lattice_reflection_diagnostics(tfim4, 4, beta=0.5)
        assert rep["all_passed"] is True

    def test_beta_2_still_passes(self, tfim4):
        rep = finite_lattice_reflection_diagnostics(tfim4, 4, beta=2.0)
        assert rep["all_passed"] is True

    def test_2_site_system(self):
        H = transverse_ising_ham(2, J=1.0, h=1.0)
        rep = finite_lattice_reflection_diagnostics(H, 2)
        assert rep["all_passed"] is True

    def test_gram_defect_small(self, tfim4):
        rep = finite_lattice_reflection_diagnostics(tfim4, 4)
        assert rep["os_gram_positivity"].defect < 1e-10

    def test_all_reports_passed_attr(self, tfim4):
        rep = finite_lattice_reflection_diagnostics(tfim4, 4)
        assert rep["transfer_positivity"].passed
        assert rep["reflection_symmetry"].passed
        assert rep["os_gram_positivity"].passed

    def test_asymmetric_ham_fails_reflection_check(self):
        H = transverse_ising_ham(4, J=1.0, h=0.5)
        Z = np.diag([1.0, -1.0])
        H_asym = H + 0.5 * np.kron(Z, np.eye(8))
        rep = finite_lattice_reflection_diagnostics(H_asym, 4)
        assert not rep["reflection_symmetry"].passed
        assert not rep["all_passed"]

    def test_deprecated_os_positivity_report_emits_warning(self, tfim4):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            os_positivity_report(tfim4, 4)
        assert any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_p0_5_regression_transfer_positivity_always_passes(self, tfim4):
        # Regression P0-5: transfer_positivity passes for ALL real symmetric H
        # by construction — eigenvalues of exp(-βH) are always positive.
        rep = finite_lattice_reflection_diagnostics(tfim4, 4)
        assert rep["transfer_positivity"].passed
        rep2 = finite_lattice_reflection_diagnostics(
            np.diag([1.0, 2.0, 3.0, 4.0]), n_sites=2
        )
        assert rep2["transfer_positivity"].passed
