"""Tests for htf/structure.py — structure verification for proof-carrying diagrams."""
import numpy as np
import pytest

from htf import Box, TensorFunctor, Wire
from htf.structure import (
    StructureReport,
    check_box_isometry,
    check_box_unitary,
    check_isometry,
    check_reflection_positivity,
    check_unitary,
    enforce_isometry,
    enforce_unitary,
    gram_min_eig,
    isometry_defect,
    unitary_defect,
)

# ──────────────────────── helpers ────────────────────────────────────

def _random_isometry(rows, cols, seed=0):
    """Return a (rows, cols) matrix M with M @ M.T = I (rows <= cols)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((rows, cols))
    U, _, Vt = np.linalg.svd(A, full_matrices=False)
    return U @ Vt


def _random_unitary(n, seed=1):
    """Return a (n, n) unitary matrix."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    U, _, Vt = np.linalg.svd(A, full_matrices=True)
    return U @ Vt


# ──────────────────────── StructureReport ────────────────────────────

class TestStructureReport:
    def test_fields_pass(self):
        r = StructureReport(
            property_name="isometry",
            passed=True,
            defect=1e-15,
            tolerance=1e-10,
        )
        assert r.property_name == "isometry"
        assert r.passed is True
        assert r.defect == pytest.approx(1e-15)
        assert r.tolerance == pytest.approx(1e-10)
        assert r.notes == ""

    def test_fields_fail(self):
        r = StructureReport(
            property_name="unitary",
            passed=False,
            defect=0.5,
            tolerance=1e-10,
            notes="something wrong",
        )
        assert r.passed is False
        assert r.notes == "something wrong"

    def test_str_pass_no_notes(self):
        r = StructureReport("isometry", True, 1e-15, 1e-10)
        s = str(r)
        assert s.startswith("[PASS]")
        assert "isometry" in s
        assert "defect=" in s
        assert "tol=" in s
        assert "—" not in s

    def test_str_fail_no_notes(self):
        r = StructureReport("unitary", False, 0.5, 1e-10)
        s = str(r)
        assert s.startswith("[FAIL]")
        assert "unitary" in s

    def test_str_with_notes(self):
        r = StructureReport("isometry", True, 0.0, 1e-10, notes="box='U'")
        s = str(r)
        assert "—" in s
        assert "box='U'" in s

    def test_str_fail_with_notes(self):
        r = StructureReport("reflection_positivity", False, 0.3, 0.0, notes="min_eig=-3.000e-01")
        s = str(r)
        assert "[FAIL]" in s
        assert "min_eig" in s


# ──────────────────────── isometry_defect ────────────────────────────

class TestIsometryDefect:
    def test_perfect_isometry_2d(self):
        M = _random_isometry(3, 6, seed=42)
        d = isometry_defect(M, n_cod=1)
        assert d == pytest.approx(0.0, abs=1e-12)

    def test_perfect_isometry_square_identity(self):
        M = np.eye(4)
        assert isometry_defect(M, n_cod=1) == pytest.approx(0.0, abs=1e-14)

    def test_non_isometric_random(self):
        rng = np.random.default_rng(7)
        M = rng.standard_normal((3, 6))
        d = isometry_defect(M, n_cod=1)
        assert d > 0.01

    def test_scalar_tensor(self):
        # 0-D tensor: both cod and dom are scalar (size 1)
        t = np.array(1.0)
        d = isometry_defect(t)
        assert d == pytest.approx(0.0, abs=1e-14)

    def test_n_cod_none_splits_axes(self):
        # 4-D tensor (2,3,2,3): n_cod=None -> split at 2, matrix (6,6)
        M_mat = _random_unitary(6, seed=3)
        tensor = M_mat.reshape(2, 3, 2, 3)
        d = isometry_defect(tensor)  # n_cod=2 by default
        assert d == pytest.approx(0.0, abs=1e-11)

    def test_n_cod_explicit(self):
        # 3-D tensor (2,3,4): n_cod=1 -> matrix (2,12)
        M = _random_isometry(2, 12, seed=9)
        tensor = M.reshape(2, 3, 4)
        d = isometry_defect(tensor, n_cod=1)
        assert d == pytest.approx(0.0, abs=1e-11)


# ──────────────────────── unitary_defect ─────────────────────────────

class TestUnitaryDefect:
    def test_perfect_unitary(self):
        U = _random_unitary(5, seed=10)
        d = unitary_defect(U, n_cod=1)
        assert d == pytest.approx(0.0, abs=1e-12)

    def test_identity_is_unitary(self):
        assert unitary_defect(np.eye(3), n_cod=1) == pytest.approx(0.0, abs=1e-14)

    def test_non_unitary_random(self):
        rng = np.random.default_rng(11)
        M = rng.standard_normal((4, 4))
        d = unitary_defect(M, n_cod=1)
        assert d > 0.01

    def test_rectangular_is_not_unitary(self):
        # A (3,6) isometry satisfies M @ M.T = I but not M.T @ M = I
        M = _random_isometry(3, 6, seed=13)
        d = unitary_defect(M, n_cod=1)
        assert d > 0.01

    def test_n_cod_none_square(self):
        U = _random_unitary(6, seed=14)
        tensor = U.reshape(2, 3, 2, 3)
        d = unitary_defect(tensor)
        assert d == pytest.approx(0.0, abs=1e-11)


# ──────────────────────── check_isometry ─────────────────────────────

class TestCheckIsometry:
    def test_returns_structurereport(self):
        M = np.eye(3)
        r = check_isometry(M, n_cod=1)
        assert isinstance(r, StructureReport)

    def test_pass_for_perfect_isometry(self):
        M = _random_isometry(4, 8, seed=20)
        r = check_isometry(M, tol=1e-10, n_cod=1)
        assert r.passed is True
        assert r.property_name == "isometry"
        assert r.defect == pytest.approx(0.0, abs=1e-11)

    def test_fail_for_random_matrix(self):
        rng = np.random.default_rng(21)
        M = rng.standard_normal((4, 8))
        r = check_isometry(M, tol=1e-10, n_cod=1)
        assert r.passed is False
        assert r.defect > 0.01

    def test_tolerance_boundary(self):
        M = _random_isometry(3, 6, seed=22)
        check_isometry(M, tol=1e-15, n_cod=1)
        r_loose = check_isometry(M, tol=1.0, n_cod=1)
        # With very tight tol it may fail due to floating-point noise;
        # with loose tol it must pass.
        assert r_loose.passed is True

    def test_square_identity_passes(self):
        r = check_isometry(np.eye(5), tol=1e-10, n_cod=1)
        assert r.passed is True

    def test_notes_empty_by_default(self):
        r = check_isometry(np.eye(2), n_cod=1)
        assert r.notes == ""


# ──────────────────────── check_unitary ──────────────────────────────

class TestCheckUnitary:
    def test_returns_structurereport(self):
        r = check_unitary(np.eye(3), n_cod=1)
        assert isinstance(r, StructureReport)

    def test_pass_for_perfect_unitary(self):
        U = _random_unitary(4, seed=30)
        r = check_unitary(U, tol=1e-10, n_cod=1)
        assert r.passed is True
        assert r.property_name == "unitary"

    def test_fail_for_random_matrix(self):
        rng = np.random.default_rng(31)
        M = rng.standard_normal((4, 4))
        r = check_unitary(M, tol=1e-10, n_cod=1)
        assert r.passed is False

    def test_fail_for_rectangular(self):
        M = _random_isometry(3, 6, seed=32)
        r = check_unitary(M, tol=1e-10, n_cod=1)
        assert r.passed is False

    def test_identity_passes(self):
        r = check_unitary(np.eye(4), tol=1e-10, n_cod=1)
        assert r.passed is True

    def test_notes_empty_by_default(self):
        r = check_unitary(np.eye(2), n_cod=1)
        assert r.notes == ""


# ──────────────────────── check_box_isometry ─────────────────────────

class TestCheckBoxIsometry:
    def _iso_box_and_functor(self, seed=40):
        """Box with cod=(q,) dom=(r,), dims 3 and 9 -> (3,9) isometric tensor."""
        q = Wire("q", 3)
        r = Wire("r", 9)
        box = Box("iso_box", (r,), (q,))  # cod=(q,), dom=(r,) -> tensor shape (3,9)
        M = _random_isometry(3, 9, seed=seed)
        F = TensorFunctor({"iso_box": M})
        return box, F

    def test_isometric_box_passes(self):
        box, F = self._iso_box_and_functor()
        r = check_box_isometry(box, F, tol=1e-10)
        assert r.passed is True
        assert r.property_name == "isometry"

    def test_non_isometric_box_fails(self):
        q = Wire("q", 3)
        r = Wire("r", 9)
        box = Box("bad_box", (r,), (q,))
        rng = np.random.default_rng(41)
        M = rng.standard_normal((3, 9))
        F = TensorFunctor({"bad_box": M})
        report = check_box_isometry(box, F, tol=1e-10)
        assert report.passed is False

    def test_notes_contain_box_name(self):
        box, F = self._iso_box_and_functor()
        r = check_box_isometry(box, F)
        assert "iso_box" in r.notes

    def test_notes_contain_dims(self):
        box, F = self._iso_box_and_functor()
        r = check_box_isometry(box, F)
        assert "cod=" in r.notes
        assert "dom=" in r.notes

    def test_square_unitary_box_also_passes_isometry(self):
        a = Wire("a", 4)
        box = Box("U_sq", (a,), (a,))
        U = _random_unitary(4, seed=42)
        F = TensorFunctor({"U_sq": U})
        r = check_box_isometry(box, F, tol=1e-10)
        assert r.passed is True

    def test_custom_tolerance(self):
        q = Wire("q", 2)
        r = Wire("r", 4)
        box = Box("slightly_off", (r,), (q,))
        M = _random_isometry(2, 4, seed=43)
        # Add small noise
        rng = np.random.default_rng(43)
        M_noisy = M + 1e-6 * rng.standard_normal(M.shape)
        F = TensorFunctor({"slightly_off": M_noisy})
        r_strict = check_box_isometry(box, F, tol=1e-10)
        r_loose = check_box_isometry(box, F, tol=1e-4)
        assert r_strict.passed is False
        assert r_loose.passed is True


# ──────────────────────── check_box_unitary ──────────────────────────

class TestCheckBoxUnitary:
    def _unitary_box_and_functor(self, seed=50):
        a = Wire("a", 4)
        box = Box("U_box", (a,), (a,))
        U = _random_unitary(4, seed=seed)
        F = TensorFunctor({"U_box": U})
        return box, F

    def test_unitary_box_passes(self):
        box, F = self._unitary_box_and_functor()
        r = check_box_unitary(box, F, tol=1e-10)
        assert r.passed is True
        assert r.property_name == "unitary"

    def test_non_unitary_box_fails(self):
        a = Wire("a", 4)
        box = Box("bad_U", (a,), (a,))
        rng = np.random.default_rng(51)
        M = rng.standard_normal((4, 4))
        F = TensorFunctor({"bad_U": M})
        r = check_box_unitary(box, F, tol=1e-10)
        assert r.passed is False

    def test_notes_contain_box_name(self):
        box, F = self._unitary_box_and_functor()
        r = check_box_unitary(box, F)
        assert "U_box" in r.notes

    def test_notes_contain_dims(self):
        box, F = self._unitary_box_and_functor()
        r = check_box_unitary(box, F)
        assert "cod=" in r.notes
        assert "dom=" in r.notes

    def test_rectangular_box_fails_unitary(self):
        q = Wire("q", 3)
        r = Wire("r", 9)
        box = Box("rect", (r,), (q,))
        M = _random_isometry(3, 9, seed=52)
        F = TensorFunctor({"rect": M})
        r_report = check_box_unitary(box, F, tol=1e-10)
        assert r_report.passed is False

    def test_identity_box_passes(self):
        a = Wire("a", 3)
        box = Box("Id3", (a,), (a,))
        F = TensorFunctor({"Id3": np.eye(3)})
        r = check_box_unitary(box, F, tol=1e-10)
        assert r.passed is True


# ──────────────────────── gram_min_eig ───────────────────────────────

class TestGramMinEig:
    def test_identity_min_eig_is_one(self):
        assert gram_min_eig(np.eye(4)) == pytest.approx(1.0, abs=1e-14)

    def test_positive_definite(self):
        rng = np.random.default_rng(60)
        A = rng.standard_normal((5, 5))
        G = A @ A.T + np.eye(5)  # strictly positive definite
        assert gram_min_eig(G) > 0.9

    def test_semidefinite_zero_eig(self):
        # Rank-deficient: zero min eigenvalue
        rng = np.random.default_rng(61)
        A = rng.standard_normal((5, 3))
        G = A @ A.T  # PSD, rank 3 -> 2 zero eigenvalues
        min_ev = gram_min_eig(G)
        assert min_ev == pytest.approx(0.0, abs=1e-10)

    def test_indefinite_negative_eig(self):
        G = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigs: 3 and -1
        assert gram_min_eig(G) == pytest.approx(-1.0, abs=1e-14)

    def test_scalar(self):
        assert gram_min_eig(np.array([[3.0]])) == pytest.approx(3.0)


# ──────────────────────── check_reflection_positivity ────────────────

class TestCheckReflectionPositivity:
    def test_psd_passes_with_zero_tol(self):
        A = np.eye(4)
        r = check_reflection_positivity(A, tol=0.0)
        assert r.passed is True
        assert r.property_name == "reflection_positivity"
        assert r.defect == pytest.approx(0.0, abs=1e-14)

    def test_strictly_positive_definite_passes(self):
        rng = np.random.default_rng(70)
        A = rng.standard_normal((4, 4))
        G = A @ A.T + np.eye(4)
        r = check_reflection_positivity(G)
        assert r.passed is True

    def test_indefinite_fails(self):
        G = np.array([[1.0, 2.0], [2.0, 1.0]])  # min_eig = -1
        r = check_reflection_positivity(G, tol=0.0)
        assert r.passed is False
        assert r.defect == pytest.approx(1.0, abs=1e-14)

    def test_tol_allows_small_negative_eig(self):
        G = np.array([[1.0, 2.0], [2.0, 1.0]])  # min_eig = -1
        r_strict = check_reflection_positivity(G, tol=0.5)
        r_loose = check_reflection_positivity(G, tol=1.5)
        assert r_strict.passed is False
        assert r_loose.passed is True

    def test_notes_contain_min_eig(self):
        G = np.eye(3)
        r = check_reflection_positivity(G)
        assert "min_eig=" in r.notes

    def test_defect_zero_for_psd(self):
        rng = np.random.default_rng(71)
        A = rng.standard_normal((3, 5))
        G = A @ A.T  # PSD
        r = check_reflection_positivity(G, tol=0.0)
        assert r.defect == pytest.approx(0.0, abs=1e-10)

    def test_defect_matches_negative_min_eig(self):
        G = np.array([[0.0, 0.0], [0.0, -0.5]])
        r = check_reflection_positivity(G, tol=0.0)
        assert r.defect == pytest.approx(0.5, abs=1e-14)

    def test_str_contains_pass_fail(self):
        G_pos = np.eye(2)
        G_neg = -np.eye(2)
        assert "[PASS]" in str(check_reflection_positivity(G_pos))
        assert "[FAIL]" in str(check_reflection_positivity(G_neg))


# ──────────────────────── enforce_isometry ───────────────────────────

class TestEnforceIsometry:
    def test_already_isometric_unchanged(self):
        M = _random_isometry(3, 6, seed=80)
        M_out = enforce_isometry(M, n_cod=1)
        assert M_out.shape == M.shape
        assert isometry_defect(M_out, n_cod=1) == pytest.approx(0.0, abs=1e-12)

    def test_random_matrix_becomes_isometric(self):
        rng = np.random.default_rng(81)
        M = rng.standard_normal((4, 8))
        M_out = enforce_isometry(M, n_cod=1)
        assert M_out.shape == M.shape
        assert isometry_defect(M_out, n_cod=1) == pytest.approx(0.0, abs=1e-11)

    def test_square_matrix_becomes_unitary(self):
        rng = np.random.default_rng(82)
        M = rng.standard_normal((4, 4))
        M_out = enforce_isometry(M, n_cod=1)
        assert isometry_defect(M_out, n_cod=1) == pytest.approx(0.0, abs=1e-11)

    def test_preserves_shape(self):
        rng = np.random.default_rng(83)
        tensor = rng.standard_normal((2, 3, 12))
        out = enforce_isometry(tensor, n_cod=1)
        assert out.shape == tensor.shape

    def test_error_when_cod_greater_than_dom(self):
        M = np.ones((6, 3))  # 6 rows > 3 cols -> invalid isometry
        with pytest.raises(ValueError, match="cod_size.*dom_size"):
            enforce_isometry(M, n_cod=1)

    def test_error_message_contains_sizes(self):
        M = np.ones((5, 2))
        with pytest.raises(ValueError) as exc_info:
            enforce_isometry(M, n_cod=1)
        assert "5" in str(exc_info.value)
        assert "2" in str(exc_info.value)

    def test_n_cod_none_default(self):
        # 4-D tensor (2,3,2,3): n_cod=2 by default -> matrix (6,6) -> square OK
        rng = np.random.default_rng(84)
        tensor = rng.standard_normal((2, 3, 2, 3))
        out = enforce_isometry(tensor)
        assert out.shape == tensor.shape
        assert isometry_defect(out) == pytest.approx(0.0, abs=1e-11)


# ──────────────────────── enforce_unitary ────────────────────────────

class TestEnforceUnitary:
    def test_already_unitary_unchanged(self):
        U = _random_unitary(4, seed=90)
        U_out = enforce_unitary(U, n_cod=1)
        assert U_out.shape == U.shape
        assert unitary_defect(U_out, n_cod=1) == pytest.approx(0.0, abs=1e-12)

    def test_random_square_becomes_unitary(self):
        rng = np.random.default_rng(91)
        M = rng.standard_normal((5, 5))
        M_out = enforce_unitary(M, n_cod=1)
        assert M_out.shape == M.shape
        assert unitary_defect(M_out, n_cod=1) == pytest.approx(0.0, abs=1e-11)

    def test_preserves_shape_square(self):
        rng = np.random.default_rng(92)
        tensor = rng.standard_normal((3, 3))
        out = enforce_unitary(tensor, n_cod=1)
        assert out.shape == tensor.shape

    def test_preserves_shape_4d(self):
        rng = np.random.default_rng(93)
        tensor = rng.standard_normal((2, 3, 2, 3))
        out = enforce_unitary(tensor)
        assert out.shape == tensor.shape
        assert unitary_defect(out) == pytest.approx(0.0, abs=1e-11)

    def test_error_when_not_square(self):
        M = np.ones((3, 6))  # rectangular -> not square
        with pytest.raises(ValueError, match="not square"):
            enforce_unitary(M, n_cod=1)

    def test_error_message_contains_dimensions(self):
        M = np.ones((3, 6))
        with pytest.raises(ValueError) as exc_info:
            enforce_unitary(M, n_cod=1)
        assert "3" in str(exc_info.value)
        assert "6" in str(exc_info.value)

    def test_n_cod_none_square_tensor(self):
        rng = np.random.default_rng(94)
        tensor = rng.standard_normal((4, 4))
        # n_cod=None -> ndim//2 = 1, matrix (4,4) -> square OK
        out = enforce_unitary(tensor)
        assert unitary_defect(out) == pytest.approx(0.0, abs=1e-11)

    def test_enforce_then_check_passes(self):
        rng = np.random.default_rng(95)
        M = rng.standard_normal((6, 6))
        M_unitary = enforce_unitary(M, n_cod=1)
        report = check_unitary(M_unitary, tol=1e-10, n_cod=1)
        assert report.passed is True

    def test_enforce_isometry_then_check_passes(self):
        rng = np.random.default_rng(96)
        M = rng.standard_normal((3, 9))
        M_iso = enforce_isometry(M, n_cod=1)
        report = check_isometry(M_iso, tol=1e-10, n_cod=1)
        assert report.passed is True
