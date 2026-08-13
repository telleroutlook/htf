"""Tests for htf/adapters/tenpy_adapter.py — TeNPy MPS → RayleighCertificate.

All tests use lightweight mock objects that implement the TeNPy duck-typing
interface (``get_theta()`` + ``.L`` + ``.bc``).  TeNPy itself is *not* required.

Mock theta shapes follow the TeNPy convention for finite single-physical-leg MPS:
    get_theta(0, L) → shape (1, d_0, d_1, …, d_{L-1}, 1)  [vL, p0, …, p_{L-1}, vR]
The C-order ravel of this (after squeezing boundary legs) gives the state vector.
"""
import math

import numpy as np
import pytest

from htf.adapters.tenpy_adapter import _extract_tenpy_state_vector, rayleigh_from_tenpy_mps
from htf.rayleigh_cert import RayleighCertificate


# ─────────────────── mock helpers ─────────────────────────────────────────────

class _MockTNArray:
    """Minimal duck-type TeNPy Array: to_ndarray() returns a numpy array."""

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def to_ndarray(self):
        return self._arr


def _theta_from_psi(psi, d=2):
    """Build a (1, d_0, d_1, …, d_{L-1}, 1) theta from a flat state vector.

    Assumes all sites have the same local dimension d.
    L is inferred as log_d(len(psi)).
    """
    n = len(psi)
    L = round(math.log(n, d))
    assert d ** L == n, f"len(psi)={n} is not d^L for d={d}"
    shape = (1,) + (d,) * L + (1,)
    return np.asarray(psi).reshape(shape)


class _MockTeNPyMPS:
    """Finite MPS mock with correct theta shape (1, d_0, …, d_{L-1}, 1)."""

    bc = "finite"

    def __init__(self, psi, d=2):
        self._psi = np.asarray(psi, dtype=float)
        n = len(self._psi)
        L = round(math.log(n, d))
        assert d ** L == n
        self.L = L
        self._d = d

    def get_theta(self, i, n):
        return _MockTNArray(_theta_from_psi(self._psi, self._d))


class _MockTeNPyMPSNumpyTheta:
    """get_theta() returns a plain numpy array (no to_ndarray())."""

    bc = "finite"

    def __init__(self, psi, d=2):
        self._psi = np.asarray(psi, dtype=float)
        n = len(self._psi)
        L = round(math.log(n, d))
        assert d ** L == n
        self.L = L
        self._d = d

    def get_theta(self, i, n):
        return _theta_from_psi(self._psi, self._d)


class _MockTeNPyMPSComplex:
    """get_theta() returns a complex theta array."""

    bc = "finite"

    def __init__(self, psi_complex, d=2):
        self._psi = np.asarray(psi_complex, dtype=complex)
        n = len(self._psi)
        L = round(math.log(n, d))
        assert d ** L == n
        self.L = L
        self._d = d

    def get_theta(self, i, n):
        return _MockTNArray(_theta_from_psi(self._psi, self._d))


class _MockTeNPyMPSInfinite:
    """bc='infinite' — must be rejected."""
    bc = "infinite"
    L = 2

    def get_theta(self, i, n):  # pragma: no cover
        return np.eye(4).reshape(1, 2, 2, 1)


class _MockTeNPyMPSSegment:
    """bc='segment' — must be rejected."""
    bc = "segment"
    L = 2

    def get_theta(self, i, n):  # pragma: no cover
        return np.eye(4).reshape(1, 2, 2, 1)


class _ToDenseMPS:
    """Fallback duck-type: has to_dense() but not get_theta.  bc='finite'."""

    bc = "finite"

    def __init__(self, psi):
        self._psi = np.asarray(psi, dtype=float)

    def to_dense(self):
        return self._psi


class _NeitherInterfaceMPS:
    """Has neither get_theta nor to_dense."""
    pass


# ─────────────────── _extract_tenpy_state_vector ──────────────────────────────

class TestExtractTeNPyStateVector:

    def test_tenpy_interface_real(self):
        psi = np.array([0.6, 0.0, 0.0, 0.8])  # 2-site, d=2
        mps = _MockTeNPyMPS(psi)
        out = _extract_tenpy_state_vector(mps)
        np.testing.assert_allclose(out, psi)

    def test_returns_float64_for_real(self):
        mps = _MockTeNPyMPS(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        out = _extract_tenpy_state_vector(mps)
        assert out.dtype == np.float64

    def test_plain_numpy_theta_accepted(self):
        psi = np.array([1.0, 0.0, 0.0, 0.0])
        mps = _MockTeNPyMPSNumpyTheta(psi)
        out = _extract_tenpy_state_vector(mps)
        np.testing.assert_allclose(out, psi)

    def test_fallback_to_dense(self):
        psi = np.array([0.5, 0.5, 0.5, 0.5])
        mps = _ToDenseMPS(psi)
        out = _extract_tenpy_state_vector(mps)
        np.testing.assert_allclose(out, psi)

    def test_complex_array_preserved_as_complex128(self):
        psi = np.array([1.0 + 0.5j, 0.0, 0.0, 0.3j])
        mps = _MockTeNPyMPSComplex(psi)
        out = _extract_tenpy_state_vector(mps)
        assert out.dtype == np.complex128
        np.testing.assert_allclose(out, psi)

    def test_small_imaginary_preserved_not_projected(self):
        psi = np.array([1.0 + 1e-15j, 0.0, 0.0, 0.0])
        mps = _MockTeNPyMPSComplex(psi)
        out = _extract_tenpy_state_vector(mps)
        assert np.iscomplexobj(out)
        assert abs(out[0].imag) > 0

    def test_bc_infinite_raises(self):
        with pytest.raises(ValueError, match="bc='finite'"):
            _extract_tenpy_state_vector(_MockTeNPyMPSInfinite())

    def test_bc_segment_raises(self):
        with pytest.raises(ValueError, match="bc='finite'"):
            _extract_tenpy_state_vector(_MockTeNPyMPSSegment())

    def test_neither_interface_raises_type_error(self):
        with pytest.raises(TypeError, match="get_theta"):
            _extract_tenpy_state_vector(_NeitherInterfaceMPS())

    def test_4site_state_ravelled(self):
        # 4-site d=2: state vector of length 16
        psi = np.zeros(16)
        psi[0] = 1.0
        mps = _MockTeNPyMPS(psi)
        out = _extract_tenpy_state_vector(mps)
        assert out.ndim == 1
        assert len(out) == 16

    def test_theta_c_order_ravel(self):
        # 2-site d=2: ψ[i*2 + j] = 10*i + j  → [0,1,10,11]
        psi = np.array([0.0, 1.0, 10.0, 11.0])
        mps = _MockTeNPyMPS(psi)
        out = _extract_tenpy_state_vector(mps)
        np.testing.assert_allclose(out, psi)


# ─────────────────── Pauli-Y counterexample (R2 correctness check) ────────────

class TestPauliYCounterexample:
    """Referee R2 counterexample: ψ with tiny imaginary part near Pauli-Y H.

    Old code: max|Im ψ| = 1e-12 < imag_tol → silently takes Re ψ.
      R_H(Re ψ) = 0.0  (WRONG)
    New code: preserves complex ψ.
      R_H(ψ) = 2.0  (CORRECT)
    """

    def setup_method(self):
        self.Y = np.array([[0, -1j], [1j, 0]], dtype=complex)  # Pauli-Y (Hermitian)
        self.H = 1e12 * self.Y
        self.psi = np.array([1.0, 1e-12 * 1j], dtype=complex)

    def test_rayleigh_quotient_uses_full_complex_state(self):
        mps = _MockTeNPyMPSComplex(self.psi)
        cert = rayleigh_from_tenpy_mps(mps, self.H)
        assert cert.upper > 1.0, (
            f"cert.upper={cert.upper:.6g} — adapter appears to project to Re ψ "
            "(should use full complex state, R_H(ψ)=2.0)"
        )

    def test_projected_state_would_give_zero(self):
        re_psi = np.array([1.0, 0.0])
        mps = _MockTeNPyMPS(re_psi)
        cert = rayleigh_from_tenpy_mps(mps, self.H)
        assert abs(cert.upper) < 1e-6


# ─────────────────── rayleigh_from_tenpy_mps: happy path ──────────────────────

class TestRayleighFromTeNPyMPS:

    @pytest.fixture
    def H4(self):
        return np.diag([0.0, 1.0, 1.0, 2.0])  # 2-site d=2 number operator

    def test_returns_rayleigh_certificate(self, H4):
        mps = _MockTeNPyMPS([1.0, 0.0, 0.0, 0.0])
        cert = rayleigh_from_tenpy_mps(mps, H4)
        assert isinstance(cert, RayleighCertificate)

    def test_upper_bounds_E0(self, H4):
        mps = _MockTeNPyMPS([1.0, 0.0, 0.0, 0.0])
        cert = rayleigh_from_tenpy_mps(mps, H4)
        assert cert.upper >= 0.0 - 1e-12

    def test_exact_gs_upper_near_E0(self, H4):
        mps = _MockTeNPyMPS([1.0, 0.0, 0.0, 0.0])
        cert = rayleigh_from_tenpy_mps(mps, H4)
        assert cert.upper <= 0.0 + 1e-9

    def test_verified_false_on_output(self, H4):
        cert = rayleigh_from_tenpy_mps(_MockTeNPyMPS([1.0, 0.0, 0.0, 0.0]), H4)
        assert cert.verified is False

    def test_notes_include_adapter_label(self, H4):
        cert = rayleigh_from_tenpy_mps(_MockTeNPyMPS([1.0, 0.0, 0.0, 0.0]), H4, notes="my_run")
        assert "tenpy-adapter" in cert.notes
        assert "my_run" in cert.notes

    def test_notes_empty_string(self, H4):
        cert = rayleigh_from_tenpy_mps(_MockTeNPyMPS([1.0, 0.0, 0.0, 0.0]), H4)
        assert "tenpy-adapter" in cert.notes

    def test_notes_include_h_source_and_basis_policy(self, H4):
        cert = rayleigh_from_tenpy_mps(_MockTeNPyMPS([1.0, 0.0, 0.0, 0.0]), H4)
        assert "H_source=caller" in cert.notes
        assert "undo_sort_charge" in cert.notes

    def test_unnormalised_state_accepted(self, H4):
        mps = _MockTeNPyMPS([5.0, 0.0, 0.0, 0.0])
        cert = rayleigh_from_tenpy_mps(mps, H4)
        assert cert.upper <= 0.0 + 1e-9

    def test_fallback_to_dense_accepted(self):
        H = np.diag([0.0, 1.0])
        mps = _ToDenseMPS([1.0, 0.0])
        cert = rayleigh_from_tenpy_mps(mps, H)
        assert isinstance(cert, RayleighCertificate)
        assert cert.upper >= 0.0 - 1e-12

    def test_complex_state_accepted(self, H4):
        psi = np.array([1.0 + 0.0j, 0.0, 0.0, 0.0])
        mps = _MockTeNPyMPSComplex(psi)
        cert = rayleigh_from_tenpy_mps(mps, H4)
        assert isinstance(cert, RayleighCertificate)
        assert cert.upper <= 0.0 + 1e-9

    def test_scaling_invariance(self, H4):
        """R_H(αψ) == R_H(ψ) for any nonzero real α."""
        psi = np.array([1.0, 1.0, 0.0, 0.0]) / math.sqrt(2)
        cert1 = rayleigh_from_tenpy_mps(_MockTeNPyMPS(psi), H4)
        cert2 = rayleigh_from_tenpy_mps(_MockTeNPyMPS(3.7 * psi), H4)
        assert abs(cert1.upper - cert2.upper) < 1e-10

    def test_tfim_n4_end_to_end(self):
        from htf.variational import transverse_ising_ham
        H = transverse_ising_ham(4, J=1.0, h=0.5)
        true_E0 = float(np.linalg.eigvalsh(H)[0])
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(H.shape[0])
        mps = _MockTeNPyMPS(psi, d=2)
        cert = rayleigh_from_tenpy_mps(mps, H, notes="tfim_test")
        assert cert.upper >= true_E0 - 1e-9

    def test_input_digest_present(self, H4):
        cert = rayleigh_from_tenpy_mps(_MockTeNPyMPS([1.0, 0.0, 0.0, 0.0]), H4)
        assert len(cert.input_digest) == 64
        int(cert.input_digest, 16)

    def test_can_be_independently_verified(self, H4):
        from htf.rayleigh_cert import verify_rayleigh_certificate
        cert = rayleigh_from_tenpy_mps(_MockTeNPyMPS([1.0, 0.0, 0.0, 0.0]), H4)
        verify_rayleigh_certificate(cert)
        assert cert.verified is True

    def test_plain_numpy_theta_end_to_end(self, H4):
        mps = _MockTeNPyMPSNumpyTheta([1.0, 0.0, 0.0, 0.0])
        cert = rayleigh_from_tenpy_mps(mps, H4)
        assert isinstance(cert, RayleighCertificate)

    def test_schema_version_correct(self, H4):
        cert = rayleigh_from_tenpy_mps(_MockTeNPyMPS([1.0, 0.0, 0.0, 0.0]), H4)
        assert cert.schema_version == "rayleigh-cert/v2"

    def test_basis_permutation_changes_rayleigh(self, H4):
        """Dimension match alone cannot detect H vs ψ basis mismatch."""
        psi = np.array([1.0, 0.0, 0.0, 0.0])        # |00⟩
        H_perm = np.diag([2.0, 1.0, 1.0, 0.0])       # reversed eigenvalues
        cert_correct = rayleigh_from_tenpy_mps(_MockTeNPyMPS(psi), H4)
        cert_perm    = rayleigh_from_tenpy_mps(_MockTeNPyMPS(psi), H_perm)
        assert cert_correct.upper <= 0.0 + 1e-9      # R_H4(|00⟩) = 0
        assert cert_perm.upper >= 1.9                 # R_H_perm(|00⟩) = 2 — mismatch silently accepted


# ─────────────────── rayleigh_from_tenpy_mps: error cases ─────────────────────

class TestRayleighFromTeNPyMPSErrors:

    def test_neither_interface_raises_type_error(self):
        H = np.diag([0.0, 1.0])
        with pytest.raises(TypeError, match="get_theta"):
            rayleigh_from_tenpy_mps(_NeitherInterfaceMPS(), H)

    def test_bc_infinite_raises(self):
        H = np.diag([0.0, 1.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="bc='finite'"):
            rayleigh_from_tenpy_mps(_MockTeNPyMPSInfinite(), H)

    def test_bc_segment_raises(self):
        H = np.diag([0.0, 1.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="bc='finite'"):
            rayleigh_from_tenpy_mps(_MockTeNPyMPSSegment(), H)

    def test_dim_mismatch_raises(self):
        H = np.diag([0.0, 1.0, 2.0])            # 3×3
        mps = _MockTeNPyMPS([1.0, 0.0, 0.0, 0.0])  # 4-component state
        with pytest.raises(ValueError):
            rayleigh_from_tenpy_mps(mps, H)

    def test_asymmetric_H_raises(self):
        H = np.array([[0.0, 1.0], [2.0, 0.0]])  # not symmetric
        mps = _MockTeNPyMPS([1.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="symmetric"):
            # H is 2×2 but psi is length 4 → dim mismatch first; use matching dim
            H2 = np.array([[0.0, 1.0, 0.0, 0.0],
                           [2.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0]])
            rayleigh_from_tenpy_mps(mps, H2)

    def test_zero_norm_psi_raises(self):
        H = np.diag([0.0, 1.0, 1.0, 2.0])
        mps = _MockTeNPyMPS([0.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="zero norm"):
            rayleigh_from_tenpy_mps(mps, H)

    def test_complex_H_hermitian_accepted(self):
        H = np.eye(4, dtype=complex)
        mps = _MockTeNPyMPS([1.0, 0.0, 0.0, 0.0])
        cert = rayleigh_from_tenpy_mps(mps, H)
        assert cert.upper <= 1.0 + 1e-9


# ─────────────────── JSON roundtrip ───────────────────────────────────────────

class TestTeNPyAdapterJSON:

    def test_to_full_json_roundtrip(self):
        import json
        from htf.verify import verify_from_dict
        H = np.diag([0.0, 1.0, 1.0, 2.0])
        cert = rayleigh_from_tenpy_mps(_MockTeNPyMPS([1.0, 0.0, 0.0, 0.0]), H)
        full = json.loads(cert.to_full_json())
        result = verify_from_dict(full)
        assert result["verified"] is True

    def test_to_json_has_adapter_note(self):
        import json
        H = np.diag([0.0, 1.0, 1.0, 2.0])
        cert = rayleigh_from_tenpy_mps(_MockTeNPyMPS([1.0, 0.0, 0.0, 0.0]), H, notes="run1")
        d = json.loads(cert.to_json())
        assert "tenpy-adapter" in d["notes"]

    def test_to_dict_schema_valid(self):
        from htf.rayleigh_cert import validate_certificate_dict
        H = np.diag([0.0, 1.0, 1.0, 2.0])
        cert = rayleigh_from_tenpy_mps(_MockTeNPyMPS([1.0, 0.0, 0.0, 0.0]), H)
        validate_certificate_dict(cert.to_dict())
