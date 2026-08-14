"""Tests for htf/adapters/quimb_adapter.py — quimb MPS → RayleighCertificate.

All tests use a lightweight mock MPS that implements the quimb duck-typing
interface (``to_dense()`` → np.ndarray).  quimb itself is *not* required.
"""
import math

import numpy as np
import pytest

from htf.adapters.quimb_adapter import _extract_state_vector, rayleigh_from_quimb_mps
from htf.rayleigh_cert import RayleighCertificate

# ─────────────────── mock helpers ────────────────────────────────────────────

class _MockMPS:
    """Minimal duck-type quimb MPS: to_dense() returns a flat real array."""
    def __init__(self, psi):
        self._psi = np.asarray(psi, dtype=float)

    def to_dense(self):
        return self._psi


class _MockMPSComplex:
    """to_dense() returns a complex array."""
    def __init__(self, psi_complex):
        self._psi = np.asarray(psi_complex, dtype=complex)

    def to_dense(self):
        return self._psi


class _MockMPS2D:
    """to_dense() returns a (D,1) ket — quimb's current default shape."""
    def __init__(self, psi):
        self._psi = np.asarray(psi, dtype=float).reshape(-1, 1)

    def to_dense(self):
        return self._psi


class _NoDenseMPS:
    """An object that lacks to_dense entirely."""
    pass


# ─────────────────── _extract_state_vector ───────────────────────────────────

class TestExtractStateVector:

    def test_real_array_passthrough(self):
        psi = np.array([0.6, 0.8])
        mps = _MockMPS(psi)
        out = _extract_state_vector(mps)
        np.testing.assert_allclose(out, psi)

    def test_returns_float64_for_real(self):
        mps = _MockMPS(np.array([1.0, 0.0], dtype=np.float32))
        out = _extract_state_vector(mps)
        assert out.dtype == np.float64

    def test_ket_shape_ravelled(self):
        # quimb default: to_dense() returns (D, 1); adapter ravels to (D,)
        mps = _MockMPS2D(np.array([0.5, 0.5, 0.5, 0.5]))
        out = _extract_state_vector(mps)
        assert out.ndim == 1
        assert len(out) == 4

    def test_complex_array_preserved_as_complex128(self):
        psi = np.array([1.0 + 0.5j, 0.0 + 0.3j])
        mps = _MockMPSComplex(psi)
        out = _extract_state_vector(mps)
        assert out.dtype == np.complex128
        np.testing.assert_allclose(out, psi)

    def test_small_imaginary_preserved_not_projected(self):
        # Imaginary part below old imag_tol should NOT be silently dropped.
        psi = np.array([1.0 + 1e-15j, 0.0 + 0.0j])
        mps = _MockMPSComplex(psi)
        out = _extract_state_vector(mps)
        assert np.iscomplexobj(out)
        assert abs(out[0].imag) > 0  # imaginary part preserved

    def test_no_to_dense_raises_type_error(self):
        with pytest.raises(TypeError, match="to_dense"):
            _extract_state_vector(_NoDenseMPS())


# ─────────────────── Pauli-Y counterexample (R2 correctness check) ────────────

class TestPauliYCounterexample:
    """Referee R2 counterexample: ψ with tiny imaginary part near Pauli-Y H.

    Old code: max|Im ψ| = 1e-12 < imag_tol=1e-10 → silently takes Re ψ.
      R_H(Re ψ) = 0.0  (WRONG)
    New code: preserves complex ψ.
      R_H(ψ) = 2.0  (CORRECT)
    """

    def setup_method(self):
        self.Y = np.array([[0, -1j], [1j, 0]], dtype=complex)  # Pauli-Y (Hermitian)
        self.H = 1e12 * self.Y
        self.psi = np.array([1.0, 1e-12 * 1j], dtype=complex)

    def test_rayleigh_quotient_uses_full_complex_state(self):
        mps = _MockMPSComplex(self.psi)
        cert = rayleigh_from_quimb_mps(mps, self.H)
        # R_H(ψ) = 2.0; if Re ψ were used silently, upper ≈ 0.0
        assert cert.upper > 1.0, (
            f"cert.upper={cert.upper:.6g} — adapter appears to project to Re ψ "
            "(should use full complex state, R_H(ψ)=2.0)"
        )

    def test_projected_state_would_give_zero(self):
        re_psi = np.array([1.0, 0.0])
        mps = _MockMPS(re_psi)
        cert = rayleigh_from_quimb_mps(mps, self.H)
        assert abs(cert.upper) < 1e-6


# ─────────────────── rayleigh_from_quimb_mps: happy path ─────────────────────

class TestRayleighFromQuimbMPS:

    @pytest.fixture
    def diag2(self):
        return np.diag([0.0, 1.0])

    def test_returns_rayleigh_certificate(self, diag2):
        mps = _MockMPS([1.0, 0.0])
        cert = rayleigh_from_quimb_mps(mps, diag2)
        assert isinstance(cert, RayleighCertificate)

    def test_upper_bounds_E0(self, diag2):
        true_E0 = 0.0
        mps = _MockMPS([1.0, 0.0])
        cert = rayleigh_from_quimb_mps(mps, diag2)
        assert cert.upper >= true_E0 - 1e-12

    def test_exact_gs_upper_near_E0(self, diag2):
        _, evecs = np.linalg.eigh(diag2)
        mps = _MockMPS(evecs[:, 0])
        cert = rayleigh_from_quimb_mps(mps, diag2)
        assert cert.upper <= 0.0 + 1e-9

    def test_verified_false_on_output(self, diag2):
        cert = rayleigh_from_quimb_mps(_MockMPS([1.0, 0.0]), diag2)
        assert cert.verified is False

    def test_notes_include_adapter_label(self, diag2):
        cert = rayleigh_from_quimb_mps(_MockMPS([1.0, 0.0]), diag2, notes="my_run")
        assert "quimb-adapter" in cert.notes
        assert "my_run" in cert.notes

    def test_notes_empty_string(self, diag2):
        cert = rayleigh_from_quimb_mps(_MockMPS([1.0, 0.0]), diag2)
        assert "quimb-adapter" in cert.notes

    def test_notes_include_h_source(self, diag2):
        cert = rayleigh_from_quimb_mps(_MockMPS([1.0, 0.0]), diag2)
        assert "H_source=caller" in cert.notes

    def test_ket_shape_state_flattened(self):
        # (D,1) ket from quimb: H = diag(0,1,1,2), GS = |00⟩
        H = np.diag([0.0, 1.0, 1.0, 2.0])
        mps = _MockMPS2D(np.array([1.0, 0.0, 0.0, 0.0]))
        cert = rayleigh_from_quimb_mps(mps, H)
        assert cert.upper <= 0.0 + 1e-9

    def test_unnormalised_state_accepted(self, diag2):
        mps = _MockMPS([5.0, 0.0])  # unnormalised — Rayleigh quotient still 0
        cert = rayleigh_from_quimb_mps(mps, diag2)
        assert cert.upper <= 0.0 + 1e-9

    def test_complex_state_accepted(self, diag2):
        psi = np.array([1.0 + 0.0j, 0.0 + 0.0j])
        mps = _MockMPSComplex(psi)
        cert = rayleigh_from_quimb_mps(mps, diag2)
        assert isinstance(cert, RayleighCertificate)
        assert cert.upper <= 0.0 + 1e-9

    def test_scaling_invariance(self, diag2):
        """R_H(αψ) == R_H(ψ) for any nonzero scalar α."""
        psi = np.array([1.0, 1.0]) / math.sqrt(2)
        cert1 = rayleigh_from_quimb_mps(_MockMPS(psi), diag2)
        cert2 = rayleigh_from_quimb_mps(_MockMPS(3.7 * psi), diag2)
        # Both should certify the same Rayleigh quotient = 0.5
        assert abs(cert1.upper - cert2.upper) < 1e-10

    def test_tfim_n4_end_to_end(self):
        from htf.variational import transverse_ising_ham
        H = transverse_ising_ham(4, J=1.0, h=0.5)
        true_E0 = float(np.linalg.eigvalsh(H)[0])
        rng = np.random.default_rng(17)
        psi = rng.standard_normal(H.shape[0])
        mps = _MockMPS(psi)
        cert = rayleigh_from_quimb_mps(mps, H, notes="tfim_test")
        assert cert.upper >= true_E0 - 1e-9

    def test_input_digest_present(self, diag2):
        cert = rayleigh_from_quimb_mps(_MockMPS([1.0, 0.0]), diag2)
        assert len(cert.input_digest) == 64
        int(cert.input_digest, 16)  # valid hex

    def test_can_be_independently_verified(self, diag2):
        from htf.rayleigh_cert import verify_rayleigh_certificate
        cert = rayleigh_from_quimb_mps(_MockMPS([1.0, 0.0]), diag2)
        verify_rayleigh_certificate(cert)
        assert cert.verified is True


# ─────────────────── rayleigh_from_quimb_mps: error cases ─────────────────────

class TestRayleighFromQuimbMPSErrors:

    def test_no_to_dense_raises(self):
        H = np.diag([0.0, 1.0])
        with pytest.raises(TypeError, match="to_dense"):
            rayleigh_from_quimb_mps(_NoDenseMPS(), H)

    def test_dim_mismatch_raises(self):
        H = np.diag([0.0, 1.0, 2.0])  # 3×3
        mps = _MockMPS([1.0, 0.0])    # length 2
        with pytest.raises(ValueError):
            rayleigh_from_quimb_mps(mps, H)

    def test_asymmetric_H_raises(self):
        H = np.array([[0.0, 1.0], [2.0, 0.0]])  # not symmetric
        mps = _MockMPS([1.0, 0.0])
        with pytest.raises(ValueError, match="symmetric"):
            rayleigh_from_quimb_mps(mps, H)

    def test_zero_norm_psi_raises(self):
        H = np.diag([0.0, 1.0])
        mps = _MockMPS([0.0, 0.0])
        with pytest.raises(ValueError, match="zero norm"):
            rayleigh_from_quimb_mps(mps, H)

    def test_complex_H_accepted(self):
        H = np.eye(2, dtype=complex)
        mps = _MockMPS([1.0, 0.0])
        cert = rayleigh_from_quimb_mps(mps, H)
        assert cert.upper <= 1.0 + 1e-9

    def test_basis_permutation_changes_rayleigh(self):
        """Dimension match alone cannot detect a site permutation in H vs ψ."""
        H_std = np.diag([1.0, 0.0])        # E0 = 0 at index 1
        H_perm = np.diag([0.0, 1.0])       # E0 = 0 at index 0
        psi = np.array([0.0, 1.0])          # ground state of H_std
        cert_std  = rayleigh_from_quimb_mps(_MockMPS(psi), H_std)
        cert_perm = rayleigh_from_quimb_mps(_MockMPS(psi), H_perm)
        # Correct pairing: R_H_std(ψ) = 0.0
        assert cert_std.upper <= 0.0 + 1e-9
        # Mismatched pairing: R_H_perm(ψ) = 1.0 — adapter cannot detect this
        assert cert_perm.upper >= 0.9


# ─────────────────── to_full_json roundtrip via adapter ───────────────────────

class TestAdapterJSONRoundtrip:

    def test_to_full_json_roundtrip(self):
        import json

        from htf.verify import verify_from_dict
        H = np.diag([0.0, 1.0])
        cert = rayleigh_from_quimb_mps(_MockMPS([1.0, 0.0]), H)
        full = json.loads(cert.to_full_json())
        result = verify_from_dict(full)
        assert result["verified"] is True

    def test_to_json_has_adapter_note(self):
        import json
        H = np.diag([0.0, 1.0])
        cert = rayleigh_from_quimb_mps(_MockMPS([1.0, 0.0]), H, notes="run1")
        d = json.loads(cert.to_json())
        assert "quimb-adapter" in d["notes"]
