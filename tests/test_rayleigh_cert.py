"""Tests for htf/rayleigh_cert.py — Validated Rayleigh Certificate."""
import json
import math

import numpy as np
import pytest

from htf.rayleigh_cert import (
    SCHEMA_VERSION,
    RayleighCertificate,
    rayleigh_certificate,
    validate_certificate_dict,
    verify_rayleigh_certificate,
)

# ─────────────────── fixtures ────────────────────────────────────────────────

@pytest.fixture
def diag2():
    """2×2 diagonal H = diag(0, 1), E0=0, E1=1."""
    return np.diag([0.0, 1.0])


@pytest.fixture
def tfim4():
    """TFIM n=4 Hamiltonian."""
    from htf.variational import transverse_ising_ham
    return transverse_ising_ham(4, J=1.0, h=0.5)


# ─────────────────── rayleigh_certificate ────────────────────────────────────

class TestRayleighCertificate:

    def test_returns_rayleigh_certificate(self, diag2):
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(diag2, psi)
        assert isinstance(cert, RayleighCertificate)

    def test_upper_bounds_E0_exact_gs(self, diag2):
        # Exact ground state: Rayleigh quotient = E0 = 0
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(diag2, psi)
        assert cert.upper >= 0.0 - 1e-12
        assert cert.upper <= 0.0 + 1e-9

    def test_upper_bounds_E0_mixed_state(self, diag2):
        # Mixed state: Rayleigh quotient > E0 but still an upper bound
        psi = np.array([1.0, 1.0]) / math.sqrt(2)
        cert = rayleigh_certificate(diag2, psi)
        true_E0 = 0.0
        assert cert.upper >= true_E0 - 1e-12

    def test_upper_bounds_E0_tfim(self, tfim4):
        evals = np.linalg.eigvalsh(tfim4)
        true_E0 = float(evals[0])
        rng = np.random.default_rng(7)
        psi = rng.standard_normal(tfim4.shape[0])
        cert = rayleigh_certificate(tfim4, psi)
        assert cert.upper >= true_E0 - 1e-9, "upper must be ≥ E0"

    def test_unnormalised_psi_gives_same_result(self, diag2):
        psi_norm = np.array([1.0, 0.0])
        psi_scaled = psi_norm * 5.0
        c1 = rayleigh_certificate(diag2, psi_norm)
        c2 = rayleigh_certificate(diag2, psi_scaled)
        # Same Rayleigh quotient, same digest (digest uses raw psi before normalisation)
        assert abs(c1.upper - c2.upper) < 1e-10

    def test_claim_contains_upper(self, diag2):
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(diag2, psi)
        assert "E0" in cert.claim
        assert str(cert.upper)[:5] in cert.claim or "≤" in cert.claim

    def test_theorem_is_rayleigh_ritz(self, diag2):
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(diag2, psi)
        assert "Rayleigh" in cert.theorem or "Ritz" in cert.theorem

    def test_assumptions_checked_symmetry(self, diag2):
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(diag2, psi)
        assert any("symmetric" in a.lower() for a in cert.assumptions)

    def test_assumptions_checked_norm(self, diag2):
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(diag2, psi)
        assert any("ψ|ψ" in a or "norm" in a.lower() for a in cert.assumptions)

    def test_input_digest_is_sha256(self, diag2):
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(diag2, psi)
        assert len(cert.input_digest) == 64  # 32 bytes → 64 hex chars
        int(cert.input_digest, 16)           # must be valid hex

    def test_digest_changes_with_different_inputs(self, diag2):
        psi1 = np.array([1.0, 0.0])
        psi2 = np.array([0.0, 1.0])
        c1 = rayleigh_certificate(diag2, psi1)
        c2 = rayleigh_certificate(diag2, psi2)
        assert c1.input_digest != c2.input_digest

    def test_radius_non_negative(self, diag2):
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(diag2, psi)
        assert cert.radius >= 0.0

    def test_interval_lower_leq_upper(self, diag2):
        psi = np.array([0.6, 0.8])
        cert = rayleigh_certificate(diag2, psi)
        assert cert.lower <= cert.upper

    def test_verified_false_before_verify(self, diag2):
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(diag2, psi)
        assert cert.verified is False

    def test_htf_version_recorded(self, diag2):
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(diag2, psi)
        assert cert.htf_version != ""

    def test_raises_on_non_square_H(self):
        with pytest.raises(ValueError, match="square"):
            rayleigh_certificate(np.ones((2, 3)), np.array([1.0, 0.0]))

    def test_raises_on_dim_mismatch(self, diag2):
        with pytest.raises(ValueError, match="length"):
            rayleigh_certificate(diag2, np.array([1.0, 0.0, 0.0]))

    def test_raises_on_asymmetric_H(self):
        H = np.array([[1.0, 2.0], [3.0, 1.0]])  # not symmetric
        with pytest.raises(ValueError, match="symmetric"):
            rayleigh_certificate(H, np.array([1.0, 0.0]))

    def test_raises_on_zero_norm_psi(self, diag2):
        with pytest.raises(ValueError, match="zero norm"):
            rayleigh_certificate(diag2, np.zeros(2))

    def test_accepts_complex_hermitian_H(self):
        # Complex Hermitian H is now supported — no longer raises
        H = np.eye(2, dtype=complex)
        cert = rayleigh_certificate(H, np.array([1.0, 0.0]))
        assert cert.upper <= 1.0 + 1e-9
        assert "flint-acb" in cert.backend or "numpy-complex" in cert.backend

    def test_accepts_complex_psi(self, diag2):
        # Complex psi with real H is now supported
        psi = np.array([1.0 + 0j, 0.0 + 0j])
        cert = rayleigh_certificate(diag2, psi)
        assert cert.upper <= 0.0 + 1e-9

    def test_to_dict_has_required_keys(self, diag2):
        cert = rayleigh_certificate(diag2, np.array([1.0, 0.0]))
        d = cert.to_dict()
        for key in ("schema_version", "claim", "theorem", "assumptions",
                    "input_digest", "interval", "backend", "htf_version", "verified"):
            assert key in d

    def test_to_dict_schema_version(self, diag2):
        cert = rayleigh_certificate(diag2, np.array([1.0, 0.0]))
        assert cert.to_dict()["schema_version"] == SCHEMA_VERSION

    def test_to_json_roundtrip(self, diag2):
        import json
        cert = rayleigh_certificate(diag2, np.array([1.0, 0.0]))
        d = json.loads(cert.to_json())
        assert d["verified"] is False
        assert "upper" in d["interval"]


# ─────────────────── v1 schema validation ────────────────────────────────────

class TestValidateCertificateDict:

    @pytest.fixture
    def valid_dict(self):
        H = np.diag([0.0, 1.0])
        cert = rayleigh_certificate(H, np.array([1.0, 0.0]))
        return cert.to_dict()

    def test_valid_dict_passes(self, valid_dict):
        validate_certificate_dict(valid_dict)  # must not raise

    def test_validate_method_passes(self):
        H = np.diag([0.0, 1.0])
        cert = rayleigh_certificate(H, np.array([1.0, 0.0]))
        cert.validate()  # must not raise

    def test_missing_required_key(self, valid_dict):
        del valid_dict["claim"]
        with pytest.raises(ValueError, match="missing"):
            validate_certificate_dict(valid_dict)

    def test_wrong_schema_version(self, valid_dict):
        valid_dict["schema_version"] = "rayleigh-cert/v0"
        with pytest.raises(ValueError, match="schema_version"):
            validate_certificate_dict(valid_dict)

    def test_bad_digest_length(self, valid_dict):
        valid_dict["input_digest"] = "abc123"
        with pytest.raises(ValueError, match="input_digest"):
            validate_certificate_dict(valid_dict)

    def test_bad_digest_uppercase(self, valid_dict):
        valid_dict["input_digest"] = valid_dict["input_digest"].upper()
        with pytest.raises(ValueError, match="input_digest"):
            validate_certificate_dict(valid_dict)

    def test_interval_lower_gt_upper(self, valid_dict):
        valid_dict["interval"]["lower"] = valid_dict["interval"]["upper"] + 1.0
        with pytest.raises(ValueError, match="lower.*upper|upper.*lower"):
            validate_certificate_dict(valid_dict)

    def test_interval_negative_radius(self, valid_dict):
        valid_dict["interval"]["radius"] = -0.1
        with pytest.raises(ValueError, match="radius"):
            validate_certificate_dict(valid_dict)

    def test_interval_inconsistent_midpoint(self, valid_dict):
        valid_dict["interval"]["midpoint"] += 999.0
        with pytest.raises(ValueError, match="midpoint"):
            validate_certificate_dict(valid_dict)

    def test_empty_assumptions(self, valid_dict):
        valid_dict["assumptions"] = []
        with pytest.raises(ValueError, match="assumptions"):
            validate_certificate_dict(valid_dict)

    def test_verified_wrong_type(self, valid_dict):
        valid_dict["verified"] = "yes"
        with pytest.raises(ValueError, match="verified"):
            validate_certificate_dict(valid_dict)

    def test_non_dict_raises(self):
        with pytest.raises(ValueError):
            validate_certificate_dict("not a dict")

    def test_interval_not_a_dict(self, valid_dict):
        valid_dict["interval"] = "not-a-dict"
        with pytest.raises(ValueError, match="interval must be a dict"):
            validate_certificate_dict(valid_dict)

    def test_interval_missing_keys(self, valid_dict):
        valid_dict["interval"] = {"lower": 0.0}
        with pytest.raises(ValueError, match="interval missing keys"):
            validate_certificate_dict(valid_dict)

    def test_interval_value_non_numeric(self, valid_dict):
        valid_dict["interval"]["lower"] = "not-a-number"
        with pytest.raises(ValueError, match="numeric"):
            validate_certificate_dict(valid_dict)

    def test_interval_value_non_finite(self, valid_dict):
        valid_dict["interval"]["upper"] = float("inf")
        with pytest.raises(ValueError, match="finite"):
            validate_certificate_dict(valid_dict)

    def test_interval_radius_too_small_but_positive(self, valid_dict):
        # radius is positive but doesn't cover both endpoints
        lo = valid_dict["interval"]["lower"]
        up = valid_dict["interval"]["upper"]
        mid = (lo + up) / 2
        valid_dict["interval"]["radius"] = abs(up - mid) * 0.1  # too small
        with pytest.raises(ValueError, match="radius"):
            validate_certificate_dict(valid_dict)

    def test_invalid_assurance_value(self, valid_dict):
        valid_dict["assurance"] = "super-rigorous"
        with pytest.raises(ValueError, match="assurance"):
            validate_certificate_dict(valid_dict)


# ─────────────────── from_dict roundtrip ─────────────────────────────────────

class TestFromDict:

    def test_from_dict_roundtrip(self):
        H = np.diag([0.0, 1.0])
        cert = rayleigh_certificate(H, np.array([1.0, 0.0]))
        cert2 = RayleighCertificate.from_dict(cert.to_dict())
        assert cert2.claim == cert.claim
        assert cert2.upper == cert.upper
        assert cert2.input_digest == cert.input_digest
        assert cert2.schema_version == SCHEMA_VERSION

    def test_from_full_dict_has_canonical(self):
        H = np.diag([0.0, 1.0])
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(H, psi)
        cert2 = RayleighCertificate.from_dict(cert.to_full_dict())
        assert len(cert2._H_canonical) == 2
        assert len(cert2._psi_canonical) == 2

    def test_from_dict_then_verify(self):
        H = np.diag([0.0, 1.0])
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(H, psi)
        # Roundtrip through full JSON
        cert2 = RayleighCertificate.from_dict(json.loads(cert.to_full_json()))
        verify_rayleigh_certificate(cert2)
        assert cert2.verified is True

    def test_from_dict_rejects_invalid(self):
        with pytest.raises(ValueError):
            RayleighCertificate.from_dict({"schema_version": "bad"})


# ─────────────────── htf_version correctness ─────────────────────────────────

class TestHtfVersion:

    def test_htf_version_is_not_unknown(self):
        H = np.diag([0.0, 1.0])
        cert = rayleigh_certificate(H, np.array([1.0, 0.0]))
        assert cert.htf_version != "unknown"

    def test_htf_version_looks_like_semver(self):
        H = np.diag([0.0, 1.0])
        cert = rayleigh_certificate(H, np.array([1.0, 0.0]))
        parts = cert.htf_version.split(".")
        assert len(parts) >= 2
        assert all(p.isdigit() for p in parts[:2])


# ─────────────────── schema file exists and is valid JSON ────────────────────

class TestSchemaFile:

    def test_schema_file_is_valid_json(self):
        from pathlib import Path
        schema_path = Path(__file__).parent.parent / "htf" / "schemas" / "rayleigh_cert_v2.json"
        assert schema_path.exists(), "htf/schemas/rayleigh_cert_v2.json not found"
        schema = json.loads(schema_path.read_text())
        assert schema.get("$id") == "htf://schemas/rayleigh-cert/v2"
        assert "required" in schema
        assert "schema_version" in schema["required"]

    def test_schema_file_matches_constant(self):
        from pathlib import Path
        schema_path = Path(__file__).parent.parent / "htf" / "schemas" / "rayleigh_cert_v2.json"
        schema = json.loads(schema_path.read_text())
        const = schema["properties"]["schema_version"]["const"]
        assert const == SCHEMA_VERSION


# ─────────────────── verify_rayleigh_certificate ─────────────────────────────

class TestVerifyRayleighCertificate:

    def test_verify_sets_verified_true(self, diag2):
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(diag2, psi)
        cert = verify_rayleigh_certificate(cert)
        assert cert.verified is True

    def test_verify_returns_certificate(self, diag2):
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(diag2, psi)
        result = verify_rayleigh_certificate(cert)
        assert isinstance(result, RayleighCertificate)

    def test_verify_recomputes_same_upper(self, diag2):
        psi = np.array([0.6, 0.8])
        cert = rayleigh_certificate(diag2, psi)
        stored_upper = cert.upper
        verify_rayleigh_certificate(cert)
        assert abs(cert.upper - stored_upper) < 1e-14

    def test_verify_rejects_tampered_digest(self, diag2):
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(diag2, psi)
        cert.input_digest = "0" * 64  # tamper
        with pytest.raises(ValueError, match="digest"):
            verify_rayleigh_certificate(cert)

    def test_verify_rejects_tampered_upper(self, diag2):
        psi = np.array([0.6, 0.8])
        cert = rayleigh_certificate(diag2, psi)
        cert.upper = -999.0  # claim a false lower upper bound
        with pytest.raises(ValueError, match="digest|upper"):
            verify_rayleigh_certificate(cert)

    def test_end_to_end_tfim(self, tfim4):
        evals = np.linalg.eigvalsh(tfim4)
        true_E0 = float(evals[0])
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(tfim4.shape[0])
        cert = rayleigh_certificate(tfim4, psi)
        cert = verify_rayleigh_certificate(cert)
        assert cert.verified
        assert cert.upper >= true_E0 - 1e-9

    def test_exact_gs_upper_close_to_E0(self, diag2):
        # Exact GS: Rayleigh quotient should equal E0 = 0 to machine precision
        _, evecs = np.linalg.eigh(diag2)
        psi_gs = evecs[:, 0]
        cert = rayleigh_certificate(diag2, psi_gs)
        cert = verify_rayleigh_certificate(cert)
        assert cert.upper <= 0.0 + 1e-9
        assert cert.verified


# ─────────────────── oracle: upper bound tightens with better state ───────────

class TestRayleighOracleTightening:

    def test_better_state_gives_tighter_upper(self):
        # H = diag(0, 1), true E0 = 0.
        # psi_good = exact GS → upper ≈ 0;  psi_bad = exact ES → upper ≈ 1.
        H = np.diag([0.0, 1.0])
        _, evecs = np.linalg.eigh(H)
        cert_gs = rayleigh_certificate(H, evecs[:, 0])
        cert_es = rayleigh_certificate(H, evecs[:, 1])
        assert cert_gs.upper < cert_es.upper

    def test_random_state_upper_strictly_above_E0(self):
        from htf.variational import transverse_ising_ham
        H = transverse_ising_ham(3)
        true_E0 = float(np.linalg.eigvalsh(H)[0])
        rng = np.random.default_rng(99)
        psi = rng.standard_normal(H.shape[0])
        cert = rayleigh_certificate(H, psi)
        assert cert.upper > true_E0 + 1e-10  # random state is not exact GS


# ─────────────────── complex Hermitian support ───────────────────────────────

class TestComplexHermitianSupport:

    @pytest.fixture
    def heis2(self):
        """2-qubit Heisenberg XXX: H = 0.25*(XX+YY+ZZ), E0=-0.75."""
        H = np.array([
            [0.25,   0,    0,  0  ],
            [0,    -0.25, 0.5, 0  ],
            [0,     0.5, -0.25, 0  ],
            [0,     0,    0,  0.25],
        ], dtype=complex)
        return H

    def test_complex_hermitian_h_accepted(self, heis2):
        psi = np.array([0, 1, -1, 0], dtype=complex) / math.sqrt(2)
        cert = rayleigh_certificate(heis2, psi)
        assert isinstance(cert, RayleighCertificate)

    def test_complex_h_upper_bounds_e0(self, heis2):
        true_E0 = float(np.linalg.eigvalsh(heis2)[0])  # -0.75
        psi = np.array([0, 1, -1, 0], dtype=complex) / math.sqrt(2)
        cert = rayleigh_certificate(heis2, psi)
        assert cert.upper >= true_E0 - 1e-9

    def test_complex_h_exact_gs_tight(self, heis2):
        evals, evecs = np.linalg.eigh(heis2)
        psi_gs = evecs[:, 0]
        cert = rayleigh_certificate(heis2, psi_gs)
        assert cert.upper <= evals[0] + 1e-9

    def test_complex_h_backend_is_acb(self, heis2):
        psi = np.array([0, 1, -1, 0], dtype=complex) / math.sqrt(2)
        cert = rayleigh_certificate(heis2, psi)
        assert "flint-acb" in cert.backend or "numpy-complex" in cert.backend

    def test_complex_h_digest_is_sha256(self, heis2):
        psi = np.array([0, 1, -1, 0], dtype=complex) / math.sqrt(2)
        cert = rayleigh_certificate(heis2, psi)
        assert len(cert.input_digest) == 64
        int(cert.input_digest, 16)

    def test_complex_h_verify_roundtrip(self, heis2):
        psi = np.array([0, 1, -1, 0], dtype=complex) / math.sqrt(2)
        cert = rayleigh_certificate(heis2, psi)
        verify_rayleigh_certificate(cert)
        assert cert.verified is True

    def test_complex_h_from_dict_roundtrip(self, heis2):
        psi = np.array([0, 1, -1, 0], dtype=complex) / math.sqrt(2)
        cert = rayleigh_certificate(heis2, psi)
        cert2 = RayleighCertificate.from_dict(json.loads(cert.to_full_json()))
        verify_rayleigh_certificate(cert2)
        assert cert2.verified is True

    def test_complex_psi_real_h(self):
        H = np.diag([0.0, 1.0])
        psi = np.array([1 + 0j, 0 + 0j])
        cert = rayleigh_certificate(H, psi)
        assert cert.upper <= 0.0 + 1e-9

    def test_non_hermitian_complex_h_raises(self):
        H = np.array([[1.0, 2 + 1j], [2 + 0j, 1.0]])  # not Hermitian
        psi = np.array([1.0 + 0j, 0.0])
        with pytest.raises(ValueError, match="[Hh]ermitian"):
            rayleigh_certificate(H, psi)

    def test_schema_version_preserved(self, heis2):
        psi = np.array([0, 1, -1, 0], dtype=complex) / math.sqrt(2)
        cert = rayleigh_certificate(heis2, psi)
        assert cert.to_dict()["schema_version"] == SCHEMA_VERSION

    def test_validate_passes_for_complex_cert(self, heis2):
        psi = np.array([0, 1, -1, 0], dtype=complex) / math.sqrt(2)
        cert = rayleigh_certificate(heis2, psi)
        cert.validate()  # must not raise


# ─────────────────── adversarial regression tests (R7) ───────────────────────

class TestAdversarialRegressions:
    """Referee R7 regression counterexamples that v1 got wrong."""

    def test_near_symmetric_H_rejected(self):
        # R7-1: old code used max|H-Hᵀ| <= 1e-10 (tolerance-based).
        # 5e-11 < 1e-10 so old code silently accepted this non-symmetric H.
        H = np.array([[0.0, 5e-11], [0.0, 0.0]])
        psi = np.array([1.0, -1.0])
        with pytest.raises(ValueError, match="symmetric"):
            rayleigh_certificate(H, psi)

    def test_nan_in_H_raises(self):
        # R7-2: NaN must fail closed before any comparison.
        H = np.array([[np.nan, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="non-finite|NaN"):
            rayleigh_certificate(H, np.array([1.0, 0.0]))

    def test_nan_in_psi_raises(self):
        H = np.diag([0.0, 1.0])
        with pytest.raises(ValueError, match="non-finite|NaN"):
            rayleigh_certificate(H, np.array([np.nan, 0.0]))

    def test_inf_in_H_raises(self):
        H = np.array([[np.inf, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="non-finite|NaN"):
            rayleigh_certificate(H, np.array([1.0, 0.0]))

    def test_one_third_upper_bound_is_sound(self):
        # R7-3: v1 float(mid)+float(rad) at prec=64 can give upper < 1/3.
        # v2 (prec=128 + nextafter) must give a sound upper bound >= 1/3.
        pytest.importorskip("flint")
        from fractions import Fraction
        H = np.diag([1.0, 0.0, 0.0])
        psi = np.array([1.0, 1.0, 1.0])
        cert = rayleigh_certificate(H, psi)
        assert Fraction.from_float(cert.upper) >= Fraction(1, 3), (
            f"cert.upper={cert.upper:.17g} is below 1/3; "
            "Arb upper bound at prec=128 must be outward-rounded"
        )

    def test_v2_digest_prevents_structural_collision(self):
        # R7-5: v1 digest used native-endian bytes with no domain separation,
        # allowing real n=3 and complex n=2 to produce the same raw bytes.
        from htf.rayleigh_cert import _canonical_digest
        H_real  = np.diag([1.0, 0.0, 1.0])
        psi_real = np.array([0.0, 1.0, 0.0])
        H_cplx  = np.diag([1.0 + 0.0j, 0.0 + 0.0j])
        psi_cplx = np.array([0.0 + 1.0j, 0.0 + 0.0j])
        d_real = _canonical_digest(H_real, psi_real)
        d_cplx = _canonical_digest(H_cplx, psi_cplx)
        assert d_real != d_cplx, (
            "v2 digest must distinguish real n=3 from complex n=2"
        )

    def test_signed_zero_preserved_in_roundtrip(self):
        # R7-6: arithmetic reconstruction (real + 1j*imag) destroys -0.0 bits.
        # Component assignment must preserve signed zeros for digest stability.
        from htf.rayleigh_cert import _canonical_digest, _decode_canonical, _encode_canonical
        H = np.zeros((2, 2), dtype=np.complex128)
        H[0, 0] = 1.0
        psi = np.zeros(2, dtype=np.complex128)
        psi[0] = 1.0
        psi.imag[0] = -0.0
        encoded = _encode_canonical(psi)
        decoded = _decode_canonical(encoded)
        assert _canonical_digest(H, psi) == _canonical_digest(H, decoded), (
            "decode-then-digest must match original digest (signed-zero preservation)"
        )

    def test_strict_verifier_rejects_tolerance_attack(self):
        # R7-7: old verifier used tol = max(|upper|*1e-14, 1e-15), allowing
        # a tampered upper (slightly below the true quotient) to pass.
        import copy
        H = np.diag([1.0, 2.0])
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(H, psi)  # RQ = 1.0; cert.upper >= 1.0

        cert_tampered = copy.deepcopy(cert)
        attacked_upper = cert.upper - 5e-15
        cert_tampered.upper    = attacked_upper
        cert_tampered.lower    = attacked_upper
        cert_tampered.midpoint = attacked_upper
        cert_tampered.radius   = 0.0
        cert_tampered.claim = (
            f"E0 ≤ {attacked_upper:.17g}  "
            "[Rayleigh-Ritz upper bound on ground-state energy]"
        )
        with pytest.raises(ValueError):
            verify_rayleigh_certificate(cert_tampered)

    def test_claim_mutation_detected(self):
        # R7-8: changing only the claim text must be detected by the verifier.
        H = np.diag([0.0, 1.0])
        psi = np.array([0.6, 0.8])
        cert = rayleigh_certificate(H, psi)
        cert.claim = "E0 ≤ 0.0  [Rayleigh-Ritz upper bound on ground-state energy]"
        with pytest.raises(ValueError, match="claim"):
            verify_rayleigh_certificate(cert)


# ── no-flint guard (P0-F1 regression, P2-A) ──────────────────────────────────

class TestNoFlintGuard:
    """Verify that the absence of python-flint is handled safely.

    These tests mock out flint to confirm fail-fast behaviour without requiring
    an environment where flint is actually missing.
    """

    def test_rayleigh_certificate_raises_without_flint(self, monkeypatch):
        # rayleigh_certificate() must raise ImportError, not silently fall back
        # to a numpy float path that would pass verify trivially (F-1).
        # Setting sys.modules["flint"] = None makes any `import flint` inside
        # the function body raise ImportError — no module reload needed.
        import sys

        from htf.rayleigh_cert import rayleigh_certificate as _rc
        monkeypatch.setitem(sys.modules, "flint", None)
        H = np.diag([1.0, 2.0])
        psi = np.array([1.0, 0.0])
        with pytest.raises(ImportError, match="python-flint"):
            _rc(H, psi)

    def test_verify_rayleigh_certificate_raises_without_flint(self, monkeypatch):
        # verify_rayleigh_certificate() must also require flint.
        import sys

        from htf.rayleigh_cert import verify_rayleigh_certificate as _vrc
        H = np.diag([1.0, 2.0])
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(H, psi)  # produced with flint (available here)
        monkeypatch.setitem(sys.modules, "flint", None)
        with pytest.raises(ImportError, match="python-flint"):
            _vrc(cert)


# ── rayleigh_estimate (non-certified float path) ──────────────────────────────

class TestRayleighEstimate:
    """rayleigh_estimate() provides a float-only non-certified path."""

    def test_returns_rayleigh_certificate(self):
        from htf.rayleigh_cert import RayleighCertificate, rayleigh_estimate
        H = np.diag([1.0, 3.0])
        psi = np.array([1.0, 0.0])
        est = rayleigh_estimate(H, psi)
        assert isinstance(est, RayleighCertificate)

    def test_assurance_is_heuristic(self):
        from htf.rayleigh_cert import rayleigh_estimate
        H = np.diag([1.0, 3.0])
        psi = np.array([1.0, 0.0])
        est = rayleigh_estimate(H, psi)
        assert est.assurance == "heuristic"

    def test_verified_is_false(self):
        from htf.rayleigh_cert import rayleigh_estimate
        H = np.diag([1.0, 3.0])
        psi = np.array([1.0, 0.0])
        est = rayleigh_estimate(H, psi)
        assert est.verified is False

    def test_radius_is_zero(self):
        from htf.rayleigh_cert import rayleigh_estimate
        H = np.diag([1.0, 3.0])
        psi = np.array([1.0, 0.0])
        est = rayleigh_estimate(H, psi)
        assert est.radius == 0.0

    def test_upper_equals_midpoint(self):
        from htf.rayleigh_cert import rayleigh_estimate
        H = np.diag([1.0, 3.0])
        psi = np.array([1.0, 0.0])
        est = rayleigh_estimate(H, psi)
        assert est.upper == est.midpoint

    def test_result_is_correct_value(self):
        from htf.rayleigh_cert import rayleigh_estimate
        H = np.diag([1.0, 3.0])
        psi = np.array([1.0, 0.0])
        est = rayleigh_estimate(H, psi)
        assert abs(est.upper - 1.0) < 1e-12

    def test_notes_warn_no_rigorous_bound(self):
        from htf.rayleigh_cert import rayleigh_estimate
        H = np.diag([1.0, 3.0])
        psi = np.array([1.0, 0.0])
        est = rayleigh_estimate(H, psi)
        assert "Float estimate" in est.notes or "no rigorous" in est.notes.lower()

    def test_backend_mentions_numpy(self):
        from htf.rayleigh_cert import rayleigh_estimate
        H = np.diag([1.0, 3.0])
        psi = np.array([1.0, 0.0])
        est = rayleigh_estimate(H, psi)
        assert "numpy" in est.backend.lower()

    def test_digest_is_valid_sha256(self):
        import re

        from htf.rayleigh_cert import rayleigh_estimate
        H = np.diag([1.0, 3.0])
        psi = np.array([1.0, 0.0])
        est = rayleigh_estimate(H, psi)
        assert re.match(r"^[0-9a-f]{64}$", est.input_digest)

    def test_complex_hamiltonian(self):
        from htf.rayleigh_cert import rayleigh_estimate
        H = np.array([[1.0+0j, 1j], [-1j, 2.0+0j]])
        psi = np.array([1.0+0j, 0.0+0j])
        est = rayleigh_estimate(H, psi)
        assert est.assurance == "heuristic"
        assert abs(est.upper - 1.0) < 1e-12

    def test_to_dict_has_assurance(self):
        from htf.rayleigh_cert import rayleigh_estimate
        H = np.diag([1.0, 3.0])
        psi = np.array([1.0, 0.0])
        d = rayleigh_estimate(H, psi).to_dict()
        assert d["assurance"] == "heuristic"


# ── C7: git_commit provenance field ──────────────────────────────────────────

class TestGitCommitProvenance:
    """C7: every certificate carries a git_commit provenance field."""

    def test_git_commit_present_in_cert(self):
        from htf.rayleigh_cert import rayleigh_certificate
        H = np.diag([0.0, 1.0])
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(H, psi)
        assert hasattr(cert, "git_commit")
        assert isinstance(cert.git_commit, str)

    def test_git_commit_in_to_dict(self):
        from htf.rayleigh_cert import rayleigh_certificate
        H = np.diag([0.0, 1.0])
        psi = np.array([1.0, 0.0])
        d = rayleigh_certificate(H, psi).to_dict()
        assert "git_commit" in d
        assert isinstance(d["git_commit"], str)

    def test_git_commit_roundtrips_from_dict(self):
        from htf.rayleigh_cert import RayleighCertificate, rayleigh_certificate
        H = np.diag([0.0, 1.0])
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(H, psi)
        cert2 = RayleighCertificate.from_dict(cert.to_dict())
        assert cert2.git_commit == cert.git_commit

    def test_git_commit_nonempty_inside_repo(self):
        from htf.rayleigh_cert import rayleigh_certificate
        H = np.diag([0.0, 1.0])
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(H, psi)
        # Inside the htf git repo this should be a non-empty short SHA.
        # Allow empty for environments where git is absent.
        if cert.git_commit:
            import re
            assert re.fullmatch(r"[0-9a-f]{4,40}", cert.git_commit), (
                f"git_commit looks malformed: {cert.git_commit!r}"
            )

    def test_estimate_also_has_git_commit(self):
        from htf.rayleigh_cert import rayleigh_estimate
        H = np.diag([1.0, 2.0])
        psi = np.array([1.0, 0.0])
        cert = rayleigh_estimate(H, psi)
        assert "git_commit" in cert.to_dict()

    def test_git_commit_invalid_type_rejected(self):
        from htf.rayleigh_cert import rayleigh_certificate, validate_certificate_dict
        H = np.diag([0.0, 1.0])
        psi = np.array([1.0, 0.0])
        d = rayleigh_certificate(H, psi).to_dict()
        d["git_commit"] = 12345  # not a string
        with pytest.raises(ValueError, match="git_commit"):
            validate_certificate_dict(d)


# ─────────────────── HTF-05 regressions (dtype, NaN, assurance) ──────────────

class TestHTF05Regressions:
    """Regressions for HTF-05 Gate-A blockers: M1–M3."""

    def test_rejects_int64_dtype(self):
        # M1: int64 H must be rejected before float64 conversion silently changes it.
        H = np.array([[1, 0], [0, 2]], dtype=np.int64)
        psi = np.array([1, 0], dtype=np.int64)
        with pytest.raises(TypeError, match="float64"):
            rayleigh_certificate(H, psi)

    def test_rejects_float32_dtype(self):
        # M1: float32 also not canonical.
        H = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        psi = np.array([1.0, 0.0], dtype=np.float32)
        with pytest.raises(TypeError, match="float64"):
            rayleigh_certificate(H, psi)

    def test_rejects_psi_non_canonical_dtype(self):
        # M1: H is float64 but psi is float32 — psi dtype path must fire.
        H = np.diag([1.0, 2.0]).astype(np.float64)
        psi = np.array([1.0, 0.0], dtype=np.float32)
        with pytest.raises(TypeError, match="psi dtype"):
            rayleigh_certificate(H, psi)

    def test_verifier_nan_upper_rejected(self):
        # M2: NaN upper must be rejected, not silently pass.
        from htf.verify import verify_from_dict
        H = np.diag([0.0, 1.0])
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(H, psi)
        full = cert.to_full_dict()
        # Tamper upper to NaN; also update claim to match so the claim check
        # doesn't fire first — we want to exercise the isfinite path.
        full["interval"]["upper"] = float("nan")
        full["claim"] = "E0 ≤ nan  [Rayleigh-Ritz upper bound on ground-state energy]"
        result = verify_from_dict(full)
        assert result["verified"] is False

    def test_verifier_nan_lower_rejected(self):
        # M2: NaN lower must be rejected (original upper/claim untouched).
        from htf.verify import verify_from_dict
        H = np.diag([0.0, 1.0])
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(H, psi)
        full = cert.to_full_dict()
        full["interval"]["lower"] = float("nan")
        result = verify_from_dict(full)
        assert result["verified"] is False

    def test_verifier_nan_radius_rejected(self):
        # M2: NaN radius must also be rejected.
        from htf.verify import verify_from_dict
        H = np.diag([0.0, 1.0])
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(H, psi)
        full = cert.to_full_dict()
        full["interval"]["radius"] = float("nan")
        result = verify_from_dict(full)
        assert result["verified"] is False

    def test_verifier_missing_assurance_rejected(self):
        # M3: cert without assurance field must be rejected, not default to rigorous.
        from htf.verify import verify_from_dict
        H = np.diag([0.0, 1.0])
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(H, psi)
        full = cert.to_full_dict()
        del full["assurance"]
        with pytest.raises(ValueError, match="assurance"):
            verify_from_dict(full)

    def test_anchor_zero_exact_nextafter(self):
        # M6: H=diag(0,1), psi=[1,0] — Rayleigh quotient is exactly 0.
        # lower = nextafter(0, -inf), upper = nextafter(0, +inf).
        H = np.diag([0.0, 1.0])
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(H, psi)
        assert cert.lower == math.nextafter(0.0, -math.inf)
        assert cert.upper == math.nextafter(0.0, math.inf)

    def test_anchor_minus_five_uses_ulp(self):
        # M6: H=diag(-5,1), psi=[1,0] — Rayleigh quotient is exactly -5.
        # upper = nextafter(-5, +inf); upper + 5.0 == math.ulp(-5.0).
        H = np.diag([-5.0, 1.0])
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(H, psi)
        assert cert.upper == math.nextafter(-5.0, math.inf)
        assert cert.upper + 5.0 == math.ulp(-5.0)
        assert cert.lower == math.nextafter(-5.0, -math.inf)


# ─────────────────── P0-1 malicious certificate corpus ───────────────────────

class TestMaliciousCertificateCorpus:
    """verify_rayleigh_certificate must reject tampered/downgraded certs."""

    def _make_cert(self):
        H = np.diag([0.0, 1.0, 2.0])
        psi = np.array([1.0, 0.0, 0.0])
        cert = rayleigh_certificate(H, psi)
        verify_rayleigh_certificate(cert)
        return cert

    def test_heuristic_assurance_rejected(self):
        cert = self._make_cert()
        cert.assurance = "heuristic"
        with pytest.raises(ValueError, match="assurance"):
            verify_rayleigh_certificate(cert)

    def test_reproducible_assurance_rejected(self):
        cert = self._make_cert()
        cert.assurance = "reproducible"
        with pytest.raises(ValueError, match="assurance"):
            verify_rayleigh_certificate(cert)

    def test_forged_theorem_rejected(self):
        cert = self._make_cert()
        cert.theorem = "Fake theorem: E0 >= 0 always."
        with pytest.raises(ValueError, match="theorem"):
            verify_rayleigh_certificate(cert)

    def test_downgraded_backend_rejected(self):
        cert = self._make_cert()
        cert.backend = "numpy-float (no certified rounding; install python-flint)"
        with pytest.raises(ValueError, match="backend"):
            verify_rayleigh_certificate(cert)

    def test_from_dict_missing_assurance_raises(self):
        cert = self._make_cert()
        d = cert.to_dict()
        del d["assurance"]
        with pytest.raises(ValueError, match="assurance"):
            RayleighCertificate.from_dict(d)

    def test_from_dict_no_default_assurance_upgrade(self):
        # from_dict must NOT silently upgrade missing assurance to "rigorous".
        cert = self._make_cert()
        d = cert.to_dict()
        del d["assurance"]
        try:
            RayleighCertificate.from_dict(d)
            assert False, "Expected ValueError for missing assurance"
        except ValueError:
            pass  # correct

    def test_extra_fields_do_not_bypass_checks(self):
        # Extra fields must not confuse the verifier into accepting a bad cert.
        cert = self._make_cert()
        cert.assurance = "heuristic"
        # No bypass: must still reject
        with pytest.raises(ValueError):
            verify_rayleigh_certificate(cert)


# ── verify_rayleigh_certificate: non-finite and interval-ordering edge cases ──

class TestVerifyRayleighCertEdgeCases:
    """Cover defensive branches in verify_rayleigh_certificate."""

    def _make_cert(self):
        # Rayleigh quotient = 1.0 (non-zero, gives non-trivial interval width)
        H   = np.diag([1.0, 2.0])
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(H, psi)
        verify_rayleigh_certificate(cert)
        return cert

    def test_non_finite_stored_upper_raises(self):
        cert = self._make_cert()
        cert.upper = float("nan")
        with pytest.raises(ValueError, match="finite"):
            verify_rayleigh_certificate(cert)

    def test_interval_ordering_violated_raises(self):
        cert = self._make_cert()
        # Move lower ABOVE midpoint so ordering lower<=mid fails,
        # but keep upper unchanged so the core bound comparison passes first.
        cert.lower = cert.midpoint + 1.0
        with pytest.raises(ValueError, match="lower|upper|ordering|interval"):
            verify_rayleigh_certificate(cert)

    def test_interval_radius_too_small_raises(self):
        cert = self._make_cert()
        # Interval around 1.0 has width ~2^-128 ≈ 3e-39; set radius far below that.
        cert.radius = 1e-300
        with pytest.raises(ValueError, match="radius"):
            verify_rayleigh_certificate(cert)


# ── _htf_version and _git_commit fallback paths ───────────────────────────────

class TestHelperFallbacks:
    """Cover exception-handler branches in _htf_version and _git_commit."""

    def test_htf_version_returns_toml_when_metadata_differs(self, monkeypatch):
        import importlib.metadata as meta

        import htf.rayleigh_cert as rc
        # metadata.version returns a stale version; pyproject.toml has the real one
        monkeypatch.setattr(meta, "version", lambda _: "0.0.0-stale")
        ver = rc._htf_version()
        # _htf_version detects mismatch and returns pyproject.toml value
        assert ver != "0.0.0-stale"
        assert "." in ver

    def test_htf_version_fallback_when_metadata_raises(self, monkeypatch):
        import importlib.metadata as meta

        import htf.rayleigh_cert as rc

        def _raise(_pkg):
            raise Exception("package not installed")

        monkeypatch.setattr(meta, "version", _raise)
        ver = rc._htf_version()
        # outer-except fallback: reads pyproject.toml directly
        assert "." in ver
        assert ver != "unknown"

    def test_htf_version_returns_unknown_when_all_fail(self, monkeypatch, tmp_path):
        import importlib.metadata as meta

        import htf.rayleigh_cert as rc

        def _raise(_pkg):
            raise Exception("package not installed")

        monkeypatch.setattr(meta, "version", _raise)
        # Redirect __file__ so the pyproject.toml read also fails
        monkeypatch.setattr(rc, "__file__", str(tmp_path / "fake_sub" / "rayleigh_cert.py"))
        ver = rc._htf_version()
        assert ver == "unknown"

    def test_git_commit_returns_empty_on_failure(self, monkeypatch):
        import subprocess

        import htf.rayleigh_cert as rc

        def _raise(*_a, **_kw):
            raise FileNotFoundError("git not found")

        monkeypatch.setattr(subprocess, "run", _raise)
        commit = rc._git_commit()
        assert commit == ""
