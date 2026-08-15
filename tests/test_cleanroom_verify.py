"""Tests for htf._cleanroom_verify — exact-rational Gate-A regression oracle.

The clean-room verifier uses Fraction arithmetic (independent of Arb/Acb) to
cross-check real RayleighCertificate objects produced by rayleigh_certificate().
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from htf._cleanroom_verify import (
    Rejected,
    _anchor_inputs,
    _check_anchors_and_second_path,
    _check_adversarial_guards,
    _check_digest_structure,
    _check_diagnostic_float_failure,
    canonical_digest,
    exact_rayleigh,
    make_modeled_certificate,
    outward_binary64,
    verify_modeled_certificate,
)
from htf.rayleigh_cert import rayleigh_certificate


# ─────────────────────── clean-room self-checks ──────────────────────────────

class TestCleanRoomSelfChecks:
    def test_all_checks_passed(self):
        """Full clean-room suite passes (mirrors main())."""
        from htf._cleanroom_verify import main
        main()  # raises AssertionError / Rejected on any failure

    def test_anchors_and_second_path(self):
        _check_anchors_and_second_path()

    def test_digest_structure(self):
        _check_digest_structure()

    def test_adversarial_guards(self):
        _check_adversarial_guards()

    def test_diagnostic_float_failure(self):
        _check_diagnostic_float_failure()


# ─────────────────────── cross-check against real certs ──────────────────────

class TestCrossCheckRealCerts:
    """Verify that real RayleighCertificate objects satisfy exact-rational bounds."""

    def _verify_cert_upper(self, H: np.ndarray, psi: np.ndarray) -> None:
        """Real cert upper must be >= exact Rayleigh quotient (Fraction arithmetic)."""
        cert = rayleigh_certificate(H, psi)
        q = exact_rayleigh(H, psi)
        assert Fraction.from_float(cert.upper) >= q, (
            f"cert.upper={cert.upper!r} < exact q={float(q)!r}"
        )

    def _verify_cert_digest(self, H: np.ndarray, psi: np.ndarray) -> None:
        """Real cert digest must match clean-room digest."""
        cert = rayleigh_certificate(H, psi)
        assert cert.input_digest == canonical_digest(H, psi)

    def test_anchor1_upper_bound(self):
        H, psi, _ = _anchor_inputs()[0]
        self._verify_cert_upper(H, psi)

    def test_anchor2_upper_bound(self):
        H, psi, _ = _anchor_inputs()[1]
        self._verify_cert_upper(H, psi)

    def test_anchor3_upper_bound(self):
        H, psi, _ = _anchor_inputs()[2]
        self._verify_cert_upper(H, psi)

    def test_anchor4_upper_bound(self):
        H, psi, _ = _anchor_inputs()[3]
        self._verify_cert_upper(H, psi)

    def test_anchor1_digest(self):
        H, psi, _ = _anchor_inputs()[0]
        self._verify_cert_digest(H, psi)

    def test_anchor2_digest(self):
        H, psi, _ = _anchor_inputs()[1]
        self._verify_cert_digest(H, psi)

    def test_anchor3_digest(self):
        H, psi, _ = _anchor_inputs()[2]
        self._verify_cert_digest(H, psi)

    def test_anchor4_digest(self):
        H, psi, _ = _anchor_inputs()[3]
        self._verify_cert_digest(H, psi)

    def test_random_5x5_upper_bound(self):
        rng = np.random.default_rng(7)
        A = rng.standard_normal((5, 5))
        H = (A + A.T) / 2   # symmetric
        psi = rng.standard_normal(5)
        H = H.astype(np.float64)
        psi = psi.astype(np.float64)
        self._verify_cert_upper(H, psi)

    def test_complex_3x3_upper_bound(self):
        rng = np.random.default_rng(13)
        A = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
        H = (A + A.conj().T) / 2   # Hermitian
        psi = (rng.standard_normal(3) + 1j * rng.standard_normal(3)).astype(np.complex128)
        H = H.astype(np.complex128)
        self._verify_cert_upper(H, psi)


# ─────────────────────── modeled certificate checks ──────────────────────────

class TestModeledCertificate:
    def test_make_and_verify_anchor1(self):
        H, psi, _ = _anchor_inputs()[0]
        cert = make_modeled_certificate(H, psi)
        verify_modeled_certificate(cert, H, psi)

    def test_make_and_verify_anchor3(self):
        H, psi, _ = _anchor_inputs()[2]
        cert = make_modeled_certificate(H, psi)
        verify_modeled_certificate(cert, H, psi)

    def test_reject_non_hermitian(self):
        H = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        psi = np.array([1.0, 0.0], dtype=np.float64)
        with pytest.raises(Rejected):
            exact_rayleigh(H, psi)

    def test_reject_non_finite_psi(self):
        H = np.eye(2, dtype=np.float64)
        psi = np.array([1.0, math.inf], dtype=np.float64)
        with pytest.raises(Rejected):
            exact_rayleigh(H, psi)

    def test_outward_binary64_contains_exact(self):
        q = Fraction(1, 3)
        lower, upper = outward_binary64(q)
        assert Fraction.from_float(lower) <= q <= Fraction.from_float(upper)

    def test_exact_rayleigh_ground_state(self):
        H = np.diag(np.array([0.0, 1.0, 2.0], dtype=np.float64))
        psi = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        assert exact_rayleigh(H, psi) == Fraction(0)

    def test_exact_rayleigh_uniform_superposition(self):
        H = np.diag(np.array([0.0, 1.0, 2.0], dtype=np.float64))
        psi = np.array([1.0, 1.0, 1.0], dtype=np.float64) / math.sqrt(3.0)
        q = exact_rayleigh(H, psi)
        assert q == Fraction(1)


class TestHTF06GateARegressions:
    """B3/B4/B5 fixes from HTF-06 Gate-A independent review (2026-08-15)."""

    def test_dbmax_input_produces_certificate(self):
        # B3: H=[[DBL_MAX]], psi=[1] satisfies A1-A5.  After the conditional
        # nextafter fix, endpoints stay finite (DBL_MAX) and a certificate is
        # produced instead of being rejected with a ValueError.
        pytest.importorskip("flint")
        from htf.rayleigh_cert import rayleigh_certificate, verify_rayleigh_certificate
        H = np.array([[np.finfo(np.float64).max]], dtype=np.float64)
        psi = np.array([1.0], dtype=np.float64)
        cert = rayleigh_certificate(H, psi)
        assert math.isfinite(cert.upper)
        assert math.isfinite(cert.lower)
        verify_rayleigh_certificate(cert)
        assert cert.verified

    def test_large_input_midpoint_finite(self):
        # B4: inputs near 1e308 used to produce midpoint=inf via (lo+up)/2.
        # The stable formula lo + (up-lo)/2 keeps it finite.
        pytest.importorskip("flint")
        from htf.rayleigh_cert import rayleigh_certificate, verify_rayleigh_certificate
        H = np.array([[1e308]], dtype=np.float64)
        psi = np.array([1.0], dtype=np.float64)
        cert = rayleigh_certificate(H, psi)
        assert math.isfinite(cert.midpoint), f"midpoint={cert.midpoint}"
        assert math.isfinite(cert.radius),   f"radius={cert.radius}"
        verify_rayleigh_certificate(cert)
        assert cert.verified

    def test_assumption_text_no_float_norm_claim(self):
        # B5: the assumption text must not claim ⟨ψ|ψ⟩ > 0 was float-computed.
        # For a subnormal psi the float inner product underflows to 0.
        pytest.importorskip("flint")
        from htf.rayleigh_cert import rayleigh_certificate
        H = np.diag([0.0, 1.0])
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(H, psi)
        for a in cert.assumptions:
            assert "⟨ψ|ψ⟩ > 0 exactly" not in a, (
                f"Assumption claims float-computed norm: {a!r}"
            )
        assert any("non-zero" in a.lower() for a in cert.assumptions)

    def test_independent_audit_script(self):
        # Runs the clean-room Fraction-arithmetic audit delivered with Gate-A review.
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from independent_rayleigh_audit import run_checks
        run_checks()  # raises AssertionError / AuditFailure on any mismatch
