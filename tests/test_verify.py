"""Tests for htf/verify.py — independent verifier (G1 gate)."""
import json
import tempfile

import numpy as np
import pytest

from htf.rayleigh_cert import rayleigh_certificate
from htf.verify import verify_file, verify_from_dict

# ─────────────────── fixtures ────────────────────────────────────────────────

@pytest.fixture
def diag2():
    return np.diag([0.0, 1.0])


@pytest.fixture
def good_full_dict(diag2):
    psi = np.array([1.0, 0.0])
    cert = rayleigh_certificate(diag2, psi)
    return cert.to_full_dict()


# ─────────────────── verify_from_dict: happy path ────────────────────────────

class TestVerifyFromDictHappy:

    def test_returns_dict(self, good_full_dict):
        result = verify_from_dict(good_full_dict)
        assert isinstance(result, dict)

    def test_verified_true(self, good_full_dict):
        result = verify_from_dict(good_full_dict)
        assert result["verified"] is True

    def test_digest_match_true(self, good_full_dict):
        result = verify_from_dict(good_full_dict)
        assert result["digest_match"] is True

    def test_recomputed_upper_close_to_stored(self, good_full_dict):
        result = verify_from_dict(good_full_dict)
        stored = result["stored_upper"]
        recomputed = result["recomputed_upper"]
        assert recomputed is not None
        assert abs(recomputed - stored) < 1e-12

    def test_message_contains_pass(self, good_full_dict):
        result = verify_from_dict(good_full_dict)
        assert "PASS" in result["message"]

    def test_end_to_end_tfim(self):
        from htf.variational import transverse_ising_ham
        H = transverse_ising_ham(4, J=1.0, h=0.5)
        true_E0 = float(np.linalg.eigvalsh(H)[0])
        rng = np.random.default_rng(7)
        psi = rng.standard_normal(H.shape[0])
        cert = rayleigh_certificate(H, psi)
        result = verify_from_dict(cert.to_full_dict())
        assert result["verified"] is True
        assert result["stored_upper"] >= true_E0 - 1e-9


# ─────────────────── verify_from_dict: tamper detection ──────────────────────

class TestVerifyFromDictTampering:

    def test_tampered_digest_fails(self, good_full_dict):
        good_full_dict["input_digest"] = "0" * 64
        result = verify_from_dict(good_full_dict)
        assert result["verified"] is False
        assert result["digest_match"] is False
        assert "FAIL" in result["message"]
        assert "digest" in result["message"].lower()

    def test_tampered_upper_bound_fails(self, good_full_dict):
        # Force a lower stored upper — recomputed will exceed it
        good_full_dict["interval"]["upper"] = -999.0
        result = verify_from_dict(good_full_dict)
        # digest will mismatch because canonical H/psi unchanged; upper mismatch
        # Either digest or upper check fires
        assert result["verified"] is False
        assert "FAIL" in result["message"]

    def test_tampered_H_in_canonical_fails(self, good_full_dict):
        # Replace H with a different matrix — digest will mismatch
        n = len(good_full_dict["canonical"]["H"])
        good_full_dict["canonical"]["H"] = [[2.0 if i == j else 0.0
                                              for j in range(n)]
                                             for i in range(n)]
        result = verify_from_dict(good_full_dict)
        assert result["verified"] is False
        assert result["digest_match"] is False

    def test_tampered_psi_in_canonical_fails(self, good_full_dict):
        psi = good_full_dict["canonical"]["psi"]
        psi[0] = psi[0] + 1.0  # modify first element
        result = verify_from_dict(good_full_dict)
        assert result["verified"] is False
        assert result["digest_match"] is False


# ─────────────────── verify_from_dict: structural errors ─────────────────────

class TestVerifyFromDictErrors:

    def test_raises_on_missing_canonical(self, good_full_dict):
        del good_full_dict["canonical"]
        with pytest.raises(ValueError, match="canonical"):
            verify_from_dict(good_full_dict)

    def test_raises_on_missing_input_digest(self, good_full_dict):
        del good_full_dict["input_digest"]
        with pytest.raises(ValueError, match="input_digest"):
            verify_from_dict(good_full_dict)

    def test_raises_on_missing_interval(self, good_full_dict):
        del good_full_dict["interval"]
        with pytest.raises(ValueError, match="interval"):
            verify_from_dict(good_full_dict)

    def test_raises_on_missing_H_in_canonical(self, good_full_dict):
        del good_full_dict["canonical"]["H"]
        with pytest.raises(ValueError, match="H"):
            verify_from_dict(good_full_dict)

    def test_raises_on_missing_psi_in_canonical(self, good_full_dict):
        del good_full_dict["canonical"]["psi"]
        with pytest.raises(ValueError, match="psi"):
            verify_from_dict(good_full_dict)


# ── P1-A: semantic field mutation matrix ──────────────────────────────────────

class TestVerifyMutationMatrix:
    """Every individually-tampered semantic field must produce verified=False.

    Acceptance criterion (P1-A): claim / theorem / interval.lower /
    interval.radius / backend — each tampered in isolation → FAIL.
    """

    def test_tampered_claim_fails(self, good_full_dict):
        good_full_dict["claim"] = "E0 ≤ -9999.0  [tampered]"
        result = verify_from_dict(good_full_dict)
        assert result["verified"] is False
        assert "claim" in result["message"].lower()

    def test_tampered_theorem_fails(self, good_full_dict):
        good_full_dict["theorem"] = "Some other theorem that was injected"
        result = verify_from_dict(good_full_dict)
        assert result["verified"] is False
        assert "theorem" in result["message"].lower()

    def test_tampered_lower_fails(self, good_full_dict):
        good_full_dict["interval"]["lower"] = -1e6
        result = verify_from_dict(good_full_dict)
        assert result["verified"] is False
        assert "lower" in result["message"].lower()

    def test_tampered_radius_inflated_fails(self, good_full_dict):
        good_full_dict["interval"]["radius"] = 1e10
        result = verify_from_dict(good_full_dict)
        assert result["verified"] is False
        assert "radius" in result["message"].lower()

    def test_tampered_backend_fails(self, good_full_dict):
        good_full_dict["backend"] = "flint-arb/prec=9999 (injected)"
        result = verify_from_dict(good_full_dict)
        assert result["verified"] is False
        assert "backend" in result["message"].lower()

    def test_all_fields_untampered_still_passes(self, good_full_dict):
        result = verify_from_dict(good_full_dict)
        assert result["verified"] is True


# ─────────────────── verify_file ─────────────────────────────────────────────

class TestVerifyFile:

    def test_verifies_written_json(self, good_full_dict):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(good_full_dict, f)
            path = f.name
        result = verify_file(path)
        assert result["verified"] is True

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            verify_file("/nonexistent/path/cert.json")

    def test_invalid_json_raises(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("NOT VALID JSON {{{")
            path = f.name
        import json as _json
        with pytest.raises(_json.JSONDecodeError):
            verify_file(path)


# ─────────────────── CLI main ─────────────────────────────────────────────────

class TestVerifyMain:

    def test_main_exits_0_on_valid_cert(self, good_full_dict, capsys):
        import json as _json

        from htf.verify import main

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            _json.dump(good_full_dict, f)
            path = f.name

        with pytest.raises(SystemExit) as exc:
            main([path])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        parsed = _json.loads(out)
        assert parsed["verified"] is True

    def test_main_exits_1_on_tampered_cert(self, good_full_dict, capsys):
        import json as _json

        from htf.verify import main

        good_full_dict["input_digest"] = "0" * 64
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            _json.dump(good_full_dict, f)
            path = f.name

        with pytest.raises(SystemExit) as exc:
            main([path])
        assert exc.value.code == 1
        out = capsys.readouterr().out
        parsed = _json.loads(out)
        assert parsed["verified"] is False

    def test_main_exits_2_on_missing_file(self, capsys):
        from htf.verify import main
        with pytest.raises(SystemExit) as exc:
            main(["/does/not/exist/cert.json"])
        assert exc.value.code == 2

    def test_main_exits_2_with_no_args(self, capsys):
        from htf.verify import main
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 2

    def test_main_exits_2_with_help(self, capsys):
        from htf.verify import main
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 2


# ── non-rigorous cert rejection (P0-F1 regression, P2-A) ─────────────────────

class TestVerifyRejectsNonRigorous:
    """verify_from_dict() must reject certs that are not rigorous (F-1 regression).

    A heuristic or reproducible cert has radius=0.0 and a numpy-float backend.
    Accepting it would be a trivial pass (recomputed == stored), not independent
    verification.
    """

    def _heuristic_full_dict(self):
        import numpy as np

        from htf.rayleigh_cert import rayleigh_estimate
        H = np.diag([1.0, 2.0])
        psi = np.array([1.0, 0.0])
        est = rayleigh_estimate(H, psi)
        return est.to_full_dict()

    def test_heuristic_cert_returns_verified_false(self):
        from htf.verify import verify_from_dict
        d = self._heuristic_full_dict()
        result = verify_from_dict(d)
        assert result["verified"] is False

    def test_heuristic_cert_message_mentions_assurance(self):
        from htf.verify import verify_from_dict
        d = self._heuristic_full_dict()
        result = verify_from_dict(d)
        assert "assurance" in result["message"] or "heuristic" in result["message"]

    def test_numpy_backend_without_assurance_field_rejected(self):
        # Old-format cert (no assurance field) must be rejected.
        # After HTF-05 M3 fix: "assurance" is required; missing field raises ValueError.
        import pytest

        from htf.verify import verify_from_dict
        d = self._heuristic_full_dict()
        d.pop("assurance", None)
        with pytest.raises(ValueError, match="assurance"):
            verify_from_dict(d)

    def test_rigorous_cert_still_passes(self):
        import numpy as np

        from htf.rayleigh_cert import rayleigh_certificate
        from htf.verify import verify_from_dict
        H = np.diag([1.0, 2.0])
        psi = np.array([1.0, 0.0])
        cert = rayleigh_certificate(H, psi)
        d = cert.to_full_dict()
        result = verify_from_dict(d)
        assert result["verified"] is True


# ── C3 full independence: mpmath cross-check ─────────────────────────────────

class TestMpmathCrossCheck:
    """_mpmath_rayleigh provides a third independent arithmetic path."""

    def test_mpmath_rayleigh_real_diagonal(self):
        import numpy as np

        from htf._rayleigh_primitives import _mpmath_rayleigh
        H   = np.diag([0.0, 1.0, 2.0]).astype(np.float64)
        psi = np.array([1.0, 0.0, 0.0])
        lo, up, _, label = _mpmath_rayleigh(H, psi)
        assert lo <= 0.0 <= up
        assert "mpmath" in label
        assert up - lo < 1e-14

    def test_mpmath_rayleigh_non_trivial_state(self):
        import numpy as np

        from htf._rayleigh_primitives import _mpmath_rayleigh
        H   = np.diag([0.0, 1.0, 2.0]).astype(np.float64)
        psi = np.array([1.0, 1.0, 1.0])
        lo, up, _, _ = _mpmath_rayleigh(H, psi)
        # true value = (0+1+2)/3 = 1.0
        assert lo <= 1.0 <= up
        assert abs((lo + up) / 2 - 1.0) < 1e-14

    def test_mpmath_rayleigh_complex_hermitian(self):
        import numpy as np

        from htf._rayleigh_primitives import _mpmath_rayleigh
        H   = np.array([[1.0, 1j], [-1j, 1.0]], dtype=np.complex128)
        psi = np.array([1.0, 0.0], dtype=np.complex128)
        lo, up, _, label = _mpmath_rayleigh(H, psi)
        # true value = 1.0
        assert lo <= 1.0 <= up
        assert "mpmath" in label

    def test_mpmath_consistent_with_arb(self):
        """mpmath and flint-arb should agree to within loose tolerance."""
        import numpy as np

        from htf._rayleigh_primitives import _arb_rayleigh, _mpmath_rayleigh
        H   = np.diag([0.5, 1.5, 2.5]).astype(np.float64)
        psi = np.array([1.0, 2.0, 3.0])
        _, arb_upper, _, _ = _arb_rayleigh(H, psi)
        mp_lower, mp_upper, _, _ = _mpmath_rayleigh(H, psi)
        # flint upper bound must be >= mpmath lower estimate (minus tiny tolerance)
        assert arb_upper >= mp_lower - 1e-6

    def test_verify_result_includes_cross_check_field(self):
        import numpy as np

        from htf.rayleigh_cert import rayleigh_certificate
        from htf.verify import verify_from_dict
        H   = np.diag([1.0, 3.0]).astype(np.float64)
        psi = np.array([1.0, 0.0])
        d   = rayleigh_certificate(H, psi).to_full_dict()
        result = verify_from_dict(d)
        assert result["verified"] is True
        assert "cross_check" in result
        assert result["cross_check"] is not None

    def test_verify_cross_check_passes_for_rigorous_cert(self):
        import numpy as np

        from htf.rayleigh_cert import rayleigh_certificate
        from htf.verify import verify_from_dict
        H   = np.diag([0.0, 1.0, 2.0]).astype(np.float64)
        psi = np.array([1.0, 1.0, 0.0])
        d   = rayleigh_certificate(H, psi).to_full_dict()
        result = verify_from_dict(d)
        assert result["verified"] is True
        # cross_check should be PASS or skipped, never a failure string
        cc = result.get("cross_check", "")
        assert "PASS" in cc or "skipped" in cc

    def test_mpmath_extra_prec_parameter(self):
        """extra_prec shifts the working precision."""
        import numpy as np

        from htf._rayleigh_primitives import _mpmath_rayleigh
        H   = np.diag([1.0, 2.0]).astype(np.float64)
        psi = np.array([1.0, 0.0])
        _, _, _, label64  = _mpmath_rayleigh(H, psi, extra_prec=64)
        _, _, _, label256 = _mpmath_rayleigh(H, psi, extra_prec=256)
        assert "192" in label64   # 128 + 64
        assert "384" in label256  # 128 + 256


# ── coverage gap tests ────────────────────────────────────────────────────────

class TestVerifyCoverageGaps:
    """Targeted tests for previously uncovered branches in verify.py."""

    def _good_full_dict(self):
        from htf.rayleigh_cert import rayleigh_certificate
        H   = np.diag([1.0, 2.0]).astype(np.float64)
        psi = np.array([1.0, 0.0])
        return rayleigh_certificate(H, psi).to_full_dict()

    def test_wrong_schema_version_raises(self):
        from htf.verify import verify_from_dict
        d = self._good_full_dict()
        d["schema_version"] = "rayleigh-cert/v0"
        with pytest.raises(ValueError, match="schema_version"):
            verify_from_dict(d)

    def test_stored_upper_nan_returns_fail(self):
        import math

        from htf.verify import verify_from_dict
        d = self._good_full_dict()
        d["interval"]["upper"] = float("nan")
        # claim must also encode the bad upper so the claim check passes
        d["claim"] = f"E0 ≤ {float('nan'):.17g}  [Rayleigh-Ritz upper bound on ground-state energy]"
        result = verify_from_dict(d)
        assert result["verified"] is False
        assert not math.isfinite(result["stored_upper"])

    def test_mpmath_cross_check_skipped_when_unavailable(self, monkeypatch):
        import sys

        from htf.verify import verify_from_dict
        # Temporarily hide mpmath from the import system
        original = sys.modules.get("mpmath")
        sys.modules["mpmath"] = None  # type: ignore[assignment]
        try:
            d = self._good_full_dict()
            result = verify_from_dict(d)
        finally:
            if original is None:
                del sys.modules["mpmath"]
            else:
                sys.modules["mpmath"] = original
        assert result["verified"] is True
        assert result["cross_check"] == "skipped (mpmath not installed)"

    def test_verify_file_nonexistent_raises(self, tmp_path):
        from htf.verify import verify_file
        with pytest.raises(FileNotFoundError):
            verify_file(tmp_path / "no_such_cert.json")

    def test_verify_file_roundtrip(self, tmp_path):
        from htf.verify import verify_file
        d = self._good_full_dict()
        p = tmp_path / "cert.json"
        p.write_text(json.dumps(d))
        result = verify_file(str(p))
        assert result["verified"] is True

    def test_main_verified_exits_0(self, tmp_path, capsys):
        import sys

        from htf.verify import main
        d = self._good_full_dict()
        p = tmp_path / "cert.json"
        p.write_text(json.dumps(d))
        with pytest.raises(SystemExit) as exc:
            main([str(p)])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert '"verified": true' in out

    def test_main_none_argv_reads_sysargv(self, monkeypatch, tmp_path, capsys):
        import sys

        from htf.verify import main
        d = self._good_full_dict()
        p = tmp_path / "cert.json"
        p.write_text(json.dumps(d))
        monkeypatch.setattr(sys, "argv", ["htf-verify", str(p)])
        with pytest.raises(SystemExit) as exc:
            main(None)
        assert exc.value.code == 0

    def test_verify_from_dict_raises_without_flint(self, monkeypatch):
        import sys

        from htf.verify import verify_from_dict
        d = self._good_full_dict()
        monkeypatch.setitem(sys.modules, "flint", None)
        with pytest.raises(ImportError, match="python-flint"):
            verify_from_dict(d)
