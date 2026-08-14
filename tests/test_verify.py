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
        # Old-format cert (no assurance field) with numpy-float backend must also fail.
        from htf.verify import verify_from_dict
        d = self._heuristic_full_dict()
        d.pop("assurance", None)
        # backend still contains "numpy"
        result = verify_from_dict(d)
        assert result["verified"] is False

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
