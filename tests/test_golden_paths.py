"""P2-B: Three certified Golden Paths.

Each path verifies the complete pipeline end-to-end:
  input → rayleigh_certificate/adapter → to_full_dict → verify_from_dict → verified=True

GP-1  Dense Hamiltonian  → rayleigh_certificate()        → verify_from_dict()
GP-2  quimb-style MPS   → rayleigh_from_quimb_mps()     → verify_from_dict()
GP-3  TeNPy-style MPS   → rayleigh_from_tenpy_mps()     → verify_from_dict()

Metrics checked for each path:
  - verified == True
  - assurance == "rigorous"
  - stored_upper >= true_E0 (Rayleigh-Ritz bound holds)
  - JSON cert size < 256 KB
  - verify_from_dict completes in < 2 s (sanity check only)
  - tampered cert returns verified == False (zero false-accept regression)
"""
from __future__ import annotations

import json
import time

import numpy as np
import pytest

# ── helpers ───────────────────────────────────────────────────────────────────

def _tfim_ham(n: int, J: float = 1.0, h: float = 0.5) -> np.ndarray:
    from htf.variational import transverse_ising_ham
    return transverse_ising_ham(n, J=J, h=h)


def _true_E0(H: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(H)[0])


def _tamper(d: dict) -> dict:
    d = json.loads(json.dumps(d))
    d["interval"]["upper"] = d["interval"]["upper"] - 100.0
    return d


class _PlainMPS:
    """Minimal duck-type accepted by both quimb and tenpy adapters."""
    def __init__(self, vec):
        self._vec = np.asarray(vec, dtype=float)

    def to_dense(self):
        return self._vec.reshape(-1, 1)


# ── GP-1: Dense Hamiltonian ───────────────────────────────────────────────────

class TestGoldenPath1Dense:
    """GP-1: dense H, random trial state, full certified pipeline."""

    @pytest.fixture
    def pipeline(self):
        from htf.rayleigh_cert import rayleigh_certificate
        from htf.verify import verify_from_dict
        H = _tfim_ham(4)
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(H.shape[0])
        cert = rayleigh_certificate(H, psi)
        full = cert.to_full_dict()
        t0 = time.monotonic()
        result = verify_from_dict(full)
        elapsed = time.monotonic() - t0
        return {"cert": cert, "full": full, "result": result,
                "elapsed": elapsed, "H": H}

    def test_verified(self, pipeline):
        assert pipeline["result"]["verified"] is True

    def test_assurance_rigorous(self, pipeline):
        assert pipeline["cert"].assurance == "rigorous"

    def test_upper_bounds_true_E0(self, pipeline):
        assert pipeline["cert"].upper >= _true_E0(pipeline["H"]) - 1e-9

    def test_cert_json_size_reasonable(self, pipeline):
        size = len(json.dumps(pipeline["full"]).encode())
        assert size < 256 * 1024, f"cert JSON is {size} bytes, expected < 256 KB"

    def test_verify_time_reasonable(self, pipeline):
        assert pipeline["elapsed"] < 2.0

    def test_tampered_cert_rejected(self, pipeline):
        from htf.verify import verify_from_dict
        assert verify_from_dict(_tamper(pipeline["full"]))["verified"] is False

    def test_message_contains_pass(self, pipeline):
        assert "PASS" in pipeline["result"]["message"]

    def test_digest_match(self, pipeline):
        assert pipeline["result"]["digest_match"] is True

    def test_recomputed_upper_leq_stored(self, pipeline):
        r = pipeline["result"]
        assert r["recomputed_upper"] <= r["stored_upper"] + 1e-15

    def test_ground_state_tight_bound(self):
        from htf.rayleigh_cert import rayleigh_certificate
        from htf.verify import verify_from_dict
        H = _tfim_ham(4)
        _, vecs = np.linalg.eigh(H)
        cert = rayleigh_certificate(H, vecs[:, 0])
        result = verify_from_dict(cert.to_full_dict())
        assert result["verified"] is True
        assert cert.upper <= _true_E0(H) + 1e-6


# ── GP-2: quimb-style MPS ────────────────────────────────────────────────────

class TestGoldenPath2Quimb:
    """GP-2: quimb MPS adapter → full certified pipeline."""

    @pytest.fixture
    def pipeline(self):
        from htf.adapters.quimb_adapter import rayleigh_from_quimb_mps
        from htf.verify import verify_from_dict
        H = np.diag([0.0, 1.0, 1.0, 2.0])
        cert = rayleigh_from_quimb_mps(_PlainMPS([1.0, 0.0, 0.0, 0.0]), H)
        full = cert.to_full_dict()
        t0 = time.monotonic()
        result = verify_from_dict(full)
        elapsed = time.monotonic() - t0
        return {"cert": cert, "full": full, "result": result,
                "elapsed": elapsed, "H": H}

    def test_verified(self, pipeline):
        assert pipeline["result"]["verified"] is True

    def test_assurance_rigorous(self, pipeline):
        assert pipeline["cert"].assurance == "rigorous"

    def test_upper_bounds_true_E0(self, pipeline):
        assert pipeline["cert"].upper >= _true_E0(pipeline["H"]) - 1e-9

    def test_cert_json_size_reasonable(self, pipeline):
        assert len(json.dumps(pipeline["full"]).encode()) < 256 * 1024

    def test_verify_time_reasonable(self, pipeline):
        assert pipeline["elapsed"] < 2.0

    def test_tampered_cert_rejected(self, pipeline):
        from htf.verify import verify_from_dict
        assert verify_from_dict(_tamper(pipeline["full"]))["verified"] is False

    def test_adapter_note_in_cert(self, pipeline):
        assert "quimb" in pipeline["cert"].notes.lower()

    def test_backend_is_flint(self, pipeline):
        assert "flint" in pipeline["cert"].backend.lower()

    def test_tfim_n4_ground_state_tight(self):
        from htf.adapters.quimb_adapter import rayleigh_from_quimb_mps
        from htf.verify import verify_from_dict
        H = _tfim_ham(4)
        _, vecs = np.linalg.eigh(H)
        cert = rayleigh_from_quimb_mps(_PlainMPS(vecs[:, 0]), H)
        result = verify_from_dict(cert.to_full_dict())
        assert result["verified"] is True
        assert cert.upper <= _true_E0(H) + 1e-6


# ── GP-3: TeNPy-style MPS ────────────────────────────────────────────────────

class TestGoldenPath3TeNPy:
    """GP-3: TeNPy MPS adapter → full certified pipeline."""

    @pytest.fixture
    def pipeline(self):
        from htf.adapters.tenpy_adapter import rayleigh_from_tenpy_mps
        from htf.verify import verify_from_dict
        H = np.diag([0.0, 1.0, 1.0, 2.0])
        cert = rayleigh_from_tenpy_mps(_PlainMPS([1.0, 0.0, 0.0, 0.0]), H)
        full = cert.to_full_dict()
        t0 = time.monotonic()
        result = verify_from_dict(full)
        elapsed = time.monotonic() - t0
        return {"cert": cert, "full": full, "result": result,
                "elapsed": elapsed, "H": H}

    def test_verified(self, pipeline):
        assert pipeline["result"]["verified"] is True

    def test_assurance_rigorous(self, pipeline):
        assert pipeline["cert"].assurance == "rigorous"

    def test_upper_bounds_true_E0(self, pipeline):
        assert pipeline["cert"].upper >= _true_E0(pipeline["H"]) - 1e-9

    def test_cert_json_size_reasonable(self, pipeline):
        assert len(json.dumps(pipeline["full"]).encode()) < 256 * 1024

    def test_verify_time_reasonable(self, pipeline):
        assert pipeline["elapsed"] < 2.0

    def test_tampered_cert_rejected(self, pipeline):
        from htf.verify import verify_from_dict
        assert verify_from_dict(_tamper(pipeline["full"]))["verified"] is False

    def test_adapter_note_in_cert(self, pipeline):
        assert "tenpy" in pipeline["cert"].notes.lower()

    def test_tfim_n4_ground_state_tight(self):
        from htf.adapters.tenpy_adapter import rayleigh_from_tenpy_mps
        from htf.verify import verify_from_dict
        H = _tfim_ham(4)
        _, vecs = np.linalg.eigh(H)
        cert = rayleigh_from_tenpy_mps(_PlainMPS(vecs[:, 0]), H)
        result = verify_from_dict(cert.to_full_dict())
        assert result["verified"] is True
        assert cert.upper <= _true_E0(H) + 1e-6


# ── Cross-path: zero false-accept ─────────────────────────────────────────────

class TestZeroFalseAccept:
    """Any tampered cert from any golden path must be rejected."""

    def _dense_full_dict(self):
        from htf.rayleigh_cert import rayleigh_certificate
        H = np.diag([1.0, 2.0])
        psi = np.array([1.0, 0.0])
        return rayleigh_certificate(H, psi).to_full_dict()

    def test_flip_upper_rejected(self):
        from htf.verify import verify_from_dict
        d = self._dense_full_dict()
        d["interval"]["upper"] = d["interval"]["lower"] - 1.0
        assert verify_from_dict(d)["verified"] is False

    def test_flip_digest_rejected(self):
        from htf.verify import verify_from_dict
        d = self._dense_full_dict()
        d["input_digest"] = "a" * 64
        assert verify_from_dict(d)["verified"] is False

    def test_modify_H_rejected(self):
        from htf.verify import verify_from_dict
        d = self._dense_full_dict()
        d["canonical"]["H"][0][0] = d["canonical"]["H"][0][0] + 99.0
        assert verify_from_dict(d)["verified"] is False

    def test_modify_psi_rejected(self):
        from htf.verify import verify_from_dict
        d = self._dense_full_dict()
        d["canonical"]["psi"][0] = d["canonical"]["psi"][0] + 5.0
        assert verify_from_dict(d)["verified"] is False

    def test_heuristic_cert_rejected(self):
        from htf.rayleigh_cert import rayleigh_estimate
        from htf.verify import verify_from_dict
        H = np.diag([1.0, 2.0])
        psi = np.array([1.0, 0.0])
        assert verify_from_dict(rayleigh_estimate(H, psi).to_full_dict())["verified"] is False
