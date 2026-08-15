"""Tests for htf.mps_cert — factorized Rayleigh certificate (MPS/MPO)."""
import json
import math

import numpy as np
import pytest

from htf.mps import MPS, mps_from_state, random_mps
from htf.mpo import MPO, mpo_from_matrix, mpo_to_matrix
from htf.mps_cert import (
    MPS_CERT_SCHEMA,
    RayleighCertificateMPS,
    _canonical_digest_mps,
    rayleigh_certificate_mps,
    verify_rayleigh_certificate_mps,
)
from htf._rayleigh_primitives import EXPECTED_THEOREM


# ─────────────────────── helpers ─────────────────────────────────────────────

def _diag_mpo_and_mps(n: int, d: int = 2) -> tuple[MPO, MPS, float]:
    """2-site diagonal Hamiltonian with known E0=0 and ground state |00…0⟩."""
    dim = d ** n
    H = np.diag(np.arange(dim, dtype=float))
    mpo = mpo_from_matrix(H, n=n, d=d)
    psi = np.zeros(dim)
    psi[0] = 1.0
    mps = mps_from_state(psi, d=d)
    return mpo, mps, 0.0


def _uniform_mpo_and_mps(n: int, d: int = 2) -> tuple[MPO, MPS, float]:
    """Equal-superposition trial state; Rayleigh = mean of eigenvalues."""
    dim = d ** n
    H = np.diag(np.arange(dim, dtype=float))
    mpo = mpo_from_matrix(H, n=n, d=d)
    psi = np.ones(dim) / math.sqrt(dim)
    mps = mps_from_state(psi, d=d)
    expected_rq = float(np.mean(np.arange(dim)))
    return mpo, mps, expected_rq


# ─────────────────────── production ──────────────────────────────────────────

class TestProduction:
    def test_ground_state_upper_bound_holds(self):
        mpo, mps, e0 = _diag_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo)
        # assurance="reproducible" uses float64, not Arb; allow one ULP of rounding
        assert cert.rayleigh_upper >= e0 - 1e-14

    def test_upper_bound_tight_for_ground_state(self):
        mpo, mps, _ = _diag_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo)
        assert cert.rayleigh_upper < 1e-10

    def test_rayleigh_upper_geq_lower(self):
        mpo, mps, _ = _uniform_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo)
        assert cert.rayleigh_upper >= cert.rayleigh_lower

    def test_schema_version(self):
        mpo, mps, _ = _diag_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo)
        assert cert.schema_version == MPS_CERT_SCHEMA

    def test_assurance_reproducible(self):
        mpo, mps, _ = _diag_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo)
        assert cert.assurance == "reproducible"

    def test_backend_float64_mps(self):
        mpo, mps, _ = _diag_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo)
        assert cert.backend == "float64-mps"

    def test_theorem_matches_expected(self):
        mpo, mps, _ = _diag_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo)
        assert cert.theorem == EXPECTED_THEOREM

    def test_n_sites_recorded(self):
        mpo, mps, _ = _diag_mpo_and_mps(3)
        cert = rayleigh_certificate_mps(mps, mpo)
        assert cert.n_sites == 3

    def test_phys_dim_recorded(self):
        mpo, mps, _ = _diag_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo)
        assert cert.phys_dim == 2

    def test_input_digest_is_sha256_hex(self):
        mpo, mps, _ = _diag_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo)
        assert len(cert.input_digest) == 64
        assert all(c in "0123456789abcdef" for c in cert.input_digest)

    def test_notes_stored(self):
        mpo, mps, _ = _diag_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo, notes="test note")
        assert cert.notes == "test note"

    def test_n_sites_mismatch_raises(self):
        mpo, _, _ = _diag_mpo_and_mps(2)
        mps3 = random_mps(3, 2, 2, seed=0)
        with pytest.raises(ValueError, match="n_sites"):
            rayleigh_certificate_mps(mps3, mpo)

    def test_phys_dim_mismatch_raises(self):
        mpo2, _, _ = _diag_mpo_and_mps(2, d=2)
        psi3 = np.zeros(3 ** 2); psi3[0] = 1.0
        mps3 = mps_from_state(psi3, d=3)
        mpo3 = mpo_from_matrix(np.eye(9), n=2, d=3)
        # mpo2 has phys_dim=2, mps3 has phys_dim=3
        mpo2_wrong = mpo2  # n_sites=2 but phys_dim=2
        # Create a mismatch: use mpo2 (d=2) with mps (d=3)
        # mps_from_state creates d=3 MPS from d^2=9 state
        with pytest.raises(ValueError, match="phys_dim"):
            rayleigh_certificate_mps(mps3, mpo2_wrong)

    def test_zero_norm_mps_raises(self):
        mpo, _, _ = _diag_mpo_and_mps(2)
        zero_tensors = [np.zeros((1, 2, 1)), np.zeros((1, 2, 1))]
        mps_zero = MPS(zero_tensors)
        with pytest.raises(ValueError, match="zero norm"):
            rayleigh_certificate_mps(mps_zero, mpo)


# ─────────────────────── agree with dense cert ───────────────────────────────

class TestAgreesWithDense:
    def test_rayleigh_matches_dense_quotient(self):
        n, d = 2, 2
        mpo, mps, _ = _uniform_mpo_and_mps(n, d)
        cert = rayleigh_certificate_mps(mps, mpo)
        # Dense cross-check
        H_dense = mpo_to_matrix(mpo)
        psi_dense = np.array(mps.tensors[0][0, :, 0].tolist()
                             if n == 1 else
                             [sum(mps.tensors[0][0, s0, 0] * mps.tensors[1][0, s1, 0]
                                  for s0 in range(d) for s1 in range(d)
                                  if s0 * d + s1 == idx)
                              for idx in range(d ** n)])
        # Simpler: just densify via mps_to_state
        from htf.mps import mps_to_state
        psi_dense = mps_to_state(mps)
        rq_dense = float(np.real(psi_dense.conj() @ H_dense @ psi_dense) /
                         float(np.real(psi_dense.conj() @ psi_dense)))
        assert abs(cert.rayleigh_upper - rq_dense) < 1e-10

    def test_upper_bound_holds_for_random_mps(self):
        n, d, chi = 4, 2, 4
        H = np.diag(np.arange(d ** n, dtype=float))
        mpo = mpo_from_matrix(H, n=n, d=d)
        mps = random_mps(n, d, chi, seed=42)
        cert = rayleigh_certificate_mps(mps, mpo)
        # Dense upper bound check
        from htf.mps import mps_to_state
        psi = mps_to_state(mps)
        rq = float(np.real(psi.conj() @ H @ psi) / np.real(psi.conj() @ psi))
        assert cert.rayleigh_upper >= rq - 1e-12


# ─────────────────────── verification ────────────────────────────────────────

class TestVerification:
    def test_verify_succeeds(self):
        mpo, mps, _ = _diag_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo)
        result = verify_rayleigh_certificate_mps(cert)
        assert result.verified is True

    def test_verify_returns_cert(self):
        mpo, mps, _ = _diag_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo)
        returned = verify_rayleigh_certificate_mps(cert)
        assert returned is cert

    def test_verify_tampered_mps_tensor_detected(self):
        mpo, mps, _ = _diag_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo)
        # Tamper first MPS tensor
        original = cert._mps_tensors[0]
        if isinstance(original, list):
            cert._mps_tensors[0] = [[[999.0], [0.0]]]
        else:
            cert._mps_tensors[0] = {"real": [[[999.0], [0.0]]], "imag": [[[0.0], [0.0]]]}
        with pytest.raises(ValueError, match="Digest mismatch"):
            verify_rayleigh_certificate_mps(cert)

    def test_verify_tampered_mpo_tensor_detected(self):
        mpo, mps, _ = _diag_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo)
        # Tamper MPO tensor 0 by adding a list item
        cert._mpo_tensors = list(cert._mpo_tensors)
        cert._mpo_tensors[0] = cert._mpo_tensors[1]
        with pytest.raises(ValueError, match="Digest mismatch"):
            verify_rayleigh_certificate_mps(cert)

    def test_verify_tampered_upper_detected(self):
        mpo, mps, _ = _diag_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo)
        cert.rayleigh_upper = -999.0  # impossible lower value
        with pytest.raises(ValueError, match="Verification failed"):
            verify_rayleigh_certificate_mps(cert)

    def test_verify_tampered_theorem_detected(self):
        mpo, mps, _ = _diag_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo)
        cert.theorem = "tampered theorem"
        with pytest.raises(ValueError, match="Theorem has been tampered"):
            verify_rayleigh_certificate_mps(cert)

    def test_verify_rigorous_succeeds_when_implemented(self):
        """verify now supports assurance='rigorous' via Arb contractions."""
        pytest.importorskip("flint")
        mpo, mps, _ = _diag_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo, assurance="rigorous")
        result = verify_rayleigh_certificate_mps(cert)
        assert result.verified is True

    def test_verify_unknown_assurance_raises(self):
        mpo, mps, _ = _diag_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo)
        cert.assurance = "heuristic"
        with pytest.raises(ValueError, match="Unknown assurance"):
            verify_rayleigh_certificate_mps(cert)

    def test_verify_missing_tensors_raises(self):
        mpo, mps, _ = _diag_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo)
        cert._mps_tensors = []
        with pytest.raises(ValueError, match="lacks stored tensors"):
            verify_rayleigh_certificate_mps(cert)


# ─────────────────────── serialisation ───────────────────────────────────────

class TestSerialisation:
    def test_to_dict_roundtrip(self):
        mpo, mps, _ = _diag_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo)
        d = cert.to_dict()
        cert2 = RayleighCertificateMPS.from_dict(d)
        assert cert2.rayleigh_upper == cert.rayleigh_upper
        assert cert2.input_digest == cert.input_digest
        assert cert2.n_sites == cert.n_sites

    def test_to_full_dict_contains_tensors(self):
        mpo, mps, _ = _diag_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo)
        full = cert.to_full_dict()
        assert "mps_tensors" in full
        assert "mpo_tensors" in full
        assert len(full["mps_tensors"]) == 2
        assert len(full["mpo_tensors"]) == 2

    def test_to_dict_no_tensors(self):
        mpo, mps, _ = _diag_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo)
        d = cert.to_dict()
        assert "mps_tensors" not in d
        assert "mpo_tensors" not in d

    def test_from_full_dict_verify_roundtrip(self):
        mpo, mps, _ = _diag_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo)
        full = cert.to_full_dict()
        cert2 = RayleighCertificateMPS.from_dict(full)
        result = verify_rayleigh_certificate_mps(cert2)
        assert result.verified is True

    def test_to_json_parseable(self):
        mpo, mps, _ = _diag_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo)
        parsed = json.loads(cert.to_json())
        assert parsed["schema_version"] == MPS_CERT_SCHEMA

    def test_to_full_json_parseable(self):
        mpo, mps, _ = _diag_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo)
        parsed = json.loads(cert.to_full_json())
        assert "mps_tensors" in parsed


# ─────────────────────── memory scaling ──────────────────────────────────────

class TestMemoryScaling:
    def test_factorized_storage_size(self):
        """Stored tensor count is O(n*chi^2*d), not O(d^n)."""
        n, d, chi = 6, 2, 4
        H = np.eye(d ** n)
        mpo = mpo_from_matrix(H, n=n, d=d)
        mps = random_mps(n, d, chi, seed=0)
        cert = rayleigh_certificate_mps(mps, mpo)
        full = cert.to_full_dict()

        # Count total float64 values stored
        def _count_floats(obj) -> int:
            if isinstance(obj, (int, float)):
                return 1
            if isinstance(obj, list):
                return sum(_count_floats(x) for x in obj)
            if isinstance(obj, dict):
                return sum(_count_floats(v) for v in obj.values())
            return 0

        stored_floats = _count_floats(full["mps_tensors"]) + \
                        _count_floats(full["mpo_tensors"])
        dense_floats = d ** (2 * n) + d ** n  # H + psi
        assert stored_floats < dense_floats


# ─────────────────────── digest ──────────────────────────────────────────────

class TestDigest:
    def test_digest_deterministic(self):
        mpo, mps, _ = _diag_mpo_and_mps(2)
        d1 = _canonical_digest_mps(mps, mpo)
        d2 = _canonical_digest_mps(mps, mpo)
        assert d1 == d2

    def test_digest_changes_on_perturbed_mps(self):
        mpo, mps, _ = _diag_mpo_and_mps(2)
        d1 = _canonical_digest_mps(mps, mpo)
        mps2 = MPS([t.copy() for t in mps.tensors])
        mps2.tensors[0][0, 0, 0] += 1e-15
        d2 = _canonical_digest_mps(mps2, mpo)
        assert d1 != d2

    def test_digest_changes_on_different_mpo(self):
        n, d = 2, 2
        _, mps, _ = _diag_mpo_and_mps(n, d)
        mpo1 = mpo_from_matrix(np.diag([0.0, 1.0, 1.0, 2.0]), n=n, d=d)
        mpo2 = mpo_from_matrix(np.diag([0.0, 2.0, 2.0, 4.0]), n=n, d=d)
        assert _canonical_digest_mps(mps, mpo1) != _canonical_digest_mps(mps, mpo2)


# ─────────────────────── rigorous assurance ──────────────────────────────────

flint = pytest.importorskip("flint", reason="python-flint not installed")


class TestRigorousAssurance:
    """assurance='rigorous': Arb/Acb transfer-matrix contractions."""

    def test_produces_rigorous_cert(self):
        mpo, mps, _ = _diag_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo, assurance="rigorous")
        assert cert.assurance == "rigorous"
        assert "arb" in cert.backend or "acb" in cert.backend

    def test_upper_bound_holds_ground_state(self):
        mpo, mps, e0 = _diag_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo, assurance="rigorous")
        # mps_from_state introduces float64 rounding; rigorous bound is for the
        # stored MPS/MPO, which may be O(1e-15) below the exact ground state energy.
        assert cert.rayleigh_upper >= e0 - 1e-12

    def test_upper_bound_holds_uniform(self):
        mpo, mps, expected_rq = _uniform_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo, assurance="rigorous")
        assert cert.rayleigh_upper >= expected_rq - 1e-12

    def test_rigorous_upper_agrees_with_reproducible(self):
        """Rigorous and reproducible upper bounds should be within a few ULPs."""
        mpo, mps, _ = _uniform_mpo_and_mps(3)
        cert_r = rayleigh_certificate_mps(mps, mpo, assurance="reproducible")
        cert_g = rayleigh_certificate_mps(mps, mpo, assurance="rigorous")
        assert abs(cert_g.rayleigh_upper - cert_r.rayleigh_upper) < 1e-10

    def test_verify_rigorous_succeeds(self):
        mpo, mps, _ = _diag_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo, assurance="rigorous")
        verified = verify_rayleigh_certificate_mps(cert)
        assert verified.verified is True

    def test_verify_rigorous_detects_tampered_tensor(self):
        mpo, mps, _ = _diag_mpo_and_mps(2)
        cert = rayleigh_certificate_mps(mps, mpo, assurance="rigorous")
        # Real tensors are stored as nested lists; shape (1, d, χ) → [[[val, ...], ...]]
        cert._mps_tensors[0][0][0][0] += 0.1
        with pytest.raises(ValueError, match="Digest mismatch"):
            verify_rayleigh_certificate_mps(cert)

    def test_three_site_rigorous(self):
        mpo, mps, e0 = _diag_mpo_and_mps(3)
        cert = rayleigh_certificate_mps(mps, mpo, assurance="rigorous")
        assert cert.rayleigh_upper >= e0 - 1e-12
        assert cert.assurance == "rigorous"

    def test_schema_accepts_rigorous(self):
        """JSON schema must include 'rigorous' in assurance enum."""
        import json as _json
        import pathlib
        schema_path = pathlib.Path(__file__).parent.parent / "htf" / "schemas" / "rayleigh_cert_mps_v1.json"
        schema = _json.loads(schema_path.read_text())
        assert "rigorous" in schema["properties"]["assurance"]["enum"]

    def test_invalid_assurance_raises(self):
        mpo, mps, _ = _diag_mpo_and_mps(2)
        with pytest.raises(ValueError, match="assurance"):
            rayleigh_certificate_mps(mps, mpo, assurance="unknown")

