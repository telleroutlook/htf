#!/usr/bin/env python3
"""Independent verifier and adversarial audit for rayleigh-cert/v1.

Run with no arguments (or ``--self-test``) to reproduce the audit findings.
Run with a certificate JSON path to verify the included finite binary64 input
and the claimed upper bound using exact rational arithmetic.  The verifier does
not trust the certificate's ``verified`` flag.

Dependencies: Python standard library, NumPy, and (for the self-test only)
python-flint.  A successful run prints exactly ``ALL_CHECKS_PASSED``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import struct
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np


class VerificationError(ValueError):
    """A certificate or an audit invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _as_finite_binary64(value: Any, label: str) -> float:
    _require(not isinstance(value, bool), f"{label} must be numeric, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise VerificationError(f"{label} is not binary64-convertible") from exc
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def decode_preserving_components(data: Any) -> np.ndarray:
    """Decode the quoted replay format without complex signed-zero arithmetic."""
    if isinstance(data, dict):
        _require(set(data) == {"real", "imag"}, "complex payload keys are invalid")
        real = np.asarray(data["real"], dtype=np.float64)
        imag = np.asarray(data["imag"], dtype=np.float64)
        _require(real.shape == imag.shape, "real/imag shapes differ")
        result = np.empty(real.shape, dtype=np.complex128)
        result.real = real
        result.imag = imag
        return result
    return np.asarray(data, dtype=np.float64)


def encode_components(array: np.ndarray) -> Any:
    array = np.asarray(array)
    if np.iscomplexobj(array):
        return {"real": array.real.tolist(), "imag": array.imag.tolist()}
    return array.tolist()


def legacy_digest_bytes(H: np.ndarray, psi: np.ndarray) -> bytes:
    """Reproduce exactly the native-endian concatenation quoted for v1."""
    H = np.asarray(H)
    psi = np.asarray(psi)
    if np.iscomplexobj(H) or np.iscomplexobj(psi):
        return (
            H.real.astype(np.float64).tobytes()
            + H.imag.astype(np.float64).tobytes()
            + psi.real.astype(np.float64).tobytes()
            + psi.imag.astype(np.float64).tobytes()
        )
    return H.astype(np.float64).tobytes() + psi.astype(np.float64).tobytes()


def legacy_digest(H: np.ndarray, psi: np.ndarray) -> str:
    return hashlib.sha256(legacy_digest_bytes(H, psi)).hexdigest()


def _field(tag: bytes, payload: bytes) -> bytes:
    return struct.pack(">H", len(tag)) + tag + struct.pack(">Q", len(payload)) + payload


def _shape_bytes(array: np.ndarray) -> bytes:
    return struct.pack(">I", array.ndim) + b"".join(
        struct.pack(">Q", int(size)) for size in array.shape
    )


def tagged_digest(H: np.ndarray, psi: np.ndarray) -> str:
    """A domain-separated, shape-bound, fixed-endian replacement digest."""
    H = np.asarray(H)
    psi = np.asarray(psi)
    complex_path = np.iscomplexobj(H) or np.iscomplexobj(psi)
    raw = b"rayleigh-cert-input/v2\x00" + (b"C" if complex_path else b"R")
    for name, array in ((b"H", H), (b"psi", psi)):
        raw += _field(name + b".shape", _shape_bytes(array))
        if complex_path:
            raw += _field(name + b".real", np.asarray(array.real, dtype=">f8").tobytes(order="C"))
            raw += _field(name + b".imag", np.asarray(array.imag, dtype=">f8").tobytes(order="C"))
        else:
            raw += _field(name + b".real", np.asarray(array, dtype=">f8").tobytes(order="C"))
    return hashlib.sha256(raw).hexdigest()


def _complex_fraction(value: complex | float | np.generic) -> tuple[Fraction, Fraction]:
    z = complex(value)
    return Fraction.from_float(z.real), Fraction.from_float(z.imag)


def _cadd(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
) -> tuple[Fraction, Fraction]:
    return left[0] + right[0], left[1] + right[1]


def _cmul(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
) -> tuple[Fraction, Fraction]:
    a, b = left
    c, d = right
    return a * c - b * d, a * d + b * c


def _cconj(value: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
    return value[0], -value[1]


def exact_rayleigh(H: np.ndarray, psi: np.ndarray) -> Fraction:
    """Compute the Rayleigh quotient of binary64 data with exact Fractions."""
    H = np.asarray(H)
    psi = np.asarray(psi)
    _require(H.ndim == 2 and H.shape[0] == H.shape[1], "H must be square")
    n = H.shape[0]
    _require(psi.ndim == 1 and psi.shape[0] == n, "psi length mismatch")
    _require(np.issubdtype(H.dtype, np.number), "H dtype is not numeric")
    _require(np.issubdtype(psi.dtype, np.number), "psi dtype is not numeric")
    _require(bool(np.isfinite(H).all()), "H contains NaN or Infinity")
    _require(bool(np.isfinite(psi).all()), "psi contains NaN or Infinity")
    _require(bool(np.any(psi != 0)), "psi is zero")
    _require(bool(np.array_equal(H, H.conj().T)), "H is not exactly Hermitian")

    p = [_complex_fraction(value) for value in psi]
    matrix = [[_complex_fraction(H[i, j]) for j in range(n)] for i in range(n)]
    numerator = (Fraction(0), Fraction(0))
    for i in range(n):
        for j in range(n):
            numerator = _cadd(
                numerator,
                _cmul(_cmul(_cconj(p[i]), matrix[i][j]), p[j]),
            )
    denominator = sum((a * a + b * b for a, b in p), Fraction(0))
    _require(denominator > 0, "exact denominator is not positive")
    _require(numerator[1] == 0, "Hermitian numerator is not exactly real")
    return numerator[0] / denominator


def _claim_for(upper: float) -> str:
    return (
        f"E0 ≤ {upper:.17g}  "
        "[Rayleigh-Ritz upper bound on ground-state energy]"
    )


def verify_certificate(
    cert: dict[str, Any],
    *,
    expected_legacy_digest: str | None = None,
    expected_tagged_digest: str | None = None,
) -> None:
    """Verify v1 replay data and its bound without any floating tolerance."""
    required = {
        "schema_version",
        "claim",
        "theorem",
        "assumptions",
        "input_digest",
        "interval",
        "backend",
        "htf_version",
        "verified",
        "notes",
        "_H_canonical",
        "_psi_canonical",
    }
    _require(required <= set(cert), "certificate is missing required replay fields")
    _require(cert["schema_version"] == "rayleigh-cert/v1", "wrong schema version")
    _require(isinstance(cert["theorem"], str) and cert["theorem"], "theorem is empty")
    _require(
        isinstance(cert["assumptions"], list)
        and all(isinstance(item, str) for item in cert["assumptions"]),
        "assumptions must be a string list",
    )
    _require(isinstance(cert["backend"], str), "backend must be a string")
    _require(isinstance(cert["htf_version"], str), "htf_version must be a string")
    _require(isinstance(cert["verified"], bool), "verified must be bool")
    _require(isinstance(cert["notes"], str), "notes must be a string")

    H = decode_preserving_components(cert["_H_canonical"])
    psi = decode_preserving_components(cert["_psi_canonical"])
    quotient = exact_rayleigh(H, psi)

    digest = legacy_digest(H, psi)
    _require(cert["input_digest"] == digest, "legacy input digest mismatch")
    if expected_legacy_digest is not None:
        _require(digest == expected_legacy_digest, "unexpected legacy input digest")
    if expected_tagged_digest is not None:
        _require(tagged_digest(H, psi) == expected_tagged_digest, "unexpected tagged digest")

    interval = cert["interval"]
    _require(isinstance(interval, dict), "interval must be an object")
    _require(
        {"lower", "upper", "midpoint", "radius"} <= set(interval),
        "interval fields are incomplete",
    )
    lower = _as_finite_binary64(interval["lower"], "interval.lower")
    upper = _as_finite_binary64(interval["upper"], "interval.upper")
    midpoint = _as_finite_binary64(interval["midpoint"], "interval.midpoint")
    radius = _as_finite_binary64(interval["radius"], "interval.radius")
    lower_q = Fraction.from_float(lower)
    upper_q = Fraction.from_float(upper)
    midpoint_q = Fraction.from_float(midpoint)
    radius_q = Fraction.from_float(radius)
    _require(lower_q <= midpoint_q <= upper_q, "interval ordering is invalid")
    _require(radius_q >= 0, "interval radius is negative")
    _require(
        radius_q >= max(midpoint_q - lower_q, upper_q - midpoint_q),
        "radius does not cover both stored endpoints",
    )
    _require(quotient <= upper_q, "stored upper is below the exact Rayleigh quotient")
    _require(cert["claim"] == _claim_for(upper), "claim text does not match interval.upper")


def _legacy_accepts_preconditions(H: np.ndarray, psi: np.ndarray) -> bool:
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        return False
    if psi.ndim != 1 or len(psi) != H.shape[0]:
        return False
    if np.iscomplexobj(H):
        if float(np.abs(H - H.conj().T).max()) > 1e-10:
            return False
    elif float(np.abs(H - H.T).max()) > 1e-10:
        return False
    norm_sq = float(np.real(psi.conj() @ psi))
    return not (norm_sq < 1e-30)


def _legacy_decode(data: Any) -> np.ndarray:
    if isinstance(data, dict):
        return np.array(data["real"]) + 1j * np.array(data["imag"])
    return np.array(data, dtype=float)


def _make_certificate(H: np.ndarray, psi: np.ndarray, upper: float) -> dict[str, Any]:
    return {
        "schema_version": "rayleigh-cert/v1",
        "claim": _claim_for(upper),
        "theorem": "Rayleigh-Ritz for nonzero psi and self-adjoint H",
        "assumptions": ["finite binary64", "H exactly Hermitian", "psi nonzero"],
        "input_digest": legacy_digest(H, psi),
        "interval": {
            "lower": upper,
            "upper": upper,
            "midpoint": upper,
            "radius": 0.0,
        },
        "backend": "flint-arb",
        "htf_version": "independent-audit",
        "verified": False,
        "notes": "Exact-rational independent replay.",
        "_H_canonical": encode_components(H),
        "_psi_canonical": encode_components(psi),
    }


def run_self_tests() -> None:
    old_err = np.seterr(all="ignore")
    try:
        # Independent exact theorem path, cross-checked against NumPy.
        H = np.diag([0.0, 1.0])
        psi = np.array([1.0, 0.0])
        _require(exact_rayleigh(H, psi) == 0, "anchor quotient is not zero")
        _require(float(np.vdot(psi, H @ psi) / np.vdot(psi, psi)) == 0.0, "NumPy anchor disagrees")

        H_complex = np.array([[2.0, 1.0j], [-1.0j, 3.0]], dtype=np.complex128)
        psi_complex = np.array([1.0 + 2.0j, -0.5 + 0.25j])
        q_exact = exact_rayleigh(H_complex, psi_complex)
        q_numpy = float(np.real(np.vdot(psi_complex, H_complex @ psi_complex) / np.vdot(psi_complex, psi_complex)))
        _require(math.isclose(float(q_exact), q_numpy, rel_tol=0.0, abs_tol=2e-15), "complex paths disagree")

        # Link 1 counterexample: tolerance admits a non-self-adjoint matrix.
        eps = np.float64(5e-11)
        H_bad = np.array([[0.0, eps], [0.0, 0.0]])
        psi_bad = np.array([1.0, -1.0])
        _require(_legacy_accepts_preconditions(H_bad, psi_bad), "near-symmetry counterexample was rejected")
        rq_bad = float(np.real(np.vdot(psi_bad, H_bad @ psi_bad) / np.vdot(psi_bad, psi_bad)))
        _require(rq_bad == -2.5e-11 and not (0.0 <= rq_bad), "near-symmetry counterexample failed")

        # NaN comparisons in both the precondition and verifier fail open.
        _require(_legacy_accepts_preconditions(np.array([[np.nan]]), np.array([1.0])), "NaN H did not pass as expected")
        _require(_legacy_accepts_preconditions(np.array([[0.0]]), np.array([np.nan])), "NaN psi did not pass as expected")
        _require(not (float("nan") > 0.0 + 1e-15), "NaN verifier comparison did not fail open")

        # Numerical anchor: quoted bytes are big-endian; code bytes are native-endian.
        H_anchor = np.diag([0.0, 1.0])
        psi_anchor = np.array([1.0, 0.0])
        native_hash = legacy_digest(H_anchor, psi_anchor)
        big_raw = H_anchor.astype(">f8").tobytes() + psi_anchor.astype(">f8").tobytes()
        big_hash = hashlib.sha256(big_raw).hexdigest()
        expected_native_anchor = {
            "little": "6f574f263c46e7cad7afd874638ab085257979c17e71e703bd2c5560010b52b8",
            "big": "037b991da8e0441b30d0128476abafc45310ce3da3957b93fae02d5891c26bc1",
        }[sys.byteorder]
        _require(native_hash == expected_native_anchor, "native anchor changed")
        _require(big_hash == "037b991da8e0441b30d0128476abafc45310ce3da3957b93fae02d5891c26bc1", "big-endian anchor changed")
        if sys.byteorder == "little":
            _require(native_hash != big_hash, "anchor endianness defect disappeared")

        # Link 4 structural collision: distinct valid real/complex inputs, same raw bytes.
        H_real = np.diag([1.0, 0.0, 1.0])
        psi_real = np.array([0.0, 1.0, 0.0])
        H_cplx = np.diag([1.0 + 0.0j, 0.0 + 0.0j])
        psi_cplx = np.array([1.0 + 1.0j, 0.0 + 0.0j])
        _require(legacy_digest_bytes(H_real, psi_real) == legacy_digest_bytes(H_cplx, psi_cplx), "structural collision bytes differ")
        expected_structural_digest = {
            "little": "5d49ac866e9b48df5b3e9ccdd996bea7b8dac77dd84ef55f66dafebb8b4efabb",
            "big": "a52bb5d3c16e98152e75775a7a15405d5f89243679b7e35f5fcb5bb06b31a877",
        }[sys.byteorder]
        _require(legacy_digest(H_real, psi_real) == expected_structural_digest, "structural collision digest changed")
        _require(exact_rayleigh(H_real, psi_real) == 0, "real collision quotient changed")
        _require(exact_rayleigh(H_cplx, psi_cplx) == 1, "complex collision quotient changed")
        _require(tagged_digest(H_real, psi_real) != tagged_digest(H_cplx, psi_cplx), "tagged digest failed to separate inputs")

        # Link 5: arithmetic reconstruction loses complex signed-zero bits.
        H_zero = np.zeros((2, 2), dtype=np.complex128)
        psi_zero = np.zeros(2, dtype=np.complex128)
        psi_zero.real[0] = 1.0
        psi_zero.imag[0] = -0.0
        encoded = encode_components(psi_zero)
        legacy_roundtrip = _legacy_decode(encoded)
        fixed_roundtrip = decode_preserving_components(encoded)
        _require(psi_zero.tobytes() != legacy_roundtrip.tobytes(), "signed-zero defect was not reproduced")
        _require(psi_zero.tobytes() == fixed_roundtrip.tobytes(), "component-preserving decode failed")
        expected_signed_zero_source = {
            "little": "520ed84f78c15b465e2894663d4123653c8dc5dd07b51607c111005b87091bab",
            "big": "51f72960c1bac3dbd5306e1284a4efd2576aaff91d3f1a78907bff029090298a",
        }[sys.byteorder]
        expected_signed_zero_replay = {
            "little": "0a61b55b94db9397645a0b77a30c7756279850b2d256284f0ffec6d4ae5c3f77",
            "big": "113f99c58debfffa4c6298024aaea8d20076feabf58290a206123c4f01b25c2c",
        }[sys.byteorder]
        _require(legacy_digest(H_zero, psi_zero) == expected_signed_zero_source, "signed-zero source digest changed")
        _require(legacy_digest(H_zero, legacy_roundtrip) == expected_signed_zero_replay, "signed-zero replay digest changed")

        # Link 6: the quoted tolerance accepts a strictly false upper bound.
        stored = 1.0 - 5e-15
        tolerance = max(abs(stored) * 1e-14, 1e-15)
        _require(not (1.0 > stored + tolerance), "legacy tolerance attack was rejected")
        _require(stored < 1.0, "mutated upper is not strictly smaller")

        sound_cert = _make_certificate(np.array([[1.0]]), np.array([1.0]), 1.0)
        verify_certificate(sound_cert)
        weak_cert = copy.deepcopy(sound_cert)
        weak_cert["interval"] = {
            "lower": stored,
            "upper": stored,
            "midpoint": stored,
            "radius": 0.0,
        }
        weak_cert["claim"] = _claim_for(stored)
        try:
            verify_certificate(weak_cert)
        except VerificationError:
            pass
        else:
            raise VerificationError("exact verifier accepted the weakened upper")

        digest_mutation = copy.deepcopy(sound_cert)
        digest_mutation["_H_canonical"] = [[2.0]]
        try:
            verify_certificate(digest_mutation)
        except VerificationError:
            pass
        else:
            raise VerificationError("digest mutation guard failed")

        claim_mutation = copy.deepcopy(sound_cert)
        claim_mutation["claim"] = "E0 ≤ -1"
        try:
            verify_certificate(claim_mutation)
        except VerificationError:
            pass
        else:
            raise VerificationError("claim mutation guard failed")

        # Links 2/3: reproduce the actual python-flint export defect at 64 bits.
        try:
            from flint import arb, arb_mat, ctx, fmpq
        except ImportError as exc:
            raise VerificationError("python-flint is required for --self-test") from exc
        old_prec = ctx.prec
        try:
            ctx.prec = 64
            H_flint = np.diag([1.0, 0.0, 0.0])
            psi_flint = np.ones(3)
            row = arb_mat([[arb(float(value)) for value in psi_flint]])
            col = arb_mat([[arb(float(value))] for value in psi_flint])
            matrix = arb_mat(
                [[arb(float(H_flint[i, j])) for j in range(3)] for i in range(3)]
            )
            numerator = (row * (matrix * col))[0, 0]
            denominator = (row * col)[0, 0]
            _require(not denominator.contains(0), "test denominator contains zero")
            quotient = numerator / denominator
            _require(quotient.contains(fmpq(1, 3)), "Arb ball lost the exact quotient")
            legacy_upper = float(quotient.mid()) + float(quotient.rad())
            _require(
                Fraction.from_float(legacy_upper) < Fraction(1, 3),
                "mid+rad export did not underbound",
            )
            candidate = float(quotient.upper())
            outward_upper = math.nextafter(candidate, math.inf)
            upper_man, upper_exp = quotient.upper().man_exp()
            exact_ball_upper = Fraction(int(upper_man)) * (Fraction(2) ** int(upper_exp))
            _require(Fraction.from_float(outward_upper) >= exact_ball_upper, "outward export failed")
        finally:
            ctx.prec = old_prec
    finally:
        np.seterr(**old_err)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--expected-input-digest")
    parser.add_argument("--expected-tagged-digest")
    args = parser.parse_args()

    try:
        if args.self_test or args.certificate is None:
            run_self_tests()
        else:
            with args.certificate.open("r", encoding="utf-8") as handle:
                cert = json.load(handle)
            _require(isinstance(cert, dict), "certificate root must be an object")
            verify_certificate(
                cert,
                expected_legacy_digest=args.expected_input_digest,
                expected_tagged_digest=args.expected_tagged_digest,
            )
    except (OSError, json.JSONDecodeError, VerificationError) as exc:
        print(f"CHECK_FAILED: {exc}", file=sys.stderr)
        return 1

    print("ALL_CHECKS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
