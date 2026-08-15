#!/usr/bin/env python3
"""Clean-room audit of the mathematical core described in HTF-06.

This program deliberately does not import HTF or python-flint.  Every binary64
or complex128 component is converted to its exact rational value and the
Rayleigh quotient is evaluated with ``fractions.Fraction``.  This is an
arithmetic path independent of the producer's Arb/Acb implementation.

HTF-06 does not specify ``_encode_canonical``/``_decode_canonical`` or the full
RayleighCertificate schema.  Consequently this program verifies an in-memory
model of the fields that HTF-06 does specify; it does not claim wire-format
compatibility with rayleigh-cert/v2.  That omission is a Gate-A finding, not a
format guessed here.

Run without arguments.  Success prints exactly ``ALL_CHECKS_PASSED``.
"""

from __future__ import annotations

import hashlib
import math
import struct
from copy import deepcopy
from fractions import Fraction
from typing import Any

import numpy as np

SCHEMA_VERSION = "rayleigh-cert/v2"
EXPECTED_THEOREM = "Rayleigh-Ritz: for any non-zero |ψ⟩ and self-adjoint H, E0 ≤ Re(⟨ψ|H|ψ⟩/⟨ψ|ψ⟩)."
CLAIM_SUFFIX = "  [Rayleigh-Ritz upper bound on ground-state energy]"


class AuditFailure(ValueError):
    """Raised when a certificate or test input fails the clean-room audit."""


RatComplex = tuple[Fraction, Fraction]


def _rat(value: float | np.floating[Any]) -> Fraction:
    value_f = float(value)
    if not math.isfinite(value_f):
        raise AuditFailure("non-finite scalar")
    return Fraction.from_float(value_f)


def _z(value: complex | np.complexfloating[Any, Any] | float) -> RatComplex:
    value_c = complex(value)
    return _rat(value_c.real), _rat(value_c.imag)


def _z_add(a: RatComplex, b: RatComplex) -> RatComplex:
    return a[0] + b[0], a[1] + b[1]


def _z_mul(a: RatComplex, b: RatComplex) -> RatComplex:
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def _z_conj(a: RatComplex) -> RatComplex:
    return a[0], -a[1]


def check_preconditions(H: Any, psi: Any) -> tuple[np.ndarray, np.ndarray]:
    """Check the stated mathematical domain without silently flattening psi."""
    H_arr = np.asarray(H)
    psi_arr = np.asarray(psi)
    allowed = {np.dtype(np.float64), np.dtype(np.complex128)}
    if H_arr.dtype not in allowed or psi_arr.dtype not in allowed:
        raise AuditFailure("inputs must have dtype float64 or complex128")
    if H_arr.ndim != 2 or H_arr.shape[0] != H_arr.shape[1]:
        raise AuditFailure("H must be a square 2-D array")
    if psi_arr.ndim != 1 or psi_arr.shape[0] != H_arr.shape[0]:
        raise AuditFailure("psi must be a matching 1-D vector")
    if not np.all(np.isfinite(H_arr)) or not np.all(np.isfinite(psi_arr)):
        raise AuditFailure("all entries must be finite")
    if not np.array_equal(H_arr, H_arr.conj().T):
        raise AuditFailure("H must be exactly self-adjoint")
    if not np.any(psi_arr != 0):
        raise AuditFailure("psi must be non-zero")
    return H_arr, psi_arr


def exact_rayleigh(H: Any, psi: Any) -> Fraction:
    """Return the exact Rayleigh quotient of the represented binary floats."""
    H_arr, psi_arr = check_preconditions(H, psi)
    n = psi_arr.shape[0]
    p = [_z(psi_arr[i]) for i in range(n)]

    denominator = Fraction(0)
    for value in p:
        denominator += value[0] * value[0] + value[1] * value[1]
    if denominator <= 0:
        raise AuditFailure("exact denominator is not positive")

    numerator: RatComplex = (Fraction(0), Fraction(0))
    for i in range(n):
        for j in range(n):
            term = _z_mul(_z_mul(_z_conj(p[i]), _z(H_arr[i, j])), p[j])
            numerator = _z_add(numerator, term)
    if numerator[1] != 0:
        raise AuditFailure("exact self-adjoint quadratic form is not real")
    return numerator[0] / denominator


def canonical_digest(H: Any, psi: Any) -> str:
    """Independent implementation of the digest preimage specified in HTF-06."""
    H_arr, psi_arr = check_preconditions(H, psi)

    def field(tag: bytes, payload: bytes) -> bytes:
        return struct.pack(">H", len(tag)) + tag + struct.pack(">Q", len(payload)) + payload

    def shape_bytes(array: np.ndarray) -> bytes:
        return struct.pack(">I", array.ndim) + b"".join(struct.pack(">Q", int(size)) for size in array.shape)

    complex_path = np.iscomplexobj(H_arr) or np.iscomplexobj(psi_arr)
    raw = b"rayleigh-cert-input/v2\x00" + (b"C" if complex_path else b"R")
    for name, array in ((b"H", H_arr), (b"psi", psi_arr)):
        raw += field(name + b".shape", shape_bytes(array))
        if complex_path:
            raw += field(name + b".real", np.asarray(array.real, dtype=">f8").tobytes(order="C"))
            raw += field(name + b".imag", np.asarray(array.imag, dtype=">f8").tobytes(order="C"))
        else:
            raw += field(name + b".real", np.asarray(array, dtype=">f8").tobytes(order="C"))
    return hashlib.sha256(raw).hexdigest()


def _backend(H: np.ndarray, psi: np.ndarray) -> str:
    if np.iscomplexobj(H) or np.iscomplexobj(psi):
        return "flint-acb/prec=128"
    return "flint-arb/prec=128"


def _claim(upper: float) -> str:
    return f"E0 ≤ {upper:.17g}{CLAIM_SUFFIX}"


def verify_in_memory(cert: dict[str, Any]) -> None:
    """Verify all reviewable semantic fields and the exact upper-bound claim."""
    required = {
        "schema_version",
        "claim",
        "theorem",
        "input_digest",
        "interval",
        "canonical",
        "assurance",
        "backend",
    }
    if not required <= set(cert):
        raise AuditFailure("missing required certificate field")
    if cert["schema_version"] != SCHEMA_VERSION:
        raise AuditFailure("wrong schema version")
    if cert["assurance"] != "rigorous":
        raise AuditFailure("wrong assurance")
    if cert["theorem"] != EXPECTED_THEOREM:
        raise AuditFailure("wrong theorem")

    canonical = cert["canonical"]
    if not isinstance(canonical, dict) or set(canonical) < {"H", "psi"}:
        raise AuditFailure("missing canonical input")
    H_arr, psi_arr = check_preconditions(canonical["H"], canonical["psi"])
    if cert["input_digest"] != canonical_digest(H_arr, psi_arr):
        raise AuditFailure("digest mismatch")
    if cert["backend"] != _backend(H_arr, psi_arr):
        raise AuditFailure("backend mismatch")

    interval = cert["interval"]
    if not isinstance(interval, dict) or "upper" not in interval:
        raise AuditFailure("missing interval.upper")
    upper_raw = interval["upper"]
    if isinstance(upper_raw, (bool, np.bool_)) or not isinstance(upper_raw, (float, np.floating)):
        raise AuditFailure("upper must be a binary64 floating-point value")
    if isinstance(upper_raw, np.floating) and np.asarray(upper_raw).dtype != np.dtype(np.float64):
        raise AuditFailure("upper must be binary64, not another NumPy float format")
    upper = float(upper_raw)
    if not math.isfinite(upper):
        raise AuditFailure("upper must be finite")
    if cert["claim"] != _claim(upper):
        raise AuditFailure("claim text mismatch")

    q_exact = exact_rayleigh(H_arr, psi_arr)
    if Fraction.from_float(upper) < q_exact:
        raise AuditFailure("stored upper is below the exact Rayleigh quotient")


def _producer_style_upper(exact_value: Fraction) -> float:
    rounded = float(exact_value)
    if not math.isfinite(rounded):
        raise AuditFailure("anchor does not have a finite binary64 export")
    return math.nextafter(rounded, math.inf)


def _make_cert(H: np.ndarray, psi: np.ndarray) -> dict[str, Any]:
    upper = _producer_style_upper(exact_rayleigh(H, psi))
    return {
        "schema_version": SCHEMA_VERSION,
        "claim": _claim(upper),
        "theorem": EXPECTED_THEOREM,
        "input_digest": canonical_digest(H, psi),
        "interval": {"upper": upper},
        "canonical": {"H": H.copy(), "psi": psi.copy()},
        "assurance": "rigorous",
        "backend": _backend(H, psi),
    }


def _must_fail(cert: dict[str, Any]) -> None:
    try:
        verify_in_memory(cert)
    except AuditFailure:
        return
    raise AssertionError("adversarial mutation unexpectedly passed")


def run_checks() -> None:
    real_H = np.diag(np.array([0.0, 1.0, 2.0], dtype=np.float64))
    real_psi = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    complex_H = np.array([[1.0, 1.0j], [-1.0j, 1.0]], dtype=np.complex128)
    complex_psi = np.array([1.0, 0.0], dtype=np.complex128)

    # Independent exact anchor calculations.
    assert exact_rayleigh(real_H, real_psi) == 0
    assert np.linalg.eigvalsh(real_H)[0] == 0.0
    assert exact_rayleigh(complex_H, complex_psi) == 1
    assert np.linalg.eigvalsh(complex_H)[0] == 0.0

    cert_real = _make_cert(real_H, real_psi)
    cert_complex = _make_cert(complex_H, complex_psi)
    assert cert_real["interval"]["upper"].hex() == "0x0.0000000000001p-1022"
    assert cert_complex["interval"]["upper"].hex() == "0x1.0000000000001p+0"
    verify_in_memory(cert_real)
    verify_in_memory(cert_complex)

    missing = deepcopy(cert_real)
    del missing["theorem"]
    _must_fail(missing)

    # One-field semantic mutations.
    for key, bad_value in (
        ("schema_version", "rayleigh-cert/v1"),
        ("assurance", "reproducible"),
        ("theorem", EXPECTED_THEOREM + " altered"),
        ("backend", "numpy/float64"),
        ("claim", "E0 ≤ -1"),
        ("input_digest", "00" * 32),
    ):
        bad = deepcopy(cert_real)
        bad[key] = bad_value
        _must_fail(bad)

    # A lowered upper fails even if the attacker also rewrites the claim.
    bad_upper = deepcopy(cert_complex)
    bad_upper["interval"]["upper"] = math.nextafter(1.0, -math.inf)
    bad_upper["claim"] = _claim(bad_upper["interval"]["upper"])
    _must_fail(bad_upper)

    # Mutating an input without its digest fails.
    bad_input = deepcopy(cert_real)
    bad_input["canonical"]["H"][0, 0] = 2.0
    _must_fail(bad_input)

    # Mutating the input and digest still fails if the old upper no longer covers q.
    forged_input = deepcopy(cert_real)
    forged_input["canonical"]["H"][0, 0] = 2.0
    forged_input["input_digest"] = canonical_digest(forged_input["canonical"]["H"], forged_input["canonical"]["psi"])
    _must_fail(forged_input)

    # Preconditions and structural digest distinctions.
    nonhermitian = complex_H.copy()
    nonhermitian[1, 0] = 1.0j
    try:
        exact_rayleigh(nonhermitian, complex_psi)
    except AuditFailure:
        pass
    else:
        raise AssertionError("non-Hermitian matrix passed")
    assert canonical_digest(real_H, real_psi) != canonical_digest(
        real_H.astype(np.complex128), real_psi.astype(np.complex128)
    )


if __name__ == "__main__":
    run_checks()
    print("ALL_CHECKS_PASSED")
