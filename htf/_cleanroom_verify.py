#!/usr/bin/env python3
"""Clean-room checks for the HTF-06 Rayleigh-certificate specification.

The primary arithmetic path converts every finite binary64 component to an
exact Fraction and evaluates the Rayleigh quotient over Q(i).  It is therefore
independent of the matrix-ball implementation described in HTF-06.  If
python-flint is installed, a second, scalar-loop ball-arithmetic path is also
run; it is optional and is not the source of rigor for the main checks.

On success this program prints exactly: ALL_CHECKS_PASSED
"""

from __future__ import annotations

import copy
import hashlib
import math
import struct
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

SCHEMA = "rayleigh-cert/v2"
THEOREM = (
    "Rayleigh-Ritz: for any non-zero |ψ⟩ and self-adjoint H, "
    "E0 ≤ Re(⟨ψ|H|ψ⟩/⟨ψ|ψ⟩)."
)
CLAIM_SUFFIX = "  [Rayleigh-Ritz upper bound on ground-state energy]"


class Rejected(ValueError):
    """Raised when a modeled certificate or input fails closed."""


@dataclass(frozen=True)
class QComplex:
    re: Fraction
    im: Fraction = Fraction(0)

    def __add__(self, other: QComplex) -> QComplex:
        return QComplex(self.re + other.re, self.im + other.im)

    def __mul__(self, other: QComplex) -> QComplex:
        return QComplex(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )

    def conjugate(self) -> QComplex:
        return QComplex(self.re, -self.im)


ZERO_QC = QComplex(Fraction(0), Fraction(0))


def _qfloat(value: float) -> Fraction:
    if not math.isfinite(value):
        raise Rejected("non-finite binary64 component")
    return Fraction.from_float(value)


def _qcomplex(value: complex | float) -> QComplex:
    z = complex(value)
    return QComplex(_qfloat(float(z.real)), _qfloat(float(z.imag)))


def _validate_inputs(H: np.ndarray, psi: np.ndarray) -> None:
    canonical = {np.dtype(np.float64), np.dtype(np.complex128)}
    if not isinstance(H, np.ndarray) or not isinstance(psi, np.ndarray):
        raise Rejected("inputs must be ndarrays")
    if H.dtype not in canonical or psi.dtype not in canonical:
        raise Rejected("inputs must use float64 or complex128")
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise Rejected("H must be square")
    if psi.ndim != 1 or psi.shape[0] != H.shape[0]:
        raise Rejected("psi has the wrong shape")
    if not np.all(np.isfinite(H)) or not np.all(np.isfinite(psi)):
        raise Rejected("inputs must be finite")
    adjoint = H.conj().T if np.iscomplexobj(H) else H.T
    if not np.array_equal(H, adjoint):
        raise Rejected("H is not exactly self-adjoint")
    if not np.any(psi != 0):
        raise Rejected("psi is exactly zero")


def _raw_exact_forms(H: np.ndarray, psi: np.ndarray) -> tuple[QComplex, Fraction]:
    """Return exact numerator and denominator without checking Hermiticity."""
    p = [_qcomplex(x) for x in psi]
    den = sum((z.re * z.re + z.im * z.im for z in p), Fraction(0))
    num = ZERO_QC
    for i in range(H.shape[0]):
        left = p[i].conjugate()
        for j in range(H.shape[1]):
            num = num + left * _qcomplex(H[i, j]) * p[j]
    return num, den


def exact_rayleigh(H: np.ndarray, psi: np.ndarray) -> Fraction:
    _validate_inputs(H, psi)
    num, den = _raw_exact_forms(H, psi)
    if den <= 0:
        raise Rejected("the exact denominator is not positive")
    if num.im != 0:
        raise Rejected("Hermitian numerator has a nonzero exact imaginary part")
    return num.re / den


def outward_binary64(value: Fraction) -> tuple[float, float]:
    """Mirror the specification's nearest conversion plus one outward step."""
    try:
        nearest = float(value)
    except OverflowError as exc:
        raise Rejected("exact quotient is outside finite binary64") from exc
    lower = math.nextafter(nearest, -math.inf)
    upper = math.nextafter(nearest, math.inf)
    if not (math.isfinite(lower) and math.isfinite(upper)):
        raise Rejected("finite outward binary64 endpoints are unavailable")
    if not (Fraction.from_float(lower) <= value <= Fraction.from_float(upper)):
        raise AssertionError("outward conversion failed")
    return lower, upper


def canonical_digest(H: np.ndarray, psi: np.ndarray) -> str:
    """Independent incremental implementation of the specified byte grammar."""
    def put_field(state: hashlib._Hash, tag: bytes, payload: bytes) -> None:
        state.update(struct.pack(">H", len(tag)))
        state.update(tag)
        state.update(struct.pack(">Q", len(payload)))
        state.update(payload)

    def shape_payload(array: np.ndarray) -> bytes:
        dims = [struct.pack(">Q", int(size)) for size in array.shape]
        return struct.pack(">I", array.ndim) + b"".join(dims)

    complex_mode = np.iscomplexobj(H) or np.iscomplexobj(psi)
    state = hashlib.sha256()
    state.update(b"rayleigh-cert-input/v2\x00")
    state.update(b"C" if complex_mode else b"R")
    for label, array in ((b"H", H), (b"psi", psi)):
        put_field(state, label + b".shape", shape_payload(array))
        if complex_mode:
            real_bytes = np.asarray(array.real, dtype=">f8").tobytes(order="C")
            imag_bytes = np.asarray(array.imag, dtype=">f8").tobytes(order="C")
            put_field(state, label + b".real", real_bytes)
            put_field(state, label + b".imag", imag_bytes)
        else:
            real_bytes = np.asarray(array, dtype=">f8").tobytes(order="C")
            put_field(state, label + b".real", real_bytes)
    return state.hexdigest()


def _backend_for(H: np.ndarray, psi: np.ndarray) -> str:
    if np.iscomplexobj(H) or np.iscomplexobj(psi):
        return "flint-acb/prec=128"
    return "flint-arb/prec=128"


def _claim(upper: float) -> str:
    return f"E0 ≤ {upper:.17g}" + CLAIM_SUFFIX


def make_modeled_certificate(H: np.ndarray, psi: np.ndarray) -> dict[str, object]:
    q = exact_rayleigh(H, psi)
    lower, upper = outward_binary64(q)
    return {
        "schema_version": SCHEMA,
        "claim": _claim(upper),
        "theorem": THEOREM,
        "input_digest": canonical_digest(H, psi),
        "lower": lower,
        "upper": upper,
        "assurance": "rigorous",
        "backend": _backend_for(H, psi),
    }


def verify_modeled_certificate(
    cert: dict[str, object], H: np.ndarray, psi: np.ndarray
) -> None:
    required = {
        "schema_version", "claim", "theorem", "input_digest", "lower",
        "upper", "assurance", "backend",
    }
    if required - cert.keys():
        raise Rejected("missing certificate field")
    if cert["schema_version"] != SCHEMA:
        raise Rejected("wrong schema")
    if cert["assurance"] != "rigorous" or cert["theorem"] != THEOREM:
        raise Rejected("wrong semantic field")
    if cert["backend"] != _backend_for(H, psi):
        raise Rejected("wrong backend field")
    if cert["input_digest"] != canonical_digest(H, psi):
        raise Rejected("input digest mismatch")

    q = exact_rayleigh(H, psi)
    lower = float(cert["lower"])
    upper = float(cert["upper"])
    if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
        raise Rejected("invalid interval endpoints")
    if cert["claim"] != _claim(upper):
        raise Rejected("claim does not match upper")
    if not (Fraction.from_float(lower) <= q <= Fraction.from_float(upper)):
        raise Rejected("stored interval does not enclose the exact quotient")


def _must_reject(action) -> None:
    try:
        action()
    except Rejected:
        return
    raise AssertionError("adversarial mutation was accepted")


def _anchor_inputs() -> list[tuple[np.ndarray, np.ndarray, Fraction]]:
    a1_H = np.diag(np.array([0.0, 1.0, 2.0], dtype=np.float64))
    a1_p = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    a2_H = a1_H.copy()
    a2_p = np.array([1.0, 1.0, 1.0], dtype=np.float64) / math.sqrt(3.0)

    a3_H = np.array([[1.0, 1.0j], [-1.0j, 1.0]], dtype=np.complex128)
    a3_p = np.array([1.0, 0.0], dtype=np.complex128)

    a4_H = np.diag(np.array([0.0, 1e-15], dtype=np.float64))
    a4_p = np.array([1.0, 1.0], dtype=np.float64) / math.sqrt(2.0)

    return [
        (a1_H, a1_p, Fraction(0)),
        (a2_H, a2_p, Fraction(1)),
        (a3_H, a3_p, Fraction(1)),
        (a4_H, a4_p, Fraction.from_float(1e-15) / 2),
    ]


def _check_anchors_and_second_path() -> None:
    anchors = _anchor_inputs()
    expected_digests = [
        "49d811a10c9827248f6bbf7219c78ccf197b0b66906a0b044216c844c93d418b",
        "a9d6b50bffbfd6ccac83b1753815e1248298cc07779eabb2305c06ed2d041149",
        "bb3d979df4b0c7e1965454affe6142522c5a6007f826320615f44f96b08c5bdc",
        "051c4237014d619346b7ddbc910e7b938d2a25e4affe269595b6a4d316dde054",
    ]
    expected_bounds = [
        (-5e-324, 5e-324),
        (math.nextafter(1.0, -math.inf), math.nextafter(1.0, math.inf)),
        (math.nextafter(1.0, -math.inf), math.nextafter(1.0, math.inf)),
    ]
    for index, (H, psi, expected_q) in enumerate(anchors):
        q = exact_rayleigh(H, psi)
        if q != expected_q:
            raise AssertionError(f"anchor {index + 1} exact quotient mismatch")
        lower, upper = outward_binary64(q)
        if index < 3 and (lower, upper) != expected_bounds[index]:
            raise AssertionError(f"anchor {index + 1} endpoint mismatch")
        if canonical_digest(H, psi) != expected_digests[index]:
            raise AssertionError(f"anchor {index + 1} digest mismatch")
        cert = make_modeled_certificate(H, psi)
        verify_modeled_certificate(cert, H, psi)

    # Independent LAPACK sanity path for the two mandatory spectral anchors.
    ev1 = np.linalg.eigvalsh(anchors[0][0])
    ev3 = np.linalg.eigvalsh(anchors[2][0])
    if not np.allclose(ev1, [0.0, 1.0, 2.0], rtol=0.0, atol=1e-15):
        raise AssertionError("anchor 1 eigenspectrum mismatch")
    if not np.allclose(ev3, [0.0, 2.0], rtol=0.0, atol=1e-15):
        raise AssertionError("anchor 3 eigenspectrum mismatch")


def _check_digest_structure() -> None:
    H_plus = np.array([[0.0]], dtype=np.float64)
    H_minus = np.array([[-0.0]], dtype=np.float64)
    p_real = np.array([1.0], dtype=np.float64)
    p_complex = np.array([1.0 + 0.0j], dtype=np.complex128)
    if canonical_digest(H_plus, p_real) == canonical_digest(H_minus, p_real):
        raise AssertionError("signed zero was not bound")
    if canonical_digest(H_plus, p_real) == canonical_digest(
        H_plus.astype(np.complex128), p_complex
    ):
        raise AssertionError("real/complex domains aliased")
    row = np.array([[1.0, 2.0]], dtype=np.float64)
    col = np.array([[1.0], [2.0]], dtype=np.float64)
    if canonical_digest(row, p_real) == canonical_digest(col, p_real):
        raise AssertionError("shape tags aliased")


def _check_adversarial_guards() -> None:
    H, psi, _ = _anchor_inputs()[2]
    cert = make_modeled_certificate(H, psi)

    changed_H = H.copy()
    changed_H[0, 0] = 2.0
    _must_reject(lambda: verify_modeled_certificate(cert, changed_H, psi))

    changed_psi = psi.copy()
    changed_psi[:] = [0.0, 1.0]
    _must_reject(lambda: verify_modeled_certificate(cert, H, changed_psi))

    for key, value in (
        ("assurance", "reproducible"),
        ("theorem", "tampered"),
        ("backend", "numpy-float"),
        ("claim", "E0 <= -1"),
        ("schema_version", "rayleigh-cert/v1"),
    ):
        bad = copy.deepcopy(cert)
        bad[key] = value
        _must_reject(lambda bad=bad: verify_modeled_certificate(bad, H, psi))

    too_low = copy.deepcopy(cert)
    too_low["upper"] = math.nextafter(1.0, -math.inf)
    too_low["claim"] = _claim(float(too_low["upper"]))
    _must_reject(lambda: verify_modeled_certificate(too_low, H, psi))

    infinite = copy.deepcopy(cert)
    infinite["upper"] = math.inf
    infinite["claim"] = _claim(math.inf)
    _must_reject(lambda: verify_modeled_certificate(infinite, H, psi))

    nonhermitian = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    real_trial = np.array([1.0, 0.0], dtype=np.float64)
    raw_num, raw_den = _raw_exact_forms(nonhermitian, real_trial)
    if raw_num.im != 0 or raw_num.re / raw_den != 1:
        raise AssertionError("non-Hermitian real-quotient witness failed")
    _must_reject(lambda: _validate_inputs(nonhermitian, real_trial))


def _check_diagnostic_float_failure() -> None:
    tiny = np.array([np.nextafter(0.0, 1.0)], dtype=np.float64)
    huge = np.array([np.finfo(np.float64).max], dtype=np.float64)
    with np.errstate(all="ignore"):
        tiny_norm = float(np.real(tiny.conj() @ tiny))
        huge_norm = float(np.real(huge.conj() @ huge))
    if tiny_norm != 0.0 or not math.isinf(huge_norm):
        raise AssertionError("expected binary64 norm underflow/overflow not observed")


def _arb_point_as_fraction(point) -> Fraction:
    mantissa, exponent = point.man_exp()
    m = int(mantissa)
    e = int(exponent)
    if e >= 0:
        return Fraction(m << e, 1)
    return Fraction(m, 1 << (-e))


def _optional_flint_crosscheck() -> None:
    try:
        from flint import acb, arb, ctx
    except ImportError:
        return

    with ctx.workprec(128):
        third = arb(1) / arb(3)
        endpoint = _arb_point_as_fraction(third.upper())
        exported = math.nextafter(float(third.upper()), math.inf)
        if Fraction.from_float(exported) < endpoint:
            raise AssertionError("Arb upper export rounded inward")

        for H, psi, exact_q in _anchor_inputs():
            if np.iscomplexobj(H) or np.iscomplexobj(psi):
                p = [acb(complex(x)) for x in psi]
                den = acb(0)
                num = acb(0)
                for z in p:
                    den += acb(complex(z).conjugate()) * z
                for i in range(H.shape[0]):
                    for j in range(H.shape[1]):
                        num += (
                            acb(complex(p[i]).conjugate())
                            * acb(complex(H[i, j]))
                            * p[j]
                        )
                ball = num / den
                if not ball.imag.contains(0):
                    raise AssertionError("Acb Hermitian result excludes zero imaginary part")
                real_ball = ball.real
            else:
                p = [arb(float(x)) for x in psi]
                den = arb(0)
                num = arb(0)
                for z in p:
                    den += z * z
                for i in range(H.shape[0]):
                    for j in range(H.shape[1]):
                        num += p[i] * arb(float(H[i, j])) * p[j]
                real_ball = num / den

            low = _arb_point_as_fraction(real_ball.lower())
            high = _arb_point_as_fraction(real_ball.upper())
            if not (low <= exact_q <= high):
                raise AssertionError("scalar Flint path missed exact quotient")


def main() -> None:
    _check_anchors_and_second_path()
    _check_digest_structure()
    _check_adversarial_guards()
    _check_diagnostic_float_failure()
    _optional_flint_crosscheck()
    print("ALL_CHECKS_PASSED")


if __name__ == "__main__":
    main()
