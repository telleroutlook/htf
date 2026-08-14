"""HTF Rayleigh primitive operations — pure arithmetic, no cert/schema dependencies.

This module contains ONLY:
  - SCHEMA_VERSION constant
  - Input validation (_check_preconditions)
  - Canonical encoding/decoding (_encode_canonical, _decode_canonical)
  - Canonical digest (_canonical_digest)
  - Interval arithmetic backends (_arb_rayleigh, _acb_rayleigh)

It has no imports from any other HTF module, so both ``rayleigh_cert`` (the
producer) and ``verify`` (the independent checker) can import from here
without either depending on the other.  This is the architectural separation
that makes the verifier non-circular: the certificate generator and the
independent verifier share only these primitive building blocks, not each
other's certificate-management logic.

Note on true independence
-------------------------
Sharing the same implementation of ``_arb_rayleigh`` means the verifier and
producer run identical arithmetic code (F-4).  Full independence would require
the verifier to use a separate arithmetic implementation (e.g. a different
library or a hand-audited clean-room copy).  That is a P2 deliverable.  This
module is a necessary first step: it removes the circular dependency
``verify → rayleigh_cert → verify`` and isolates the arithmetic so it can
later be audited or replaced independently.
"""
from __future__ import annotations

import hashlib
import math
import struct

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Schema version constant (single authoritative definition)
# ──────────────────────────────────────────────────────────────────────────────

SCHEMA_VERSION = "rayleigh-cert/v2"


# ──────────────────────────────────────────────────────────────────────────────
# Precondition check
# ──────────────────────────────────────────────────────────────────────────────

def _check_preconditions(H: np.ndarray, psi: np.ndarray) -> list[str]:
    """Verify Rayleigh-Ritz preconditions; return list of passed checks.

    Accepts real symmetric H or complex Hermitian H.
    Uses **exact** checks throughout: finite, exact symmetry/Hermiticity,
    exact non-zero.  Raises ``ValueError`` on any violation.
    """
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be a square 2-D array; got shape {H.shape}")
    n = H.shape[0]
    if psi.ndim != 1 or len(psi) != n:
        raise ValueError(
            f"|ψ⟩ must be a 1-D vector of length {n}; got shape {psi.shape}"
        )

    # Finite check (must precede all comparisons; NaN comparisons are False,
    # causing fail-open rather than fail-closed if this check is skipped).
    if not np.all(np.isfinite(H)):
        raise ValueError("H contains non-finite values (NaN or Infinity)")
    if not np.all(np.isfinite(psi)):
        raise ValueError("|ψ⟩ contains non-finite values (NaN or Infinity)")

    # Exact symmetry / Hermiticity (not approximate).
    if np.iscomplexobj(H):
        if not np.array_equal(H, H.conj().T):
            raise ValueError(
                "H is not exactly Hermitian (H ≠ H†). "
                "If H is only approximately Hermitian, symmetrize first: "
                "Hsym = (H + H.conj().T) / 2"
            )
        h_check = f"H is complex Hermitian ({n}×{n}): exactly H == H† verified"
    else:
        if not np.array_equal(H, H.T):
            raise ValueError(
                "H is not exactly symmetric (H ≠ Hᵀ). "
                "If H is only approximately symmetric, symmetrize first: "
                "Hsym = (H + H.T) / 2"
            )
        h_check = f"H is real and square ({n}×{n}): exactly symmetric (H == Hᵀ)"

    if not np.any(psi != 0):
        raise ValueError("|ψ⟩ has zero norm")

    norm_sq = float(np.real(psi.conj() @ psi))
    psi_dtype = "complex" if np.iscomplexobj(psi) else "real"
    return [
        "H and |ψ⟩ are finite (no NaN or Infinity)",
        h_check,
        f"|ψ⟩ is a {psi_dtype} vector of length {n} with exact non-zero check",
        f"⟨ψ|ψ⟩ = {norm_sq:.6g} > 0",
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Canonical encoding / decoding / digest
# ──────────────────────────────────────────────────────────────────────────────

def _encode_canonical(arr: np.ndarray) -> object:
    """JSON-serialisable representation of a numpy array (real or complex)."""
    if np.iscomplexobj(arr):
        return {"real": arr.real.tolist(), "imag": arr.imag.tolist()}
    return arr.tolist()


def _decode_canonical(data: object) -> np.ndarray:
    """Reconstruct a numpy array from ``_encode_canonical`` output.

    Uses component-wise assignment for complex arrays to preserve ``-0.0``
    bit patterns.  Arithmetic reconstruction (``real + 1j * imag``) can flip
    signed-zero bits, causing a digest mismatch for untampered certificates.
    """
    if isinstance(data, dict):
        real = np.asarray(data["real"], dtype=np.float64)
        imag = np.asarray(data["imag"], dtype=np.float64)
        result = np.empty(real.shape, dtype=np.complex128)
        result.real = real
        result.imag = imag
        return result
    return np.asarray(data, dtype=np.float64)


def _canonical_digest(H: np.ndarray, psi: np.ndarray) -> str:
    """Domain-separated, shape-tagged, fixed big-endian SHA-256 digest (v2).

    Format::

        "rayleigh-cert-input/v2\\x00" + type_flag (C|R)
        For each array (H, psi):
            field(name.shape, big-endian ndim + each dimension as uint64)
            field(name.real,  C-order big-endian float64 bytes)
            [if complex: field(name.imag, C-order big-endian float64 bytes)]

    Each field is encoded as:
        uint16 tag_length + tag_bytes + uint64 payload_length + payload_bytes
    """
    def _field(tag: bytes, payload: bytes) -> bytes:
        return struct.pack(">H", len(tag)) + tag + struct.pack(">Q", len(payload)) + payload

    def _shape_bytes(array: np.ndarray) -> bytes:
        return struct.pack(">I", array.ndim) + b"".join(
            struct.pack(">Q", int(size)) for size in array.shape
        )

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


# ──────────────────────────────────────────────────────────────────────────────
# Interval arithmetic backends
# ──────────────────────────────────────────────────────────────────────────────

def _arb_rayleigh(H: np.ndarray, psi: np.ndarray) -> tuple[float, float, float, str]:
    """Compute ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ for *real* H and psi using Arb interval arithmetic.

    Uses a local ``ctx.prec = 128`` context and extracts endpoints with
    ``math.nextafter`` outward rounding to guarantee binary64 soundness.

    Returns (lower, upper, radius, backend_label).

    Fallback: when python-flint is not installed the fallback returns a
    numpy-float midpoint with ``radius=0.0``.  Callers that require rigorous
    bounds (``rayleigh_certificate``, ``verify_rayleigh_certificate``,
    ``verify_from_dict``) must check for flint availability **before** calling
    this function and raise ``ImportError`` themselves if it is absent.
    """
    try:
        from flint import arb, arb_mat, ctx  # type: ignore[import]

        n = len(psi)
        saved_prec = ctx.prec
        try:
            ctx.prec = 128

            s_row = arb_mat([[arb(float(psi[i])) for i in range(n)]])
            s_col = arb_mat([[arb(float(psi[i]))] for i in range(n)])
            H_mat = arb_mat([[arb(float(H[i, j])) for j in range(n)] for i in range(n)])

            denominator = (s_row * s_col)[0, 0]
            if denominator.contains(0):
                raise ValueError(
                    "Arb denominator ball contains zero; cannot certify. "
                    "Check that |ψ⟩ is non-zero and finite."
                )
            numerator = (s_row * (H_mat * s_col))[0, 0]
            quotient  = numerator / denominator

            lower = math.nextafter(float(quotient.lower()), -math.inf)
            upper = math.nextafter(float(quotient.upper()),  math.inf)

            if not (math.isfinite(lower) and math.isfinite(upper)):
                raise ValueError(
                    f"Arb ball endpoints are not finite after export: "
                    f"lower={lower}, upper={upper}"
                )
            radius = (upper - lower) / 2
            return lower, upper, radius, "flint-arb/prec=128"

        finally:
            ctx.prec = saved_prec

    except ImportError:
        norm_sq = float(psi @ psi)
        mid = float(psi @ H @ psi) / norm_sq
        return mid, mid, 0.0, "numpy-float (no certified rounding; install python-flint)"


def _acb_rayleigh(H: np.ndarray, psi: np.ndarray) -> tuple[float, float, float, str]:
    """Compute ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ for complex H and/or psi using Acb ball arithmetic.

    Uses a local ``ctx.prec = 128`` context.  Checks that the imaginary part
    ball contains zero (exact criterion for Hermitian H).

    Returns (lower, upper, radius, backend_label) for the REAL part.

    Fallback: same policy as ``_arb_rayleigh`` — callers requiring rigorous
    bounds must guard against absent flint before calling.
    """
    try:
        from flint import acb, acb_mat, ctx  # type: ignore[import]

        n = len(psi)
        H_c   = H.astype(complex)
        psi_c = psi.astype(complex)

        saved_prec = ctx.prec
        try:
            ctx.prec = 128

            psi_dag = acb_mat([[acb(complex(psi_c[i].conjugate())) for i in range(n)]])
            psi_col = acb_mat([[acb(complex(psi_c[i]))]              for i in range(n)])
            H_acb   = acb_mat([[acb(complex(H_c[i, j])) for j in range(n)] for i in range(n)])

            den = (psi_dag * psi_col)[0, 0]
            if den.real.contains(0):
                raise ValueError(
                    "Acb denominator real part ball contains zero; cannot certify. "
                    "Check that |ψ⟩ is non-zero and finite."
                )
            num = (psi_dag * (H_acb * psi_col))[0, 0]
            q   = num / den

            if not q.imag.contains(0):
                raise ValueError(
                    "Rayleigh quotient imaginary part ball does not contain zero — "
                    "check that H is exactly Hermitian"
                )

            lower = math.nextafter(float(q.real.lower()), -math.inf)
            upper = math.nextafter(float(q.real.upper()),  math.inf)

            if not (math.isfinite(lower) and math.isfinite(upper)):
                raise ValueError(
                    f"Acb real endpoints are not finite after export: "
                    f"lower={lower}, upper={upper}"
                )
            radius    = (upper - lower) / 2
            imag_rad  = float(q.imag.rad())
            return lower, upper, radius, f"flint-acb/prec=128 (im_ball_rad={imag_rad:.2e})"

        finally:
            ctx.prec = saved_prec

    except ImportError:
        norm_sq = float(np.real(psi.conj() @ psi))
        mid = float(np.real(psi.conj() @ H @ psi)) / norm_sq
        return mid, mid, 0.0, "numpy-complex (no certified rounding; install python-flint)"
