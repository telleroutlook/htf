"""HTF — Validated Rayleigh Certificate  (schema: rayleigh-cert/v2).

Certifies ``E0 ≤ upper`` for a self-adjoint Hamiltonian H (real symmetric
or complex Hermitian) and a non-zero trial state |ψ⟩, using Arb/Acb interval
arithmetic via ``python-flint``.

What this *does*
---------------
* Machine-checks: H is **exactly** real symmetric or complex Hermitian, all
  entries finite, ⟨ψ|ψ⟩ exactly non-zero.
* Computes the Rayleigh quotient in a fixed-precision (128-bit) Arb/Acb local
  context.  Uses ``q.upper()`` with ``math.nextafter`` outward rounding to
  guarantee the stored ``upper`` is a sound binary64 upper bound.
* Emits a :class:`RayleighCertificate` conforming to schema
  ``rayleigh-cert/v2`` (see ``htf/schemas/rayleigh_cert_v2.json``).
* Provides an independent :func:`verify_rayleigh_certificate` that re-derives
  the upper bound from stored canonical inputs with no floating-point tolerance.

What this *does not*
--------------------
* Bond-dimension truncation bias is ``[OUT]``.
* Modeling error is ``[OUT]``.
* The continuum limit is ``[OUT]``.
* This is *not* a spectral-gap bound — it only certifies ``E0 ≤ upper``.

Changes from v1 → v2
--------------------
* Schema version bumped; v1 certificates are **not** automatically certified.
* Precondition A3 replaced: exact ``H == H†`` required (not ``max|H-H†| ≤ ε``).
* Precondition A4 replaced: exact ``np.any(psi != 0)`` required.
* NaN/Inf inputs now fail closed before any arithmetic.
* Arb/Acb export: local ``ctx.prec = 128``; ``q.upper()`` + ``nextafter``
  instead of ``float(mid) + float(rad)`` (which was not outward-rounded).
* Complex Acb path: ``q.imag.contains(0)`` replaces the ``1e-8`` threshold.
* Digest: domain-separated, shape-tagged, fixed big-endian v2 encoding.
* Codec: complex decode uses component assignment (preserves ``-0.0``).
* Verifier: strict ``upper_v > stored_upper`` (no tolerance); checks claim
  text, interval consistency via exact Fraction; ignores incoming ``verified``.

Threat model
------------
* **Input substitution** — tagged SHA-256 ``input_digest`` binds the
  certificate to exact (H, ψ) shape, dtype domain, and big-endian bytes.
* **Backend downgrade** — ``numpy-float`` certificates carry ``radius=0.0``
  but are not independently certified; treat ``backend != "flint-arb/prec=128"``
  / ``"flint-acb/prec=128"`` as discovery-tier only.
* **Stale HTF version** — ``htf_version`` records the issuer version.
* **Truncation / modelling / continuum-limit error** — outside scope (``[OUT]``).

Replay recipe
-------------
1. Load the full JSON (``cert.to_full_json()``).
2. Re-derive H and ψ from the ``canonical`` section.
3. Confirm ``SHA-256(tagged_bytes(H, ψ)) == input_digest``.
4. Recompute Rayleigh quotient with Arb/Acb at ``prec=128``; confirm
   ``upper_v ≤ stored upper`` (strict, no tolerance).
5. Confirm all ``assumptions`` independently.

Call :func:`verify_rayleigh_certificate` to perform steps 3–5 automatically.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from ._rayleigh_primitives import (
    SCHEMA_VERSION,
    _acb_rayleigh,
    _arb_rayleigh,
    _canonical_digest,
    _check_preconditions,
    _decode_canonical,
    _encode_canonical,
)

# ──────────────────────────────────────────────────────────────────────────────
# Dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RayleighCertificate:
    """A validated certificate that ``E0 ≤ upper``, conforming to schema
    ``rayleigh-cert/v2``.

    Fields
    ------
    schema_version : schema identifier; always ``"rayleigh-cert/v2"``.
    claim          : human-readable statement of what is certified.
    theorem        : the mathematical theorem invoked (Rayleigh-Ritz).
    assumptions    : list of machine-checked preconditions.
    input_digest   : tagged SHA-256 hex digest of canonical (H, ψ) inputs.
    lower          : lower endpoint of the Arb/Acb real interval.
    upper          : upper endpoint (the certified E0 bound).
    midpoint       : midpoint of the interval (display only).
    radius         : half-width (floating-point rounding radius).
    backend        : arithmetic backend used.
    htf_version    : version of HTF that produced this certificate.
    assurance      : machine-readable assurance level:
                     ``"rigorous"``     — flint Arb/Acb, independently verifiable.
                     ``"reproducible"`` — float, digest-bound, not rigorous.
                     ``"heuristic"``    — float estimate, no digest binding.
    verified       : True if :func:`verify_rayleigh_certificate` confirmed.
    notes          : additional context.
    """
    claim: str
    theorem: str
    assumptions: list[str]
    input_digest: str
    lower: float
    upper: float
    midpoint: float
    radius: float
    backend: str
    htf_version: str
    schema_version: str = SCHEMA_VERSION
    assurance: str = "rigorous"
    verified: bool = False
    notes: str = ""

    # Stored for independent replay; not exposed in to_dict().
    # For real inputs: list[list[float]] / list[float].
    # For complex inputs: {"real": ..., "imag": ...} dicts.
    _H_canonical: Any = field(default_factory=list, repr=False)
    _psi_canonical: Any = field(default_factory=list, repr=False)

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict (schema ``rayleigh-cert/v2``)."""
        return {
            "schema_version": self.schema_version,
            "claim": self.claim,
            "theorem": self.theorem,
            "assumptions": self.assumptions,
            "input_digest": self.input_digest,
            "interval": {
                "lower": self.lower,
                "upper": self.upper,
                "midpoint": self.midpoint,
                "radius": self.radius,
            },
            "backend": self.backend,
            "htf_version": self.htf_version,
            "assurance": self.assurance,
            "verified": self.verified,
            "notes": self.notes,
        }

    def to_full_dict(self) -> dict:
        """Full serialisation including canonical inputs for independent replay."""
        d = self.to_dict()
        d["canonical"] = {
            "H": self._H_canonical,
            "psi": self._psi_canonical,
        }
        return d

    def to_full_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_full_dict(), ensure_ascii=False, **kwargs)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)

    # ── validation ────────────────────────────────────────────────────────────

    def validate(self) -> None:
        """Validate against the ``rayleigh-cert/v2`` schema (no Arb re-run).

        Raises ``ValueError`` on any structural violation.
        """
        validate_certificate_dict(self.to_dict())

    # ── deserialisation ───────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: dict) -> RayleighCertificate:
        """Reconstruct from a ``to_dict()`` or ``to_full_dict()`` dict.

        Validates the dict against the v2 schema first.
        ``_H_canonical`` / ``_psi_canonical`` are populated only when a
        ``canonical`` section is present (from ``to_full_dict()``).

        Raises ``ValueError`` if the dict does not conform to v2 schema.
        """
        validate_certificate_dict(d)
        iv = d["interval"]
        canonical = d.get("canonical", {})
        return cls(
            claim=d["claim"],
            theorem=d["theorem"],
            assumptions=list(d["assumptions"]),
            input_digest=d["input_digest"],
            lower=float(iv["lower"]),
            upper=float(iv["upper"]),
            midpoint=float(iv["midpoint"]),
            radius=float(iv["radius"]),
            backend=d["backend"],
            htf_version=d["htf_version"],
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            assurance=str(d.get("assurance", "rigorous")),
            verified=bool(d.get("verified", False)),
            notes=str(d.get("notes", "")),
            _H_canonical=canonical.get("H", []),
            _psi_canonical=canonical.get("psi", []),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Schema validation (no external dependency)
# ──────────────────────────────────────────────────────────────────────────────

_REQUIRED_KEYS = {
    "schema_version", "claim", "theorem", "assumptions",
    "input_digest", "interval", "backend", "htf_version", "verified",
}
_INTERVAL_KEYS = {"lower", "upper", "midpoint", "radius"}
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def validate_certificate_dict(d: dict) -> None:
    """Validate a certificate dict against the ``rayleigh-cert/v2`` schema.

    Raises ``ValueError`` with a descriptive message on the first violation.
    """
    if not isinstance(d, dict):
        raise ValueError(f"certificate must be a dict; got {type(d).__name__}")

    missing = _REQUIRED_KEYS - d.keys()
    if missing:
        raise ValueError(f"certificate missing required keys: {sorted(missing)}")

    sv = d["schema_version"]
    if sv != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION!r}; got {sv!r}"
        )

    digest = d["input_digest"]
    if not isinstance(digest, str) or not _DIGEST_RE.match(digest):
        raise ValueError(
            f"input_digest must be a 64-char lowercase hex string; got {digest!r}"
        )

    iv = d["interval"]
    if not isinstance(iv, dict):
        raise ValueError(f"interval must be a dict; got {type(iv).__name__}")
    missing_iv = _INTERVAL_KEYS - iv.keys()
    if missing_iv:
        raise ValueError(f"interval missing keys: {sorted(missing_iv)}")

    for key in ("lower", "upper", "midpoint", "radius"):
        val = iv[key]
        if not isinstance(val, (int, float)):
            raise ValueError(f"interval.{key} must be numeric; got {type(val).__name__}")
        if not math.isfinite(val):
            raise ValueError(f"interval.{key} must be finite; got {val}")

    lower_q = Fraction.from_float(float(iv["lower"]))
    upper_q = Fraction.from_float(float(iv["upper"]))
    mid_q   = Fraction.from_float(float(iv["midpoint"]))
    rad_q   = Fraction.from_float(float(iv["radius"]))

    if not (lower_q <= mid_q <= upper_q):
        raise ValueError(
            f"interval ordering violated: lower={iv['lower']} <= "
            f"midpoint={iv['midpoint']} <= upper={iv['upper']} required"
        )
    if rad_q < 0:
        raise ValueError(f"interval.radius must be >= 0; got {iv['radius']}")
    required_rad = max(mid_q - lower_q, upper_q - mid_q)
    if rad_q < required_rad:
        raise ValueError(
            f"interval.radius {iv['radius']} does not cover both endpoints: "
            f"need >= {float(required_rad):.6g}"
        )

    if not isinstance(d["assumptions"], list) or not d["assumptions"]:
        raise ValueError("assumptions must be a non-empty list")
    if not isinstance(d["verified"], bool):
        raise ValueError(f"verified must be bool; got {type(d['verified']).__name__}")
    if "assurance" in d:
        _valid_assurance = {"rigorous", "reproducible", "heuristic"}
        if d["assurance"] not in _valid_assurance:
            raise ValueError(
                f"assurance must be one of {sorted(_valid_assurance)}; got {d['assurance']!r}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _htf_version() -> str:
    """Return the installed HTF version with pyproject.toml fallback."""
    try:
        from importlib.metadata import version as _v
        ver = _v("htf")
        _root = Path(__file__).parent.parent
        toml_path = _root / "pyproject.toml"
        if toml_path.exists():
            text = toml_path.read_text()
            m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
            if m and m.group(1) != ver:
                return m.group(1)
        return ver
    except Exception:
        pass
    try:
        _root = Path(__file__).parent.parent
        text = (_root / "pyproject.toml").read_text()
        m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        if m:
            return m.group(1)
    except Exception:
        pass
    return "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def rayleigh_certificate(
    H: np.ndarray,
    psi: np.ndarray,
    *,
    notes: str = "",
) -> RayleighCertificate:
    """Produce a validated certificate that ``E0 ≤ upper``.

    Supports both real symmetric H and complex Hermitian H.  The Rayleigh-Ritz
    theorem guarantees that for any non-zero |ψ⟩::

        E0 ≤ Re(⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩)

    This function:

    1. Machine-checks all preconditions exactly (finite, exact Hermitian/
       symmetric, exact non-zero).
    2. Computes the Rayleigh quotient in a local Arb (real) or Acb (complex)
       context with ``ctx.prec = 128``, using outward-rounded binary64 export.
    3. Returns a :class:`RayleighCertificate` conforming to
       ``rayleigh-cert/v2`` with the certified claim ``E0 ≤ upper``.

    Parameters
    ----------
    H   : exactly real symmetric or exactly complex Hermitian Hamiltonian,
          shape (n, n), all entries finite.
    psi : trial state, shape (n,).  Need not be normalised.  Must be finite
          and have at least one non-zero component.

    Returns
    -------
    cert : :class:`RayleighCertificate` with ``verified=False``.
           Call :func:`verify_rayleigh_certificate` to set ``verified=True``.
    """
    # Fail closed: rigorous interval arithmetic requires python-flint.
    # Use rayleigh_estimate() for a float-only non-certified path.
    try:
        import flint  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "rayleigh_certificate() requires python-flint for rigorous interval "
            "arithmetic (pip install python-flint). "
            "For a float-only non-certified estimate use rayleigh_estimate()."
        ) from exc

    H_raw   = np.asarray(H)
    psi_raw = np.asarray(psi).ravel()

    is_complex = np.iscomplexobj(H_raw) or np.iscomplexobj(psi_raw)

    if is_complex:
        H   = H_raw.astype(complex)
        psi = psi_raw.astype(complex)
    else:
        H   = H_raw.astype(float)
        psi = psi_raw.astype(float)

    assumptions = _check_preconditions(H, psi)
    digest      = _canonical_digest(H, psi)

    if is_complex:
        lower, upper, _, backend = _acb_rayleigh(H, psi)
    else:
        lower, upper, _, backend = _arb_rayleigh(H, psi)

    midpoint = (lower + upper) / 2
    # Outward-round the radius by 1 ULP to ensure the stored float64 radius
    # always covers both endpoints under exact (Fraction) arithmetic.
    # float64 subtraction rounds to nearest, so nextafter guarantees:
    #   Fraction(radius) >= max(Fraction(midpoint)-Fraction(lower),
    #                           Fraction(upper)-Fraction(midpoint))
    radius = math.nextafter(max(midpoint - lower, upper - midpoint), math.inf)

    return RayleighCertificate(
        claim=f"E0 ≤ {upper:.17g}  [Rayleigh-Ritz upper bound on ground-state energy]",
        theorem=(
            "Rayleigh-Ritz: for any non-zero |ψ⟩ and self-adjoint H, "
            "E0 ≤ Re(⟨ψ|H|ψ⟩/⟨ψ|ψ⟩)."
        ),
        assumptions=assumptions,
        input_digest=digest,
        lower=lower,
        upper=upper,
        midpoint=midpoint,
        radius=radius,
        backend=backend,
        htf_version=_htf_version(),
        assurance="rigorous",
        verified=False,
        notes=notes,
        _H_canonical=_encode_canonical(H),
        _psi_canonical=_encode_canonical(psi),
    )


def verify_rayleigh_certificate(cert: RayleighCertificate) -> RayleighCertificate:
    """Independently re-verify a :class:`RayleighCertificate`.

    Checks (all without floating-point tolerance):

    1. Schema validation (v2 invariants, exact Fraction interval checks).
    2. Digest matches the stored canonical (H, ψ) via the v2 tagged encoding.
    3. All preconditions still hold (exact finite / Hermitian / non-zero).
    4. Recomputed upper bound ≤ stored upper (strict, no tolerance).
    5. Claim text exactly matches ``interval.upper``.
    6. Interval lower/midpoint/upper/radius consistency (exact Fraction).

    The incoming ``verified`` flag is **ignored**; only independent arithmetic
    sets ``cert.verified = True``.

    Sets ``cert.verified = True`` and returns the updated certificate.
    Raises ``ValueError`` on any mismatch.
    Raises ``ImportError`` when python-flint is not installed (cannot verify
    without rigorous interval arithmetic).
    """
    try:
        import flint  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "verify_rayleigh_certificate() requires python-flint "
            "(pip install python-flint)."
        ) from exc

    # 1. Schema check
    cert.validate()

    H   = _decode_canonical(cert._H_canonical)
    psi = _decode_canonical(cert._psi_canonical)

    # 2. Digest check — exact, no tolerance
    recomputed_digest = _canonical_digest(H, psi)
    if recomputed_digest != cert.input_digest:
        raise ValueError(
            f"Input digest mismatch: stored {cert.input_digest!r} "
            f"vs recomputed {recomputed_digest!r}"
        )

    # 3. Preconditions
    _check_preconditions(H, psi)

    # 4. Recompute interval
    is_complex = np.iscomplexobj(H) or np.iscomplexobj(psi)
    if is_complex:
        _, upper_v, _, _ = _acb_rayleigh(H, psi)
    else:
        _, upper_v, _, _ = _arb_rayleigh(H, psi)

    # Fail closed on non-finite bounds
    if not math.isfinite(upper_v):
        raise ValueError(
            f"Recomputed upper bound is not finite: {upper_v}"
        )
    if not math.isfinite(cert.upper):
        raise ValueError(
            f"Stored upper bound is not finite: {cert.upper}"
        )

    # Strict comparison — no tolerance
    if upper_v > cert.upper:
        raise ValueError(
            f"Recomputed upper bound {upper_v:.17g} exceeds stored "
            f"upper bound {cert.upper:.17g}"
        )

    # 5. Claim text must match stored upper exactly
    expected_claim = (
        f"E0 ≤ {cert.upper:.17g}  "
        "[Rayleigh-Ritz upper bound on ground-state energy]"
    )
    if cert.claim != expected_claim:
        raise ValueError(
            f"claim text does not match interval.upper:\n"
            f"  expected: {expected_claim!r}\n"
            f"  stored:   {cert.claim!r}"
        )

    # 6. Interval consistency (exact Fraction arithmetic)
    lower_q = Fraction.from_float(cert.lower)
    upper_q = Fraction.from_float(cert.upper)
    mid_q   = Fraction.from_float(cert.midpoint)
    rad_q   = Fraction.from_float(cert.radius)
    if not (lower_q <= mid_q <= upper_q):
        raise ValueError(
            f"Interval ordering violated: lower={cert.lower} <= "
            f"midpoint={cert.midpoint} <= upper={cert.upper} required"
        )
    required_rad = max(mid_q - lower_q, upper_q - mid_q)
    if rad_q < required_rad:
        raise ValueError(
            f"Interval radius {cert.radius} does not cover both stored endpoints"
        )

    cert.verified = True
    return cert


def rayleigh_estimate(
    H: np.ndarray,
    psi: np.ndarray,
    *,
    notes: str = "",
) -> RayleighCertificate:
    """Float-only (non-certified) Rayleigh quotient estimate.

    This is the **non-rigorous** counterpart of :func:`rayleigh_certificate`.
    It does **not** require python-flint and produces no rigorous interval bound.
    The returned certificate carries ``assurance="heuristic"`` and
    ``verified=False``; it must never be treated as a proof of ``E0 ≤ upper``.

    Use this for quick discovery-tier estimates.  For a rigorous bound with
    independent verification, use :func:`rayleigh_certificate` +
    :func:`verify_rayleigh_certificate` (requires python-flint).

    Parameters
    ----------
    H   : real symmetric or complex Hermitian Hamiltonian, shape (n, n).
    psi : trial state, shape (n,).

    Returns
    -------
    cert : :class:`RayleighCertificate` with ``assurance="heuristic"``,
           ``radius=0.0``, and ``verified=False``.
    """
    H_raw   = np.asarray(H)
    psi_raw = np.asarray(psi).ravel()

    is_complex = np.iscomplexobj(H_raw) or np.iscomplexobj(psi_raw)
    H   = H_raw.astype(complex if is_complex else float)
    psi = psi_raw.astype(complex if is_complex else float)

    assumptions = _check_preconditions(H, psi)
    digest      = _canonical_digest(H, psi)

    norm_sq = float(np.real(psi.conj() @ psi))
    mid     = float(np.real(psi.conj() @ H @ psi)) / norm_sq

    return RayleighCertificate(
        claim=f"E0 ≤ {mid:.17g}  [Rayleigh-Ritz estimate — NOT rigorous, no interval arithmetic]",
        theorem=(
            "Rayleigh-Ritz: for any non-zero |ψ⟩ and self-adjoint H, "
            "E0 ≤ Re(⟨ψ|H|ψ⟩/⟨ψ|ψ⟩)."
        ),
        assumptions=assumptions,
        input_digest=digest,
        lower=mid,
        upper=mid,
        midpoint=mid,
        radius=0.0,
        backend="numpy-float (no certified rounding; install python-flint)",
        htf_version=_htf_version(),
        assurance="heuristic",
        verified=False,
        notes=(
            "Float estimate only — no rigorous interval arithmetic. "
            "radius=0.0 does NOT mean zero error; it means no bound was computed. "
            + (notes or "")
        ).strip(),
        _H_canonical=_encode_canonical(H),
        _psi_canonical=_encode_canonical(psi),
    )
