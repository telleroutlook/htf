"""HTF independent verifier.

Reads a full Rayleigh certificate JSON (produced by ``htf rayleigh --full``)
and re-derives the bound from scratch, without reusing any intermediate state
from the producer.

This is the G1 gate deliverable: a clean-room re-check that confirms
``E0 ≤ upper`` from the stored canonical inputs alone.

Usage::

    htf-verify certificate_full.json
    python -m htf.verify certificate_full.json

Exit codes: 0 = verified, 1 = failed, 2 = usage/IO error.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


def verify_from_dict(full_cert: dict) -> dict:
    """Re-derive and confirm a RayleighCertificate from its full serialisation.

    Parameters
    ----------
    full_cert : dict produced by :meth:`RayleighCertificate.to_full_dict`.

    Returns
    -------
    result dict with keys ``verified``, ``stored_upper``,
    ``recomputed_upper``, ``digest_match``, ``message``.

    Raises ``ValueError`` on structural problems with the cert dict itself.
    Raises ``ImportError`` when python-flint is not installed.
    """
    # Require flint: without it the _arb_rayleigh fallback returns the same
    # float midpoint as the stored value, so recomputed_upper == stored_upper
    # trivially — that is not independent verification.
    try:
        import flint  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "verify_from_dict() requires python-flint "
            "(pip install python-flint)."
        ) from exc

    # --- imports from _rayleigh_primitives so verify has no dependency on rayleigh_cert ---
    from ._rayleigh_primitives import (
        EXPECTED_THEOREM,
        SCHEMA_VERSION,
        _acb_rayleigh,
        _arb_rayleigh,
        _canonical_digest,
        _check_preconditions,
        _decode_canonical,
        _mpmath_rayleigh,
    )

    required = {"schema_version", "claim", "input_digest", "interval", "canonical", "assurance", "backend"}
    missing = required - set(full_cert)
    if missing:
        raise ValueError(f"Certificate dict missing required keys: {missing}")

    sv = full_cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION!r}; got {sv!r}"
        )

    # Reject non-rigorous certificates before attempting verification.
    # A numpy-float backend carries radius=0.0 and would trivially pass the
    # recomputed_upper <= stored_upper check — that is not independent verification.
    backend = full_cert.get("backend", "")
    assurance = full_cert["assurance"]
    if assurance != "rigorous" or "numpy" in backend.lower():
        return {
            "verified": False,
            "stored_upper": full_cert["interval"]["upper"],
            "recomputed_upper": None,
            "digest_match": None,
            "message": (
                f"FAIL — certificate assurance={assurance!r} / backend={backend!r} "
                "is not rigorous interval arithmetic; independent verification requires "
                "a certificate produced by rayleigh_certificate() with python-flint."
            ),
        }

    stored_upper = float(full_cert["interval"]["upper"])

    # Claim must encode the stored upper value in the canonical format.
    expected_claim = (
        f"E0 ≤ {stored_upper:.17g}  [Rayleigh-Ritz upper bound on ground-state energy]"
    )
    stored_claim = full_cert.get("claim", "")
    if stored_claim != expected_claim:
        return {
            "verified": False,
            "stored_upper": stored_upper,
            "recomputed_upper": None,
            "digest_match": None,
            "message": (
                f"FAIL — claim text does not match interval.upper:\n"
                f"  expected: {expected_claim!r}\n"
                f"  stored:   {stored_claim!r}"
            ),
        }

    # Theorem must match the canonical Rayleigh-Ritz statement.
    stored_theorem = full_cert.get("theorem", "")
    if stored_theorem != EXPECTED_THEOREM:
        return {
            "verified": False,
            "stored_upper": stored_upper,
            "recomputed_upper": None,
            "digest_match": None,
            "message": (
                f"FAIL — theorem text has been tampered:\n"
                f"  expected: {EXPECTED_THEOREM!r}\n"
                f"  stored:   {stored_theorem!r}"
            ),
        }

    canonical = full_cert["canonical"]
    if "H" not in canonical or "psi" not in canonical:
        raise ValueError("canonical section must contain 'H' and 'psi'")
    H   = _decode_canonical(canonical["H"])
    psi = _decode_canonical(canonical["psi"])

    # 1. Recompute digest from canonical inputs
    recomputed_digest = _canonical_digest(H, psi)
    stored_digest     = full_cert["input_digest"]
    digest_match      = (recomputed_digest == stored_digest)

    if not digest_match:
        return {
            "verified": False,
            "stored_upper": full_cert["interval"]["upper"],
            "recomputed_upper": None,
            "digest_match": False,
            "message": (
                f"FAIL — digest mismatch: "
                f"stored={stored_digest!r}, recomputed={recomputed_digest!r}"
            ),
        }

    # 2. Precondition checks
    try:
        _check_preconditions(H, psi)
    except (ValueError, TypeError) as exc:
        return {
            "verified": False,
            "stored_upper": full_cert["interval"]["upper"],
            "recomputed_upper": None,
            "digest_match": True,
            "message": f"FAIL — precondition check: {exc}",
        }

    # 3. Recompute Rayleigh interval (choose real or complex path)
    is_complex = np.iscomplexobj(H) or np.iscomplexobj(psi)
    if is_complex:
        recomputed_lower, recomputed_upper, recomputed_radius, recomputed_backend = _acb_rayleigh(H, psi)
    else:
        recomputed_lower, recomputed_upper, recomputed_radius, recomputed_backend = _arb_rayleigh(H, psi)

    if not math.isfinite(recomputed_upper):
        return {
            "verified": False,
            "stored_upper": stored_upper,
            "recomputed_upper": recomputed_upper,
            "digest_match": True,
            "message": f"FAIL — recomputed upper is not finite: {recomputed_upper}",
        }
    if not math.isfinite(stored_upper):
        return {
            "verified": False,
            "stored_upper": stored_upper,
            "recomputed_upper": recomputed_upper,
            "digest_match": True,
            "message": f"FAIL — stored upper is not finite: {stored_upper}",
        }

    if not (recomputed_upper <= stored_upper):
        return {
            "verified": False,
            "stored_upper": stored_upper,
            "recomputed_upper": recomputed_upper,
            "digest_match": True,
            "message": (
                f"FAIL — recomputed upper {recomputed_upper:.17g} exceeds "
                f"stored upper {stored_upper:.17g}"
            ),
        }

    # 4. Semantic field checks — backend, lower, radius must match recomputed values.
    # These catch tampered metadata that would not be detected by digest or upper checks.
    _rtol = 1e-12
    stored_backend = full_cert.get("backend", "")
    if stored_backend != recomputed_backend:
        return {
            "verified": False,
            "stored_upper": stored_upper,
            "recomputed_upper": recomputed_upper,
            "digest_match": True,
            "message": (
                f"FAIL — backend field tampered:\n"
                f"  expected: {recomputed_backend!r}\n"
                f"  stored:   {stored_backend!r}"
            ),
        }

    stored_lower = float(full_cert["interval"]["lower"])
    if not math.isfinite(stored_lower):
        return {
            "verified": False,
            "stored_upper": stored_upper,
            "recomputed_upper": recomputed_upper,
            "digest_match": True,
            "message": f"FAIL — interval.lower is not finite: {stored_lower}",
        }
    lower_tol = _rtol * max(1.0, abs(recomputed_lower))
    if abs(stored_lower - recomputed_lower) > lower_tol:
        return {
            "verified": False,
            "stored_upper": stored_upper,
            "recomputed_upper": recomputed_upper,
            "digest_match": True,
            "message": (
                f"FAIL — interval.lower tampered: "
                f"stored={stored_lower:.17g}, recomputed={recomputed_lower:.17g}"
            ),
        }

    stored_radius = float(full_cert["interval"]["radius"])
    if not math.isfinite(stored_radius):
        return {
            "verified": False,
            "stored_upper": stored_upper,
            "recomputed_upper": recomputed_upper,
            "digest_match": True,
            "message": f"FAIL — interval.radius is not finite: {stored_radius}",
        }
    # radius is derived from lo/up as nextafter(max(mid-lo, up-mid), inf),
    # not the raw Arb ball radius — recompute from the verified endpoints.
    _mid = (recomputed_lower + recomputed_upper) / 2
    expected_radius = math.nextafter(
        max(_mid - recomputed_lower, recomputed_upper - _mid), math.inf
    )
    radius_tol = _rtol * max(1e-300, expected_radius)
    if abs(stored_radius - expected_radius) > radius_tol:
        return {
            "verified": False,
            "stored_upper": stored_upper,
            "recomputed_upper": recomputed_upper,
            "digest_match": True,
            "message": (
                f"FAIL — interval.radius tampered: "
                f"stored={stored_radius:.17g}, recomputed={recomputed_radius:.17g}"
            ),
        }

    # 5. Numpy cross-check (C3 partial independence).
    # Compute the Rayleigh quotient via numpy, independently of _arb_rayleigh.
    # This catches bugs where the Arb interval is computed from wrong inputs
    # or the interval arithmetic itself has a systematic error.
    numpy_rq = float(np.real(psi.conj() @ H @ psi)) / float(np.real(psi.conj() @ psi))
    _interval_tol = max(1e-10, (recomputed_upper - recomputed_lower) * 10)
    if not (recomputed_lower - _interval_tol <= numpy_rq <= recomputed_upper + _interval_tol):
        return {
            "verified": False,
            "stored_upper": stored_upper,
            "recomputed_upper": recomputed_upper,
            "digest_match": True,
            "message": (
                f"FAIL — numpy cross-check: Rayleigh quotient {numpy_rq:.17g} "
                f"falls outside Arb interval "
                f"[{recomputed_lower:.17g}, {recomputed_upper:.17g}]"
            ),
        }

    # 6. mpmath cross-check (C3 full arithmetic independence).
    # Uses mpmath at 2× the production precision as a third independent arithmetic
    # path.  Detects systematic bugs in _arb_rayleigh that numpy (float64) could
    # silently mask.  mpmath is an optional dependency; if absent the check is
    # skipped and the result includes cross_check="skipped".
    _mpmath_cross_check: str | None
    try:
        mp_lower, _mp_upper, _, mp_backend = _mpmath_rayleigh(H, psi)
        # The stored_upper is a certified Arb upper bound on the true Rayleigh
        # quotient.  The mpmath lower bound (2-ULP below the mpmath midpoint) must
        # not exceed stored_upper by more than a loose relative tolerance; if it
        # does, the Arb computation produced an incorrectly small upper bound.
        _loose_tol = max(1e-6, 1e-6 * abs(mp_lower))
        if mp_lower > stored_upper + _loose_tol:
            return {
                "verified": False,
                "stored_upper": stored_upper,
                "recomputed_upper": recomputed_upper,
                "digest_match": True,
                "message": (
                    f"FAIL — mpmath cross-check ({mp_backend}): "
                    f"mpmath lower bound {mp_lower:.17g} exceeds "
                    f"stored upper {stored_upper:.17g} beyond tolerance; "
                    "flint-arb may have produced an incorrect upper bound"
                ),
            }
        _mpmath_cross_check = f"PASS ({mp_backend})"
    except ImportError:
        _mpmath_cross_check = "skipped (mpmath not installed)"

    return {
        "verified": True,
        "stored_upper": stored_upper,
        "recomputed_upper": recomputed_upper,
        "digest_match": True,
        "backend": recomputed_backend,
        "cross_check": _mpmath_cross_check,
        "message": (
            f"PASS — E0 ≤ {stored_upper:.17g} independently confirmed "
            f"(recomputed upper = {recomputed_upper:.17g})"
        ),
    }


def verify_file(path: str | Path) -> dict:
    """Load a full certificate JSON file and verify it."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Certificate file not found: {p}")
    with p.open() as fh:
        full_cert = json.load(fh)
    return verify_from_dict(full_cert)


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print("Usage: htf-verify <certificate_full.json>", file=sys.stderr)
        print("       Independently verifies a Rayleigh Certificate.", file=sys.stderr)
        sys.exit(2)

    cert_path = argv[0]
    try:
        result = verify_file(cert_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"verified": False, "message": str(exc)}), flush=True)
        sys.exit(2)

    print(json.dumps(result, indent=2), flush=True)
    sys.exit(0 if result["verified"] else 1)


if __name__ == "__main__":
    main()
