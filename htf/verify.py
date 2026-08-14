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
        SCHEMA_VERSION,
        _acb_rayleigh,
        _arb_rayleigh,
        _canonical_digest,
        _check_preconditions,
        _decode_canonical,
    )

    required = {"schema_version", "claim", "input_digest", "interval", "canonical"}
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
    assurance = full_cert.get("assurance", "rigorous")
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

    canonical = full_cert["canonical"]
    if "H" not in canonical or "psi" not in canonical:
        raise ValueError("canonical section must contain 'H' and 'psi'")

    # _decode_canonical handles both real (list) and complex ({"real":…,"imag":…}) inputs.
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
        _, recomputed_upper, _, backend = _acb_rayleigh(H, psi)
    else:
        _, recomputed_upper, _, backend = _arb_rayleigh(H, psi)
    stored_upper = float(full_cert["interval"]["upper"])

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

    if recomputed_upper > stored_upper:
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

    return {
        "verified": True,
        "stored_upper": stored_upper,
        "recomputed_upper": recomputed_upper,
        "digest_match": True,
        "backend": backend,
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
