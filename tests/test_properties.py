"""P2-C: Property-based tests for core certified pipeline (hypothesis).

Tests mathematical invariants that must hold for all valid inputs:

1. Rayleigh-Ritz bound    : upper >= true_E0 for any Hermitian H, non-zero psi
2. Estimate accuracy      : rayleigh_estimate value matches Re(<psi|H|psi>/<psi|psi>)
3. Serialisation roundtrip: to_dict() → from_dict() is lossless
4. Digest determinism     : _canonical_digest returns equal strings on repeated calls
5. verify idempotent      : verify_from_dict returns same result on repeated calls
6. Assurance invariant    : rayleigh_certificate always has assurance="rigorous"
7. Heuristic invariant    : rayleigh_estimate always has assurance="heuristic"
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

# ── strategies ────────────────────────────────────────────────────────────────

# Dimensions to test: small enough to keep Arb contraction fast
_DIMS = st.integers(min_value=2, max_value=6)

# Well-conditioned float values (no inf/nan, not too extreme)
_FLOATS = st.floats(min_value=-10.0, max_value=10.0,
                    allow_nan=False, allow_infinity=False)


@st.composite
def real_symmetric_H(draw) -> np.ndarray:
    """Draw a random real symmetric matrix of size n×n."""
    n = draw(_DIMS)
    vals = draw(st.lists(_FLOATS, min_size=n * n, max_size=n * n))
    A = np.array(vals, dtype=float).reshape(n, n)
    return A + A.T  # symmetric; eigenvalues are real


@st.composite
def real_psi(draw, n: int | None = None) -> tuple[np.ndarray, int]:
    """Draw a non-zero real state vector of length n (or random n if None)."""
    if n is None:
        n = draw(_DIMS)
    vals = draw(st.lists(_FLOATS, min_size=n, max_size=n))
    psi = np.array(vals, dtype=float)
    assume(np.linalg.norm(psi) > 1e-8)  # reject near-zero vectors
    return psi, n


@st.composite
def hermitian_H_and_psi(draw) -> tuple[np.ndarray, np.ndarray]:
    """Draw a real symmetric H and a compatible non-zero psi."""
    n = draw(_DIMS)
    vals = draw(st.lists(_FLOATS, min_size=n * n, max_size=n * n))
    A = np.array(vals, dtype=float).reshape(n, n)
    H = A + A.T
    psi_vals = draw(st.lists(_FLOATS, min_size=n, max_size=n))
    psi = np.array(psi_vals, dtype=float)
    assume(np.linalg.norm(psi) > 1e-8)
    return H, psi


# ── Property 1: Rayleigh-Ritz bound ──────────────────────────────────────────

@given(hermitian_H_and_psi())
@settings(max_examples=50, deadline=5000)
def test_rayleigh_ritz_upper_bounds_E0(H_psi):
    """upper >= true_E0 for any Hermitian H and non-zero psi."""
    from htf.rayleigh_cert import rayleigh_certificate
    H, psi = H_psi
    true_E0 = float(np.linalg.eigvalsh(H)[0])
    cert = rayleigh_certificate(H, psi)
    assert cert.upper >= true_E0 - 1e-9, (
        f"Rayleigh-Ritz violated: upper={cert.upper:.17g}, E0={true_E0:.17g}"
    )


# ── Property 2: Estimate value matches formula ────────────────────────────────

@given(hermitian_H_and_psi())
@settings(max_examples=80, deadline=3000)
def test_rayleigh_estimate_matches_formula(H_psi):
    """rayleigh_estimate().upper == Re(<psi|H|psi>) / <psi|psi>."""
    from htf.rayleigh_cert import rayleigh_estimate
    H, psi = H_psi
    est = rayleigh_estimate(H, psi)
    expected = float(psi @ H @ psi) / float(psi @ psi)
    assert math.isfinite(expected)
    assert abs(est.upper - expected) < 1e-10, (
        f"estimate={est.upper:.17g}, formula={expected:.17g}, diff={abs(est.upper-expected):.2e}"
    )


# ── Property 3: Serialisation roundtrip ──────────────────────────────────────

@given(hermitian_H_and_psi())
@settings(max_examples=40, deadline=5000)
def test_to_dict_from_dict_roundtrip(H_psi):
    """to_dict() → from_dict() preserves claim, upper, lower, radius, assurance."""
    from htf.rayleigh_cert import RayleighCertificate, rayleigh_certificate
    H, psi = H_psi
    cert = rayleigh_certificate(H, psi)
    d = cert.to_dict()
    cert2 = RayleighCertificate.from_dict(d)
    assert cert2.claim == cert.claim
    assert abs(cert2.upper - cert.upper) < 1e-15
    assert abs(cert2.lower - cert.lower) < 1e-15
    assert cert2.assurance == cert.assurance
    assert cert2.verified == cert.verified
    assert cert2.input_digest == cert.input_digest


@given(hermitian_H_and_psi())
@settings(max_examples=40, deadline=3000)
def test_rayleigh_estimate_roundtrip(H_psi):
    """rayleigh_estimate to_dict/from_dict roundtrip preserves heuristic assurance."""
    from htf.rayleigh_cert import RayleighCertificate, rayleigh_estimate
    H, psi = H_psi
    est = rayleigh_estimate(H, psi)
    d = est.to_dict()
    est2 = RayleighCertificate.from_dict(d)
    assert est2.assurance == "heuristic"
    assert abs(est2.upper - est.upper) < 1e-15


# ── Property 4: Digest determinism ───────────────────────────────────────────

@given(hermitian_H_and_psi())
@settings(max_examples=60, deadline=2000)
def test_digest_is_deterministic(H_psi):
    """_canonical_digest(H, psi) returns the same string on two calls."""
    from htf._rayleigh_primitives import _canonical_digest
    H, psi = H_psi
    d1 = _canonical_digest(H, psi)
    d2 = _canonical_digest(H, psi)
    assert d1 == d2
    assert isinstance(d1, str)
    assert len(d1) == 64  # SHA-256 hex


# ── Property 5: verify_from_dict idempotent ───────────────────────────────────

@given(hermitian_H_and_psi())
@settings(max_examples=30, deadline=8000)
def test_verify_from_dict_idempotent(H_psi):
    """Calling verify_from_dict twice on the same cert returns the same result."""
    from htf.rayleigh_cert import rayleigh_certificate
    from htf.verify import verify_from_dict
    H, psi = H_psi
    full = rayleigh_certificate(H, psi).to_full_dict()
    r1 = verify_from_dict(full)
    r2 = verify_from_dict(full)
    assert r1["verified"] == r2["verified"]
    assert r1["digest_match"] == r2["digest_match"]
    assert r1["stored_upper"] == r2["stored_upper"]


# ── Property 6: Assurance invariant (rigorous path) ───────────────────────────

@given(hermitian_H_and_psi())
@settings(max_examples=40, deadline=5000)
def test_rayleigh_certificate_always_rigorous(H_psi):
    """rayleigh_certificate() always produces assurance='rigorous', verified=False.

    verified=False by design: the producer does not self-verify.
    Call verify_rayleigh_certificate() to promote to verified=True.
    """
    from htf.rayleigh_cert import rayleigh_certificate
    H, psi = H_psi
    cert = rayleigh_certificate(H, psi)
    assert cert.assurance == "rigorous"
    assert cert.verified is False  # producer never self-verifies
    assert cert.radius > 0.0  # Arb ball has nonzero radius


# ── Property 7: Assurance invariant (heuristic path) ─────────────────────────

@given(hermitian_H_and_psi())
@settings(max_examples=60, deadline=2000)
def test_rayleigh_estimate_always_heuristic(H_psi):
    """rayleigh_estimate() always produces assurance='heuristic', verified=False."""
    from htf.rayleigh_cert import rayleigh_estimate
    H, psi = H_psi
    est = rayleigh_estimate(H, psi)
    assert est.assurance == "heuristic"
    assert est.verified is False
    assert est.radius == 0.0


# ── Property 8: Tight-bound monotonicity ─────────────────────────────────────

@given(real_symmetric_H())
@settings(max_examples=30, deadline=8000)
def test_ground_state_psi_gives_tight_bound(H):
    """Using the true ground-state eigenvector gives upper ≈ E0."""
    from htf.rayleigh_cert import rayleigh_certificate
    # H may be degenerate; eigvalsh always gives the true minimum
    eigvals, eigvecs = np.linalg.eigh(H)
    true_E0 = float(eigvals[0])
    psi = eigvecs[:, 0]
    assume(np.linalg.norm(psi) > 1e-8)
    cert = rayleigh_certificate(H, psi)
    # The Rayleigh quotient of the true ground state equals E0 exactly;
    # the Arb ball should give upper in [E0 - eps, E0 + small_radius]
    assert cert.upper >= true_E0 - 1e-9
    assert cert.upper <= true_E0 + 1e-4  # loose; Arb radius is ~1e-30 typically


# ── Property 9: verify_from_dict rejects all heuristic certs ─────────────────

@given(hermitian_H_and_psi())
@settings(max_examples=40, deadline=3000)
def test_verify_always_rejects_heuristic(H_psi):
    """verify_from_dict must return verified=False for every heuristic cert."""
    from htf.rayleigh_cert import rayleigh_estimate
    from htf.verify import verify_from_dict
    H, psi = H_psi
    d = rayleigh_estimate(H, psi).to_full_dict()
    result = verify_from_dict(d)
    assert result["verified"] is False
