# HTF-06 — Post-P0 Comprehensive External Review

**Type:** Gate-A external independent review (G5 gate deliverable)

**Context.** HTF-01 through HTF-05 were produced as the Rayleigh certificate
pipeline was built and debugged.  HTF-06 is the single comprehensive review
request for the pipeline *after* all P0 blockers have been applied.  The
reviewer needs nothing from the HTF repository; all definitions and code are
reproduced in full below.

**Non-circularity (mandatory).** No step assumes the claim it is asking to
verify.  The certificate claims `E0 ≤ upper`; this claim is not assumed anywhere
in the review questions.

---

## The claim under review

For a finite-dimensional self-adjoint operator H and a non-zero trial state |ψ⟩,
HTF produces a certificate asserting:

> **E0 ≤ upper**
>
> where `upper` is a binary64 floating-point number and E0 is the smallest
> eigenvalue of H.

The certificate schema is `rayleigh-cert/v2`.  The claim is trusted only when
`assurance == "rigorous"` and `backend` is `"flint-arb/prec=128"` (real H) or
`"flint-acb/prec=128"` (complex H).

---

## Pipeline overview (5 links in sequence)

```
H, psi (float64/complex128)
   │
   ▼  Link 1 — Precondition check (_check_preconditions)
   │   exact finite, exact Hermitian/symmetric, exact non-zero
   │
   ▼  Link 2 — Interval arithmetic (_arb_rayleigh or _acb_rayleigh)
   │   Arb/Acb at prec=128; outward-rounded binary64 export
   │
   ▼  Link 3 — Canonical digest (_canonical_digest)
   │   domain-separated SHA-256 binding H and psi to the certificate
   │
   ▼  Link 4 — Certificate production (rayleigh_certificate)
   │   assembles RayleighCertificate v2; assurance="rigorous"
   │
   ▼  Link 5 — Independent verification (verify_rayleigh_certificate / verify_from_dict)
              re-executes Links 1–3 from stored canonical inputs
              checks assurance / theorem / backend semantic fields
```

---

## All definitions and code (self-contained)

### Theorem invoked

**Rayleigh-Ritz.** Let H be a self-adjoint n×n matrix.  For any non-zero |ψ⟩:

    E0 ≤ Re(⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩)

*Reference:* Courant & Hilbert, *Methods of Mathematical Physics*, Vol. I, §VI.1.

The canonical theorem string stored in every certificate and re-checked by every
verifier is exactly:

    "Rayleigh-Ritz: for any non-zero |ψ⟩ and self-adjoint H, "
    "E0 ≤ Re(⟨ψ|H|ψ⟩/⟨ψ|ψ⟩)."

### Link 1 — Precondition check (complete code)

```python
def _check_preconditions(H: np.ndarray, psi: np.ndarray) -> list[str]:
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be a square 2-D array; got shape {H.shape}")
    n = H.shape[0]
    if psi.ndim != 1 or len(psi) != n:
        raise ValueError(
            f"|psi> must be a 1-D vector of length {n}; got shape {psi.shape}"
        )
    # Finite check first (NaN comparisons are False → fail-open if skipped)
    if not np.all(np.isfinite(H)):
        raise ValueError("H contains non-finite values (NaN or Infinity)")
    if not np.all(np.isfinite(psi)):
        raise ValueError("|psi> contains non-finite values (NaN or Infinity)")
    # Exact symmetry / Hermiticity (not approximate)
    if np.iscomplexobj(H):
        if not np.array_equal(H, H.conj().T):
            raise ValueError("H is not exactly Hermitian (H != H†)")
        h_check = f"H is complex Hermitian ({n}×{n}): exactly H == H† verified"
    else:
        if not np.array_equal(H, H.T):
            raise ValueError("H is not exactly symmetric (H != H^T)")
        h_check = f"H is real and square ({n}×{n}): exactly symmetric (H == H^T)"
    if not np.any(psi != 0):
        raise ValueError("|psi> has zero norm")
    norm_sq = float(np.real(psi.conj() @ psi))
    psi_dtype = "complex" if np.iscomplexobj(psi) else "real"
    return [
        "H and |psi> are finite (no NaN or Infinity)",
        h_check,
        f"|psi> is a {psi_dtype} vector of length {n} with exact non-zero check",
        f"<psi|psi> = {norm_sq:.6g} > 0",
    ]
```

**Preconditions checked:**
- A1: H is a square 2-D array
- A2: psi has length matching H
- A3: All entries of H and psi are finite (NaN/Infinity rejected)
- A4: H is **exactly** Hermitian (`np.array_equal(H, H.conj().T)`)
- A5: psi is **exactly** non-zero (at least one non-zero component)

### Link 2a — Real interval arithmetic (_arb_rayleigh, complete code)

```python
def _arb_rayleigh(H, psi):
    """Returns (lower, upper, radius, backend_label) for real H, psi."""
    from flint import arb, arb_mat, ctx

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
                "Arb denominator ball contains zero; cannot certify."
            )
        numerator = (s_row * (H_mat * s_col))[0, 0]
        quotient  = numerator / denominator

        lower = math.nextafter(float(quotient.lower()), -math.inf)
        upper = math.nextafter(float(quotient.upper()),  math.inf)

        if not (math.isfinite(lower) and math.isfinite(upper)):
            raise ValueError(f"Arb endpoints not finite: lower={lower}, upper={upper}")
        radius = (upper - lower) / 2
        return lower, upper, radius, "flint-arb/prec=128"
    finally:
        ctx.prec = saved_prec
```

**Soundness claims:**
- SC-1: Arb at prec=128 propagates rounding outward through all matrix operations.
- SC-2: `quotient.lower()` / `quotient.upper()` export the infimum/supremum of the Arb ball.
- SC-3: `math.nextafter(x, -inf)` / `math.nextafter(x, +inf)` outward-round to the nearest representable binary64 outside the ball, guaranteeing the stored float64 [lower, upper] is a superset of the true mathematical Arb ball.
- SC-4: If `denominator.contains(0)` the function raises rather than dividing; division by an Arb ball that contains zero is undefined.

### Link 2b — Complex interval arithmetic (_acb_rayleigh, complete code)

```python
def _acb_rayleigh(H, psi):
    """Returns (lower, upper, radius, backend_label) for complex H and/or psi."""
    from flint import acb, acb_mat, ctx

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
            raise ValueError("Acb denominator real part ball contains zero; cannot certify.")
        num = (psi_dag * (H_acb * psi_col))[0, 0]
        q   = num / den

        if not q.imag.contains(0):
            raise ArithmeticError(
                "Acb result violates the exact-Hermitian invariant; "
                "check the call path, input finiteness, and backend"
            )

        lower = math.nextafter(float(q.real.lower()), -math.inf)
        upper = math.nextafter(float(q.real.upper()),  math.inf)

        if not (math.isfinite(lower) and math.isfinite(upper)):
            raise ValueError(f"Acb real endpoints not finite: lower={lower}, upper={upper}")
        radius    = (upper - lower) / 2
        return lower, upper, radius, "flint-acb/prec=128"
    finally:
        ctx.prec = saved_prec
```

**Soundness claims:**
- SC-5: For an exactly Hermitian H (guaranteed by A4), ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ is real.  The Acb imaginary-part ball must therefore contain zero; if it does not, the invariant has been violated by arithmetic error or input inconsistency.
- SC-6: The check `q.imag.contains(0)` is scale-independent (Arb ball containment, not a fixed epsilon), so it is valid for all input magnitudes.
- SC-7: Only `q.real` is used for the certified upper bound; the imaginary part is used solely as a sanity guard.

### Link 3 — Canonical digest (_canonical_digest, complete code)

```python
def _canonical_digest(H, psi):
    """Domain-separated, shape-tagged, big-endian SHA-256 digest."""
    def _field(tag: bytes, payload: bytes) -> bytes:
        return struct.pack(">H", len(tag)) + tag + struct.pack(">Q", len(payload)) + payload

    def _shape_bytes(array):
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
```

**Digest design:**
- D1: Domain separator `"rayleigh-cert-input/v2\x00"` prevents cross-context collisions.
- D2: Type flag `"C"` / `"R"` prevents real/complex aliasing.
- D3: Each field is length-prefixed (tag length as uint16, payload length as uint64), preventing extension/concatenation attacks within the preimage.
- D4: All integers serialised as big-endian (platform-independent).
- D5: Float64 serialised as IEEE 754 big-endian (`">f8"`), preserving the exact bit pattern including sign of zero.

### Link 4 — Certificate production (rayleigh_certificate, relevant excerpt)

```python
def rayleigh_certificate(H, psi, *, notes=""):
    # Fail closed if flint absent
    try:
        import flint
    except ImportError as exc:
        raise ImportError("rayleigh_certificate() requires python-flint ...") from exc

    # Enforce canonical dtype
    _CANONICAL_DTYPES = {np.dtype(np.float64), np.dtype(np.complex128)}
    if H.dtype not in _CANONICAL_DTYPES:
        raise TypeError(f"H dtype must be float64 or complex128; got {H.dtype}")
    if psi.dtype not in _CANONICAL_DTYPES:
        raise TypeError(f"psi dtype must be float64 or complex128; got {psi.dtype}")

    assumptions = _check_preconditions(H, psi)   # Link 1
    digest      = _canonical_digest(H, psi)       # Link 3
    lower, upper, _, backend = (
        _acb_rayleigh(H, psi) if is_complex else _arb_rayleigh(H, psi)  # Link 2
    )
    # ...
    return RayleighCertificate(
        claim      = f"E0 ≤ {upper:.17g}  [Rayleigh-Ritz upper bound on ground-state energy]",
        theorem    = EXPECTED_THEOREM,             # canonical string, not user input
        assumptions= assumptions,
        input_digest = digest,
        lower      = lower,
        upper      = upper,
        midpoint   = (lower + upper) / 2,
        radius     = math.nextafter(max(mid - lower, upper - mid), math.inf),
        backend    = backend,
        assurance  = "rigorous",
    )
```

### Link 5a — verify_rayleigh_certificate (relevant excerpt)

```python
def verify_rayleigh_certificate(cert):
    # Semantic field checks (must all pass before re-executing arithmetic)
    if cert.assurance != "rigorous":
        raise ValueError(f"assurance must be 'rigorous'; got {cert.assurance!r}")
    if cert.theorem != EXPECTED_THEOREM:
        raise ValueError(f"theorem has been tampered:\n  expected: {EXPECTED_THEOREM!r}\n  stored: {cert.theorem!r}")

    H   = _decode_canonical(cert.canonical["H"])
    psi = _decode_canonical(cert.canonical["psi"])

    # Re-derive digest
    recomputed_digest = _canonical_digest(H, psi)
    if recomputed_digest != cert.input_digest:
        raise ValueError("Digest mismatch — inputs have been modified")

    # Re-run preconditions
    _check_preconditions(H, psi)

    # Re-run interval arithmetic
    _, upper_v, _, recomputed_backend = (
        _acb_rayleigh(H, psi) if is_complex else _arb_rayleigh(H, psi)
    )

    # Backend cross-check
    if cert.backend != recomputed_backend:
        raise ValueError(f"Backend mismatch: stored={cert.backend!r}, recomputed={recomputed_backend!r}")

    # The core claim: recomputed upper must cover the stored upper
    if not (recomputed_upper <= cert.upper):
        raise ValueError(
            f"Verification failed: recomputed_upper={recomputed_upper} > stored upper={cert.upper}"
        )
    cert.verified = True
    return cert
```

### Link 5b — verify_from_dict (key excerpt for semantic checks)

```python
def verify_from_dict(full_cert):
    # Require flint (no trivial numpy-float "verification")
    try:
        import flint
    except ImportError as exc:
        raise ImportError("verify_from_dict() requires python-flint") from exc

    required = {"schema_version", "claim", "input_digest", "interval",
                "canonical", "assurance", "backend"}
    missing = required - set(full_cert)
    if missing:
        raise ValueError(f"Certificate dict missing required keys: {missing}")

    assurance = full_cert["assurance"]
    backend   = full_cert.get("backend", "")

    # Reject non-rigorous certificates
    if assurance != "rigorous" or "numpy" in backend.lower():
        return {"verified": False, ..., "message": f"FAIL — assurance={assurance!r} ..."}

    # Claim text must encode the stored upper value exactly
    expected_claim = f"E0 ≤ {stored_upper:.17g}  [Rayleigh-Ritz upper bound on ground-state energy]"
    if stored_claim != expected_claim:
        return {"verified": False, ..., "message": "FAIL — claim text does not match interval.upper"}

    # Theorem must be the canonical Rayleigh-Ritz statement
    if stored_theorem != EXPECTED_THEOREM:
        return {"verified": False, ..., "message": "FAIL — theorem text has been tampered"}

    # Digest, preconditions, interval arithmetic (same logic as Link 5a)
    ...
    # Core claim: recomputed_upper must be ≤ stored_upper
    if not (recomputed_upper <= stored_upper):
        return {"verified": False, ...}
    return {"verified": True, ...}
```

### Architecture note: shared arithmetic

Both the producer (`rayleigh_certificate`) and the independent verifier
(`verify_rayleigh_certificate`, `verify_from_dict`) share the same arithmetic
primitives in `_rayleigh_primitives.py`.  They do **not** share certificate
management logic, schema parsing, or intermediate state.  The shared module is
the auditable surface for the interval arithmetic.

This is a conscious design choice documented in `_rayleigh_primitives.py`:

> Sharing the same implementation of `_arb_rayleigh` means the verifier and
> producer run identical arithmetic code.  Full independence would require
> the verifier to use a separate arithmetic implementation (e.g. a different
> library or a hand-audited clean-room copy).  That is a P2 deliverable.

The verifier is therefore independent at the **certificate management** level
but not at the **arithmetic implementation** level.

---

## Numerical anchor points (sanity-only; computed by HTF)

All values below were produced by `rayleigh_certificate` + `verify_from_dict`
on the HTF host.  The reviewer should independently recompute these using
any available tool (python-flint, Julia Arb.jl, mpmath, etc.) and confirm
the bounds are satisfied.

### Anchor 1 — 3×3 diagonal, ground state

```
H = diag(0.0, 1.0, 2.0)   psi = [1, 0, 0]  (exact ground state)
Expected: upper ≈ 0.0 (+ small Arb rounding gap)
E0 = 0.0  →  E0 ≤ upper must hold
```

### Anchor 2 — 3×3 diagonal, non-trivial trial state

```
H = diag(0.0, 1.0, 2.0)   psi = [1, 1, 1] / sqrt(3)
Rayleigh quotient = (0 + 1 + 2) / 3 = 1.0
Expected: lower ≤ 1.0 ≤ upper (tight ball at prec=128)
E0 = 0.0  →  E0 = 0.0 ≤ upper must hold
```

### Anchor 3 — 2×2 complex Hermitian

```
H = [[1.0, 1j], [-1j, 1.0]]   psi = [1, 0]
Rayleigh quotient = 1.0 (real, exact)
Expected: upper ≈ 1.0 (+ small Arb rounding gap)
E0 = 0.0  →  E0 = 0.0 ≤ upper must hold
```

### Anchor 4 — near-degenerate

```
H = diag(0.0, 1e-15)   psi = [1, 1] / sqrt(2)
Rayleigh quotient = 5e-16
Expected: E0 = 0.0 ≤ upper holds; upper > 0
```

---

## Review questions

**Q1 — Precondition sufficiency.**
Are preconditions A1–A5 jointly sufficient for the Rayleigh-Ritz theorem to hold?
Specifically: does exact symmetry (`np.array_equal`) in float64 guarantee the
theorem applies, given that float64 entries are rational numbers (exact)?

**Q2 — Arb outward-rounding soundness.**
Is `math.nextafter(float(quotient.upper()), +inf)` a sound outward-round export
from the Arb ball to a binary64 upper bound?  In particular: does
`float(quotient.upper())` round the Arb upper endpoint inward or outward, and
does the subsequent `nextafter` always compensate?

**Q3 — Acb imaginary-part check.**
For exactly Hermitian H (A4), is it guaranteed that the Acb computation of
`⟨ψ|H|ψ⟩/⟨ψ|ψ⟩` at prec=128 yields an imaginary-part ball that contains zero?
Are there inputs where this check could spuriously fail (false positive) or
spuriously pass (false negative) despite H not being exactly Hermitian?

**Q4 — Canonical digest pre-image resistance.**
Is the digest scheme (domain-separated, length-prefixed, big-endian SHA-256)
sufficient to bind the certificate to the specific (H, psi) pair?  Are there
structural weaknesses (length extension, real/complex aliasing, dimension
collisions) that the current design does not close?

**Q5 — Semantic field tamper resistance.**
The verifier checks `assurance == "rigorous"`, `theorem == EXPECTED_THEOREM`,
and `backend == recomputed_backend`.  Are these checks sufficient to prevent
a certificate with a weaker arithmetic path from passing verification as
"rigorous"?  Is the backend string `"flint-arb/prec=128"` an adequate proxy
for the arithmetic quality?

**Q6 — Shared arithmetic and independence.**
Given that the producer and verifier share `_arb_rayleigh` / `_acb_rayleigh`,
what is the actual independence guarantee of the current verifier?  Is it
sufficient to detect: (a) a tampered `upper` field; (b) a tampered `H` or
`psi` in the canonical section; (c) a systematic bug in `_arb_rayleigh`?
What would a reviewer recommend as the minimal additional step to reach true
arithmetic independence?

**Q7 — Stated limitations.**
HTF explicitly states the following limitations.  Are they honest and complete?
- Certificate `upper` bounds only floating-point rounding error (not truncation error
  from a finite bond dimension χ approximation of an infinite system).
- The Rayleigh-Ritz bound is one-sided; no certified lower bound on E0 is claimed.
- Dense matrix representation: the certificate stores H and ψ as O(n²) and O(n) arrays.
- The `prec=128` precision is fixed; it is not adaptive to the condition number of H.

---

## Acceptance criteria

A returned review should contain:

1. **Overall verdict:** `PASS` / `CONDITIONAL` / `BLOCKED`
2. **Per-link assessment:** for each of Links 1–5, `CONFIRMED` / `PARTIAL` / `REFUTED`
3. **Per-question answer:** direct answer to Q1–Q7, with independent calculations where applicable
4. For `CONDITIONAL`: the exact required edit (file, line, text)
5. For `BLOCKED`: a minimal counterexample or gap description + minimal repair
6. **Numerical cross-check:** independent recomputation of at least Anchors 1 and 3 with
   a different tool (Julia Arb.jl, mpmath, SageMath, etc.) confirming the stated bounds hold

---

## What this review does NOT need to assess

- The Temple lower-bound heuristic (`htf/gap.py`) — documented as `[heuristic]`, not certified
- The OS-positivity diagnostics (`htf/os_axioms.py`) — documented as structural diagnostics,
  not true OS-positivity for infinite systems
- MPS/MPO adapter semantics — covered in HTF-02
- ZX-calculus rewrites — documented as `[research]` with independent correctness tests

---

## Version

Code excerpts taken from commit `02d0bb8` (2026-08-14).  All P0 blockers
(P0-1 through P0-5) from the strategic review v0.23.0 have been applied.
P0-6 (factorized MPS/MPO certificate, O(χ) storage) remains open as a
long-term research goal and is explicitly out of scope for this review.
