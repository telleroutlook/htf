# HTF-05 — Rayleigh Certificate Pipeline: External Independent Review

**Type:** Gate-A external independent review (full pipeline re-audit)

**What this is.** HTF-01 was an internal self-review.  This file requests an
**external** independent review of the same pipeline by a domain expert who was
not involved in writing the code.  The code excerpts are self-contained; the
reviewer needs nothing from the HTF repository.

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
`backend` is `"flint-arb/prec=128"` (real H) or `"flint-acb/prec=128"` (complex H).

---

## All definitions and code (self-contained)

### The Rayleigh-Ritz theorem (invoked)

Let H be a self-adjoint n×n matrix.  For any non-zero |ψ⟩:

    E0 ≤ Re(⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩)

*Reference:* Courant & Hilbert, *Methods of Mathematical Physics*, Vol. I, §VI.1.

### Precondition checks (`_check_preconditions`) — complete code

```python
def _check_preconditions(H, psi):
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be square; got shape {H.shape}")
    n = H.shape[0]
    if psi.ndim != 1 or len(psi) != n:
        raise ValueError(f"|ψ⟩ must be 1-D of length {n}; got {psi.shape}")

    if not np.all(np.isfinite(H)):
        raise ValueError("H contains non-finite values")
    if not np.all(np.isfinite(psi)):
        raise ValueError("|ψ⟩ contains non-finite values")

    if np.iscomplexobj(H):
        if not np.array_equal(H, H.conj().T):          # EXACT check
            raise ValueError("H is not exactly Hermitian")
        h_check = f"H is complex Hermitian ({n}×{n}): exactly H == H†"
    else:
        if not np.array_equal(H, H.T):                 # EXACT check
            raise ValueError("H is not exactly symmetric")
        h_check = f"H is real and square ({n}×{n}): exactly H == Hᵀ"

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
```

### Real Arb backend (`_arb_rayleigh`) — complete code

```python
def _arb_rayleigh(H, psi):
    """Returns (lower, upper, radius, backend_label). Requires python-flint."""
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
            raise ValueError("denominator ball contains zero")
        numerator = (s_row * (H_mat * s_col))[0, 0]
        quotient  = numerator / denominator

        lower = math.nextafter(float(quotient.lower()), -math.inf)
        upper = math.nextafter(float(quotient.upper()),  math.inf)

        if not (math.isfinite(lower) and math.isfinite(upper)):
            raise ValueError(f"non-finite endpoints: lower={lower}, upper={upper}")
        radius = (upper - lower) / 2
        return lower, upper, radius, "flint-arb/prec=128"
    finally:
        ctx.prec = saved_prec
```

### Complex Acb backend (`_acb_rayleigh`) — complete code

```python
def _acb_rayleigh(H, psi):
    """Returns (lower, upper, radius, backend_label). Requires python-flint."""
    from flint import acb, acb_mat, ctx

    n = len(psi)
    H_c, psi_c = H.astype(complex), psi.astype(complex)
    saved_prec = ctx.prec
    try:
        ctx.prec = 128

        psi_dag = acb_mat([[acb(complex(psi_c[i].conjugate())) for i in range(n)]])
        psi_col = acb_mat([[acb(complex(psi_c[i]))]              for i in range(n)])
        H_acb   = acb_mat([[acb(complex(H_c[i, j])) for j in range(n)] for i in range(n)])

        den = (psi_dag * psi_col)[0, 0]
        if den.real.contains(0):
            raise ValueError("denominator ball contains zero")
        num = (psi_dag * (H_acb * psi_col))[0, 0]
        q   = num / den

        if not q.imag.contains(0):
            raise ValueError("imaginary part ball does not contain zero")

        lower = math.nextafter(float(q.real.lower()), -math.inf)
        upper = math.nextafter(float(q.real.upper()),  math.inf)
        radius = (upper - lower) / 2
        imag_rad = float(q.imag.rad())
        return lower, upper, radius, f"flint-acb/prec=128 (im_ball_rad={imag_rad:.2e})"
    finally:
        ctx.prec = saved_prec
```

### Canonical digest (`_canonical_digest`) — complete code

```python
def _canonical_digest(H, psi):
    """Domain-separated, shape-tagged SHA-256 digest (v2)."""
    def _field(tag, payload):
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

### Canonical encoding / decoding

```python
def _encode_canonical(arr):
    if np.iscomplexobj(arr):
        return {"real": arr.real.tolist(), "imag": arr.imag.tolist()}
    return arr.tolist()

def _decode_canonical(data):
    if isinstance(data, dict):
        real = np.asarray(data["real"], dtype=np.float64)
        imag = np.asarray(data["imag"], dtype=np.float64)
        result = np.empty(real.shape, dtype=np.complex128)
        result.real = real
        result.imag = imag
        return result
    return np.asarray(data, dtype=np.float64)
```

### Independent verifier (`verify_from_dict`) — complete code

```python
def verify_from_dict(full_cert):
    """Re-derive and confirm a RayleighCertificate from its full serialisation.
    Raises ImportError without python-flint.
    """
    import flint  # raises ImportError if absent

    # Reject non-rigorous certificates
    backend   = full_cert.get("backend", "")
    assurance = full_cert.get("assurance", "rigorous")
    if assurance != "rigorous" or "numpy" in backend.lower():
        return {"verified": False, "message": f"FAIL — not rigorous: assurance={assurance!r}"}

    stored_upper = float(full_cert["interval"]["upper"])

    # Claim text must encode the stored upper value
    expected_claim = (
        f"E0 ≤ {stored_upper:.17g}  [Rayleigh-Ritz upper bound on ground-state energy]"
    )
    if full_cert.get("claim", "") != expected_claim:
        return {"verified": False, "message": "FAIL — claim text mismatch"}

    # Theorem must match canonical statement
    EXPECTED_THEOREM = (
        "Rayleigh-Ritz: for any non-zero |ψ⟩ and self-adjoint H, "
        "E0 ≤ Re(⟨ψ|H|ψ⟩/⟨ψ|ψ⟩)."
    )
    if full_cert.get("theorem", "") != EXPECTED_THEOREM:
        return {"verified": False, "message": "FAIL — theorem tampered"}

    canonical = full_cert["canonical"]
    H   = _decode_canonical(canonical["H"])
    psi = _decode_canonical(canonical["psi"])

    # Digest check
    recomputed_digest = _canonical_digest(H, psi)
    if recomputed_digest != full_cert["input_digest"]:
        return {"verified": False, "message": "FAIL — digest mismatch", "digest_match": False}

    # Precondition checks
    _check_preconditions(H, psi)

    # Recompute interval
    is_complex = np.iscomplexobj(H) or np.iscomplexobj(psi)
    if is_complex:
        recomputed_lower, recomputed_upper, _, recomputed_backend = _acb_rayleigh(H, psi)
    else:
        recomputed_lower, recomputed_upper, _, recomputed_backend = _arb_rayleigh(H, psi)

    if recomputed_upper > stored_upper:
        return {"verified": False, "message":
            f"FAIL — recomputed upper {recomputed_upper:.17g} > stored {stored_upper:.17g}"}

    # Semantic field checks
    if full_cert.get("backend", "") != recomputed_backend:
        return {"verified": False, "message": "FAIL — backend field tampered"}

    stored_lower = float(full_cert["interval"]["lower"])
    _rtol = 1e-12
    lower_tol = _rtol * max(1.0, abs(recomputed_lower))
    if abs(stored_lower - recomputed_lower) > lower_tol:
        return {"verified": False, "message": "FAIL — interval.lower tampered"}

    # Numpy cross-check (independent of Arb)
    numpy_rq = float(np.real(psi.conj() @ H @ psi)) / float(np.real(psi.conj() @ psi))
    _interval_tol = max(1e-10, (recomputed_upper - recomputed_lower) * 10)
    if not (recomputed_lower - _interval_tol <= numpy_rq <= recomputed_upper + _interval_tol):
        return {"verified": False, "message":
            f"FAIL — numpy cross-check: RQ={numpy_rq:.17g} outside Arb interval"}

    return {
        "verified": True,
        "message": f"PASS — E0 ≤ {stored_upper:.17g} confirmed",
        "digest_match": True,
        "backend": recomputed_backend,
    }
```

---

## Numerical anchor (independently verifiable)

H = diag(0, 1),  ψ = [1, 0]:

    RQ = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩ = 0.0

Expected: `interval.upper ≤ 1e-15` (Arb radius ≈ 0 for exactly representable inputs).

H = diag(−5.0, 1.0),  ψ = [1, 0]:

    RQ = −5.0

Expected: `interval.upper ≤ −5.0 + ε` for machine-epsilon ε.

---

## The Gate-A questions

### Q1 — Precondition completeness

Are the checks in `_check_preconditions` (finite, exact symmetric/Hermitian,
exact non-zero) sufficient to guarantee Rayleigh-Ritz applies?

Specifically: is there any input that passes all checks but for which
`_arb_rayleigh` or `_acb_rayleigh` could produce an interval NOT containing
the true Rayleigh quotient?

### Q2 — Real Arb soundness

For the real path (`_arb_rayleigh`):
(a) Does `arb_mat([[arb(float(x))]...])` correctly wrap each float64 entry as
    an Arb ball, including rounding from float64 → Arb representation?
(b) Is the matrix-vector product `H_mat * s_col` and the inner products
    `s_row * (H_mat * s_col)` and `s_row * s_col` computed by Arb in a
    way that rigorously encloses the exact arithmetic result?
(c) Does `float(quotient.lower())` + `math.nextafter(..., -inf)` guarantee
    that the exported `lower` is a true lower bound (≤ true RQ)?
    Does the same hold for `upper`?

### Q3 — Complex Acb soundness

For the complex path (`_acb_rayleigh`):
(a) Is `q.imag.contains(0)` the correct criterion to verify that the Rayleigh
    quotient is real (given exactly Hermitian H, verified by `_check_preconditions`)?
(b) If `q.imag.contains(0)` is True, does the real interval `[lower, upper]`
    (from `q.real.lower()` / `q.real.upper()`) contain the true real Rayleigh
    quotient?  In other words: does passing the imaginary-part check guarantee
    the real-part interval is a valid enclosure?

### Q4 — Digest security

Is the v2 domain-separated encoding (`_canonical_digest`) collision-resistant
in practice?  Specifically:
(a) Can two distinct (H₁, ψ₁) ≠ (H₂, ψ₂) with the same shape produce the
    same digest?  (SHA-256 collisions aside — consider only structural collisions
    from the encoding.)
(b) Is the big-endian float64 encoding (`dtype=">f8"`) a bijection on IEEE 754
    float64 values (including ±0, ±Inf, NaN)?  Note: NaN is excluded by
    `_check_preconditions`.

### Q5 — Verifier non-circularity

The verifier (`verify_from_dict`) imports from `_rayleigh_primitives` — the same
module used by the producer.  This means both run the same Arb code.

(a) Is this architecturally sufficient for a meaningful independent check,
    given that the verifier at least (i) re-runs from canonical inputs, (ii)
    checks the digest, (iii) checks semantic fields (claim text, theorem, backend,
    lower), and (iv) runs a numpy cross-check?
(b) What is the minimal gap left by this shared-implementation architecture?
    What would a truly independent verifier require (e.g. a different library,
    a different language, a hand-checked proof)?

### Q6 — Gate-A verdict

Given Q1–Q5: does the `rayleigh-cert/v2` pipeline constitute a sound,
tamper-detectable evidence record for `E0 ≤ upper`?

Verdict: PASS / CONDITIONAL (exact required change) / BLOCKED (exact gap + repair).

---

## Acceptance criteria

1. **PASS:** all six questions confirmed; pipeline is sound for its stated scope
   (finite matrices, entries of moderate magnitude, float64 inputs).
2. **CONDITIONAL:** one or more questions identify a gap requiring a specific
   code or documentation change.  Give the exact edit.
3. **BLOCKED:** a mathematical or implementation gap exists.  Give the
   counter-example and minimal repair.

An honest CONDITIONAL verdict is a valuable outcome.
