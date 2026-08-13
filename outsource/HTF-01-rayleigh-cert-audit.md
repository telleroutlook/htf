# HTF-01 — Rayleigh Certificate: Code + Math Correctness Audit

**Type:** Gate-A independent review (whole-pipeline inspection)

**What this is.** A request to independently verify that `htf/rayleigh_cert.py` correctly
implements the Rayleigh-Ritz upper bound with certified interval arithmetic, and that the
resulting `RayleighCertificate` schema is a sound, tamper-detectable evidence record.
The reviewer needs nothing from the HTF repository beyond what is quoted in this file.

**Non-circularity (mandatory).** The claim under review is `E0 ≤ upper` for an explicitly
given (H, ψ). No step should assume RH, eigenvalue location results, or anything beyond
standard linear algebra and the Rayleigh-Ritz theorem.

---

## All definitions (self-contained)

### The Rayleigh-Ritz theorem (what is invoked)

**Theorem (Rayleigh-Ritz).** Let H be a self-adjoint operator on a finite-dimensional
complex Hilbert space with smallest eigenvalue E0.  For any non-zero vector |ψ⟩:

```
E0 ≤ Re(⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩)
```

The right-hand side is the Rayleigh quotient, a real number (for Hermitian H) satisfying
E0 ≤ RQ ≤ E_{n-1}.

*Reference:* Courant & Hilbert, *Methods of Mathematical Physics*, Vol. I, §VI.1.

### The certificate schema (`rayleigh-cert/v1`)

A `RayleighCertificate` is a dict with the following fields:

```
schema_version : "rayleigh-cert/v1"
claim          : "E0 ≤ {upper:.17g}  [Rayleigh-Ritz upper bound on ground-state energy]"
theorem        : "Rayleigh-Ritz: for any non-zero |ψ⟩ and self-adjoint H, ..."
assumptions    : list of strings (machine-checked preconditions)
input_digest   : SHA-256 hex of canonical (H, ψ) input bytes
interval       : {lower, upper, midpoint, radius}   (all float)
backend        : "flint-arb" | "flint-acb" | "numpy-float (...)"
htf_version    : string
verified       : bool
notes          : string
```

The **certified claim** is `E0 ≤ interval.upper`.  The bound is trusted only when
`backend` starts with `"flint-arb"` or `"flint-acb"`.

### The full pipeline (the six links under review)

**Link 1 — Precondition checks** (`_check_preconditions`):

```python
def _check_preconditions(H, psi):
    # A1: H is square
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(...)
    n = H.shape[0]
    # A2: psi has length n
    if psi.ndim != 1 or len(psi) != n:
        raise ValueError(...)
    # A3: H is real symmetric or complex Hermitian
    if np.iscomplexobj(H):
        herm_defect = float(np.abs(H - H.conj().T).max())
        if herm_defect > 1e-10: raise ValueError(...)
    else:
        sym_defect = float(np.abs(H - H.T).max())
        if sym_defect > 1e-10: raise ValueError(...)
    # A4: psi is non-zero
    norm_sq = float(np.real(psi.conj() @ psi))
    if norm_sq < 1e-30: raise ValueError("zero norm")
    return [list of passed check strings]
```

**Link 2 — Real interval arithmetic** (`_arb_rayleigh`, when H and ψ are real):

```python
# python-flint Arb balls
s_row = arb_mat([[arb(float(psi[i])) for i in range(n)]])
s_col = arb_mat([[arb(float(psi[i]))] for i in range(n)])
H_mat = arb_mat([[arb(float(H[i, j])) for j in range(n)] for i in range(n)])

numerator   = (s_row * (H_mat * s_col))[0, 0]   # ⟨ψ|H|ψ⟩ as Arb ball
denominator = (s_row * s_col)[0, 0]              # ⟨ψ|ψ⟩ as Arb ball
quotient    = numerator / denominator             # Arb ball for RQ

mid = float(quotient.mid())
rad = float(quotient.rad())
return mid - rad, mid + rad, rad, "flint-arb"
```

**Link 3 — Complex interval arithmetic** (`_acb_rayleigh`, when H or ψ is complex):

```python
# python-flint Acb balls (complex)
psi_dag = acb_mat([[acb(complex(psi[i].conjugate())) for i in range(n)]])
psi_col = acb_mat([[acb(complex(psi[i]))]              for i in range(n)])
H_acb   = acb_mat([[acb(complex(H[i, j])) for j in range(n)] for i in range(n)])

num = (psi_dag * (H_acb * psi_col))[0, 0]   # ⟨ψ|H|ψ⟩ as Acb ball
den = (psi_dag * psi_col)[0, 0]             # ⟨ψ|ψ⟩ as Acb ball
q   = num / den

mid_re = float(q.real.mid())
rad_re = float(q.real.rad())
mid_im = float(q.imag.mid())
rad_im = float(q.imag.rad())
imag_bound = abs(mid_im) + rad_im
if imag_bound > 1e-8:
    raise ValueError("imaginary part too large")
return mid_re - rad_re, mid_re + rad_re, rad_re, "flint-acb (...)"
```

**Link 4 — SHA-256 input digest** (`_canonical_digest`):

```python
# Real path:  SHA-256(H_f64_bytes || psi_f64_bytes)
# Complex path: SHA-256(H.real_f64 || H.imag_f64 || psi.real_f64 || psi.imag_f64)
if np.iscomplexobj(H) or np.iscomplexobj(psi):
    raw = (H.real.astype(np.float64).tobytes()
         + H.imag.astype(np.float64).tobytes()
         + psi.real.astype(np.float64).tobytes()
         + psi.imag.astype(np.float64).tobytes())
else:
    raw = H.astype(np.float64).tobytes() + psi.astype(np.float64).tobytes()
return hashlib.sha256(raw).hexdigest()
```

**Link 5 — Canonical encoding / decoding** (for replay):

```python
# Encode (stored in _H_canonical, _psi_canonical):
def _encode_canonical(arr):
    if np.iscomplexobj(arr):
        return {"real": arr.real.tolist(), "imag": arr.imag.tolist()}
    return arr.tolist()

# Decode (used in verify_rayleigh_certificate and htf-verify):
def _decode_canonical(data):
    if isinstance(data, dict):
        return np.array(data["real"]) + 1j * np.array(data["imag"])
    return np.array(data, dtype=float)
```

**Link 6 — Independent verifier** (`verify_rayleigh_certificate`):

```python
def verify_rayleigh_certificate(cert):
    cert.validate()                              # 1. schema check
    H   = _decode_canonical(cert._H_canonical)
    psi = _decode_canonical(cert._psi_canonical)
    recomputed_digest = _canonical_digest(H, psi)
    if recomputed_digest != cert.input_digest:   # 2. digest check
        raise ValueError("Input digest mismatch")
    _check_preconditions(H, psi)                 # 3. preconditions
    is_complex = np.iscomplexobj(H) or np.iscomplexobj(psi)
    if is_complex:
        lower_v, upper_v, _, _ = _acb_rayleigh(H, psi)
    else:
        lower_v, upper_v, _, _ = _arb_rayleigh(H, psi)
    tol = max(abs(cert.upper) * 1e-14, 1e-15)
    if upper_v > cert.upper + tol:               # 4. recomputed ≤ stored
        raise ValueError("Recomputed upper exceeds stored")
    cert.verified = True
    return cert
```

---

## Numerical anchor (independently verifiable — sanity only)

For H = diag(0, 1), ψ = [1, 0] (exact ground state):

```
⟨ψ|H|ψ⟩ = 0.0,  ⟨ψ|ψ⟩ = 1.0,  RQ = 0.0
input_digest = SHA-256(
    bytes([0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,   # H[0,0]=0, H[0,1]=0
           0,0,0,0,0,0,0,0, 63,240,0,0,0,0,0,0, # H[1,0]=0, H[1,1]=1.0
           63,240,0,0,0,0,0,0,                  # psi[0]=1.0
           0,0,0,0,0,0,0,0])                    # psi[1]=0.0
)
```

(IEEE 754: `1.0 = 0x3FF0000000000000`, `0.0 = 0x0000000000000000`, big-endian.)
The expected `interval.upper ≤ 1e-9` (Arb radius ≈ 0, since inputs are exactly representable).

---

## The Gate-A questions (the actual deliverable)

### Q1 — Precondition completeness
Are the four checks A1–A4 **sufficient** to guarantee the Rayleigh-Ritz theorem applies?
Specifically:
- Is checking `max|H − H†| ≤ 1e-10` instead of exact symmetry a valid weakening?
  (The theorem requires self-adjointness; numerical symmetry is a proxy.)
- Is there any case where `⟨ψ|ψ⟩ ≥ 1e-30` but the theorem fails or the quotient is
  numerically unreliable?
- Are there any missing checks (e.g. finiteness of H entries, NaN/Inf guards)?

### Q2 — Real interval arithmetic correctness (Link 2)
Does the Arb computation correctly produce a **certified** interval containing the true
Rayleigh quotient?
- Does `arb_mat([[arb(float(x))]...]) * arb_mat(...)` correctly propagate rounding errors
  in the matrix-vector product `H_mat * s_col`?
- Is the division `numerator / denominator` safe when `denominator` could contain zero in
  its ball?  (When ψ has very small norm, `denominator.rad()` might be larger than
  `denominator.mid()`.)
- Does the `mid ± rad` construction give a **true superset** of the mathematical quotient?
  (i.e. is `rad` truly a radius bound, not just a midpoint error estimate?)

### Q3 — Complex interval arithmetic correctness (Link 3)
For complex Hermitian H:
- Does `Re(⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩)` equal the Rayleigh quotient?  (For Hermitian H, the
  imaginary part is exactly zero; but the Acb ball has non-zero `imag_bound` due to
  rounding.  Is the threshold `1e-8` appropriate, or could it accept a non-Hermitian H?)
- Is `psi_dag` constructed correctly as the conjugate transpose?  The code uses
  `acb(complex(psi[i].conjugate()))` for the row vector — confirm this is `⟨ψ|` not `|ψ⟩`.
- Is the `Acb` division `q = num / den` safe when `den` is nearly zero?

### Q4 — Digest security and round-trip soundness (Links 4–5)
- Is the canonical encoding `arr.real.tolist()` → `np.array(list)` a **bit-exact
  round-trip** for IEEE 754 float64 values?  (Python `float` is C `double`; confirm that
  `numpy.float64 → Python float → numpy.float64` preserves all 64 bits.)
- Can two different (H, ψ) pairs produce the same SHA-256 digest (collision)?  The digest
  is SHA-256 of 8n² + 8n bytes (real case); collisions are computationally infeasible but
  confirm there is no structural shortcut (e.g. H and ψ bytes are concatenated in a fixed,
  unambiguous order with no length prefix — is prefix-free encoding needed?).
- Does `_decode_canonical` produce the **same** array that `_encode_canonical` started from,
  up to the float64 round-trip?  If not, verify_rayleigh_certificate will report a digest
  mismatch even for untampered certificates.

### Q5 — Honest limitations
- The certificate only bounds `E0 ≤ upper`.  It does NOT bound the gap `upper − E0` or
  the truncation error in ψ (if ψ came from an MPS with bond dimension χ).  Are these
  limitations clearly stated in the schema?
- `radius = 0.0` when `backend` contains `"numpy-float"` (flint absent).  The certificate
  is then NOT certified — it is discovery-tier only.  Is this clearly communicated to a
  reader who only looks at the JSON?
- Is there any claim in the certificate (e.g. in the `claim` or `theorem` fields) that goes
  beyond what the Rayleigh-Ritz theorem guarantees?

### Q6 — Gate-A verdict
Given Q1–Q5 and the six links: does the `RayleighCertificate` pipeline, as described,
constitute a **correct, sound, tamper-detectable** evidence record for `E0 ≤ upper`?
Verdict: PASS / CONDITIONAL (with exact required change) / BLOCKED (with exact gap).

---

## Acceptance criteria

1. **PASS:** all six links confirmed, Q1–Q5 answered with no blocking gap.
2. **CONDITIONAL:** links are correct but a specific code or schema change is required
   before the certificate should be trusted.  Give the exact required edit.
3. **BLOCKED:** a genuine mathematical or implementation gap exists.
   Identify the link/question and give the minimal repair.

An honest "CONDITIONAL — add NaN/Inf guard to precondition checks" is a valuable outcome.
