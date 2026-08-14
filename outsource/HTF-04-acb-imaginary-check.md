# HTF-04 — Acb Interval Arithmetic: Imaginary-Part Containment Review

**Type:** Gate-A independent review (interval arithmetic correctness)

**What this is.** A request to verify whether the Acb (complex ball arithmetic)
check used in the HTF Rayleigh certificate pipeline correctly certifies that the
Rayleigh quotient is real for a complex Hermitian Hamiltonian.

The reviewer needs nothing from the HTF repository beyond what is quoted in this
file.  python-flint / FLINT documentation is freely available at
https://python-flint.readthedocs.io and https://flintlib.org.

---

## Context

For a complex Hermitian H (H = H†) and any non-zero |ψ⟩, the Rayleigh quotient

    RQ = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩

is mathematically **real** (the imaginary part is identically zero).
In Acb ball arithmetic, the computed quotient `q` has a non-zero imaginary ball
due to rounding.  The code checks `q.imag.contains(0)` to certify that the
imaginary part is consistent with zero.

If this check fails the function raises `ValueError`; if it passes, only the
real part `q.real` is exported as the certified interval.

---

## Complete code under review

```python
def _acb_rayleigh(H, psi):
    """Compute ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ using Acb ball arithmetic.

    Returns (lower, upper, radius, backend_label) for the REAL part.
    Raises ValueError if the imaginary part ball does not contain zero.
    """
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
            raise ValueError("denominator ball contains zero; cannot certify")
        num = (psi_dag * (H_acb * psi_col))[0, 0]
        q   = num / den

        # Key check: imaginary part ball must contain zero.
        if not q.imag.contains(0):
            raise ValueError(
                "Rayleigh quotient imaginary part ball does not contain zero — "
                "check that H is exactly Hermitian"
            )

        lower = math.nextafter(float(q.real.lower()), -math.inf)
        upper = math.nextafter(float(q.real.upper()),  math.inf)
        radius = (upper - lower) / 2
        imag_rad = float(q.imag.rad())
        return lower, upper, radius, f"flint-acb/prec=128 (im_ball_rad={imag_rad:.2e})"
    finally:
        ctx.prec = saved_prec
```

---

## The four questions under review

### Q1 — Is `q.imag.contains(0)` a *sound* check?

**Formal claim:** if `q.imag.contains(0)` returns True, then the true
mathematical value of `Re(⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩)` is contained in the Acb real
ball `q.real`.

In particular: the imaginary part of the true Rayleigh quotient is exactly zero
for Hermitian H.  Does `q.imag.contains(0)` — as defined by FLINT's Acb ball
arithmetic — guarantee that the imaginary rounding error is accounted for in
`q.real`, so that `q.real` contains the true real Rayleigh quotient?

Or is it possible that `q.imag.contains(0)` passes but `q.real` does NOT
contain the true value (because the imaginary rounding spilled into the real
part without being tracked)?

### Q2 — Is `q.imag.contains(0)` *complete*?

**Formal claim:** if H is NOT exactly Hermitian (H ≠ H†), can `q.imag.contains(0)`
still return True, causing the function to silently certify a wrong bound?

Specifically: consider H = A + iε·B where A = Aᵀ (real symmetric) and B is
a real antisymmetric matrix with ||B||_∞ = δ.  For what values of δ (relative
to n and the magnitude of A's entries) does `q.imag.contains(0)` still pass at
prec=128?

If δ > 0 but small, can the function produce a certified interval that does NOT
contain the true Rayleigh quotient of H?

### Q3 — Scale dependence

The `q.imag.contains(0)` check is **scale-dependent**: for matrices with very
large entries (e.g. all H[i,j] ~ 10^{12}), the Acb ball radius grows with the
magnitude of the entries, so a small imaginary rounding error (|imag| ~ 10^{-4})
would be contained in the ball and the check would pass — but the imaginary part
is NOT zero.

At prec=128, what is the maximum entry magnitude for which the check reliably
detects non-Hermitian H with ||H − H†||_∞ = 1 ULP?

### Q4 — Gate-A verdict

Given Q1–Q3: does `q.imag.contains(0)` correctly certify that the exported
real interval `[lower, upper]` contains the true Rayleigh quotient, for:
(a) exactly Hermitian H (entries of moderate magnitude, ~ O(1))?
(b) approximately Hermitian H (H = A + ε·B, small ε)?
(c) large-entry H (entries ~ 10^{k} for k = 6, 9, 12)?

Verdict: PASS / CONDITIONAL (with exact required change) / BLOCKED (with
counter-example and minimal repair).

---

## Acceptance criteria

1. **PASS:** Q1 confirmed sound; Q2 shows the check is complete for any
   exactly Hermitian H; Q3 scale limit documented or shown not to matter
   for the precondition check (H is already verified to be exactly Hermitian
   before `_acb_rayleigh` is called).
2. **CONDITIONAL:** the check is sound but an additional guard (e.g. relative
   threshold, explicit Acb imaginary-part bound) is required.  Give the exact
   code change.
3. **BLOCKED:** the check is mathematically unsound (does not guarantee the
   real interval contains the true value).  Give the counter-example and
   minimal repair.

---

## Key note for the reviewer

Before `_acb_rayleigh` is called, `_check_preconditions` verifies **exact**
Hermiticity:

```python
if not np.array_equal(H, H.conj().T):
    raise ValueError("H is not exactly Hermitian")
```

This means H satisfies H[i,j] == H[j,i].conjugate() **exactly** in IEEE 754
float64.  The imaginary rounding in `q.imag` therefore comes entirely from Acb
ball arithmetic propagation, not from a non-Hermitian H.

The Q3 scale concern is relevant only if `_check_preconditions` might pass for
a matrix that is not "morally" Hermitian (e.g. due to float64 rounding in the
construction of H before the check).  This is a user-error scenario, but it may
be worth documenting.
