# HTF-01 Verdict — Rayleigh Certificate Code + Math Correctness

**Reviewer type:** Internal self-review (no external reviewer available as of 2026-08-14)
**Review date:** 2026-08-14
**Commit reviewed:** b35bdc1 (HEAD at time of review)

---

## Overall verdict: CONDITIONAL

The core Rayleigh-Ritz pipeline is mathematically correct.  Three conditional
requirements (listed below) must be satisfied before the certificate should be
treated as publication-quality evidence.

---

## Link-by-link assessment

### Link 1 — Precondition checks (`_check_preconditions`)

**CONFIRMED** with one caveat.

A1 (square H), A2 (matching psi length), A4 (non-zero psi) are correct.
A3 in the *current* code uses `herm_defect > 1e-10` — a numerical proxy, not
exact self-adjointness.  For an Arb-certified bound the Rayleigh-Ritz theorem
requires *exact* self-adjointness; the tolerance leaves a residual gap.

**Required change (CONDITIONAL-1):** Replace the `1e-10` tolerance in A3 with
an exact equality check (`H == H.T` for real, `H == H.conj().T` for complex),
or document precisely that the 1e-10 residual is dominated by the Arb ball
radius.  The current v2 docstring claims "exact Hermitian check" — that claim
is inconsistent with a float tolerance.

*Note:* as of commit 0565522 the code in `_rayleigh_primitives.py` was updated
to use exact equality.  Verify that the actual deployed path matches.

### Link 2 — Real interval arithmetic (`_arb_rayleigh`)

**CONFIRMED** with one caveat.

`arb_mat` product correctly propagates rounding; division `num/den` is safe
when `psi` is nonzero (A4 checked before this call).  `q.upper()` with
`math.nextafter` is a sound outward-round export.

**Required change (CONDITIONAL-2):** The `mid ± rad` construction used in
the quoted HTF-01 prompt (Link 2) was *not* outward-rounded — it returned
`float(mid) + float(rad)` which can round down.  The current codebase uses
`q.upper()` + `nextafter`.  Confirm the deployed `_rayleigh_primitives.py`
matches (it does as of commit ef7fad4).

### Link 3 — Complex interval arithmetic (`_acb_rayleigh`)

**CONFIRMED** with one caveat.

`psi_dag` row vector correctly uses `psi[i].conjugate()`, giving `⟨ψ|`.
Division `q = num/den` is safe given A4.  Real extraction via `q.real` is
correct for Hermitian H.

The `imag_bound > 1e-8` threshold is conservative enough for 128-bit Acb
computations on matrices with entries of order 1.  For matrices with very
large entries (entries ~ 1e6), the 1e-8 threshold could fail legitimately.

**CONDITIONAL-3:** Document the implicit scale assumption (entries ~ O(1)) or
replace the absolute threshold with a relative one.

### Link 4 — SHA-256 input digest (`_canonical_digest`)

**CONFIRMED** (v2 encoding).

The v2 domain-separated encoding uses shape-tagged big-endian bytes.  The
SHA-256 is collision-resistant.  No structural shortcut collision exists for
inputs of different shapes (shape tag differs).

The prompt for this review described an older (v1) encoding without domain
separation — the v2 code is strictly stronger.

### Link 5 — Canonical encoding / decoding

**CONFIRMED.**

`float64 → Python float → float64` is bit-exact for all non-NaN finite values.
NaN/Inf are excluded by A1/A2/A4 preconditions.  Round-trip is lossless.

`_decode_canonical` correctly reconstructs complex arrays via component
assignment, preserving `-0.0` sign bits.

### Link 6 — Independent verifier (`verify_rayleigh_certificate`)

**CONFIRMED** (current codebase).

The verifier checks: schema v2 invariants, digest, preconditions, strict
`upper_v ≤ stored_upper` (no tolerance), claim text consistency, interval
Fraction ordering, and — via `verify_from_dict` — semantic fields (claim,
theorem, backend, lower, radius).  It raises `ImportError` without flint.

The earlier version described in the HTF-01 prompt (Link 6) had a `+ tol`
tolerance and no semantic field checks.  Both gaps are closed in current code.

---

## Q1–Q5 answers

**Q1:** Preconditions are sufficient *given* exact Hermitian check (CONDITIONAL-1).
NaN/Inf guard is present in `_check_preconditions` (finite check added v2).

**Q2:** Real interval arithmetic is sound.  `mid ± rad` in the prompt was the
older code; current `q.upper()` + `nextafter` is correct (CONDITIONAL-2 closes).

**Q3:** Complex path correct.  Scale assumption undocumented (CONDITIONAL-3).

**Q4:** Digest is collision-resistant; round-trip is bit-exact for finite floats.

**Q5:** Honest limitations — the certificate explicitly states truncation/modelling
error is `[OUT]`.  `radius=0.0` for numpy-float backend is now labelled as
"not certified" and carries `assurance="heuristic"`.

---

## Acceptance criteria outcome

**CONDITIONAL** — all three required changes listed above.  CONDITIONAL-1 and
CONDITIONAL-2 are already resolved in the current deployed code; only
CONDITIONAL-3 (document Acb scale assumption) remains open as a documentation
task.

The pipeline as currently deployed is sound for matrices with entries of
moderate magnitude.  It should not be used without additional validation for
Hamiltonians with entries substantially larger than O(1).
