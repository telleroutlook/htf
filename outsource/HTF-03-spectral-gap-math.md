# HTF-03 — Spectral Gap Estimation: Mathematical Correctness Review

**Type:** Gate-A independent review (math correctness)

**What this is.** A request to independently verify the mathematical claims in
two functions used for spectral gap estimation: `first_excited_upper` and
`temple_lanczos`.  The reviewer needs nothing from the HTF repository beyond
what is quoted in this file.

**Non-circularity (mandatory).** Every claim in this file is either a quoted
definition or a mathematical proposition.  No step assumes the conclusion it
is asking to verify.

---

## Definitions

### Setting

Let H be a real symmetric n×n matrix with eigenvalues E_0 ≤ E_1 ≤ … ≤ E_{n-1}
and corresponding orthonormal eigenvectors v_0, v_1, …, v_{n-1}.

The **spectral gap** is Δ = E_1 − E_0.

### The Rayleigh-Ritz theorem (background — assumed correct, not under review)

For any non-zero vector φ:

    E_0 ≤ ⟨φ|H|φ⟩ / ⟨φ|φ⟩ ≤ E_{n-1}

If φ is orthogonal to ALL eigenvectors with eigenvalue < E_k, then:

    E_k ≤ ⟨φ|H|φ⟩ / ⟨φ|φ⟩

In particular, if φ ⊥ v_0 (exact ground-state eigenvector), then E_1 ≤ RQ(φ).

---

## Function 1 — `first_excited_upper`

### Code (complete, as deployed)

```python
def first_excited_upper(ham, state_gs, state_es):
    """Heuristic E_1 estimate from a state orthogonalised to an approximate |ψ_0⟩.

    WARNING: NOT a rigorous upper bound on E_1 unless state_gs is the
    EXACT ground-state eigenvector.  When state_gs is only approximate,
    the orthogonalised trial state is not in the true E_0 eigenspace's
    orthogonal complement, so the variational principle does not guarantee
    E_1 ≤ ⟨φ_⊥|H|φ_⊥⟩.

    Counter-example: H = diag(0, 1), state_gs = (1,1)/√2 (approximate),
    state_es = (0,1).  This function returns ≈ 0.5, but E_1 = 1.0.

    Use this only as a [heuristic] discovery-tier estimate.
    """
    psi0 = state_gs / norm(state_gs)
    phi_perp = state_es - (psi0 @ state_es) * psi0
    phi_perp /= norm(phi_perp)
    return energy_expectation(ham, phi_perp)   # = ⟨φ_⊥|H|φ_⊥⟩
```

### The counter-example (independently verifiable)

H = diag(0, 1),  exact eigenvectors: v_0 = (1,0), v_1 = (0,1).
Approximate ground state: ψ_0 = (1,1)/√2  (not an eigenvector).
Trial excited state: φ = (0,1).

Step 1 — orthogonalise φ against ψ_0:
    ⟨ψ_0|φ⟩ = 1/√2
    φ_⊥ = (0,1) − (1/√2) · (1,1)/√2 = (0,1) − (1/2)(1,1) = (−1/2, 1/2)
    normalised: φ_⊥ = (−1, 1)/√2

Step 2 — Rayleigh quotient:
    ⟨φ_⊥|H|φ_⊥⟩ = (1/2)(0 · 1 + 1 · 1) = 1/2 = 0.5

Step 3 — true E_1 = 1.0.

The function returns 0.5 < E_1 = 1.0.  The claim "E_1 ≤ result" is **false**
for this input.

### The current code's position

The function is labelled **[heuristic]** with the docstring warning shown above.
It is used internally to compute a "trial energy difference" (discovery-tier gap
estimate), not as a certified bound.

---

## Function 2 — Temple lower bound

### Mathematical background

The **Temple inequality** (1928) states: given H with eigenvalues E_0 < E_1 ≤ …,
any normalised trial vector φ with Rayleigh quotient λ = ⟨φ|H|φ⟩, let

    μ_1 = a known lower bound on E_1    (must satisfy μ_1 > λ)
    r²   = ⟨φ|H²|φ⟩ − λ²             (residual variance)

Then:

    E_0 ≥ λ − r² / (μ_1 − λ)          (Temple lower bound)

**Critical requirement:** μ_1 must be a **true lower bound** on E_1, i.e.
μ_1 ≤ E_1.  If μ_1 > E_1 then the formula can produce a value ABOVE E_0,
giving a false lower bound.

### Code (complete, as deployed)

```python
def temple_lanczos(H, k=30, seed=0):
    """Lanczos ground-state bounds: Ritz upper + Temple heuristic lower.

    Returns TwoSidedBounds with fields:
      E0_upper  : certified Rayleigh upper bound (rigorous)
      E0_lower  : Temple heuristic (see below — NOT rigorous in general)
      E1_ritz   : second Ritz value (used as μ_1 in Temple formula)
      temple_condition_met : True if E1_ritz > E0_upper (Temple requirement)
      notes     : description of assurance level
    """
    eigs, vecs = lanczos_eigs(H, k=k, seed=seed)
    E0_ritz = eigs[0]
    E1_ritz = eigs[1] if len(eigs) > 1 else float('inf')
    v0 = vecs[:, 0]

    # Rayleigh quotient (certified upper bound on E_0)
    E0_upper = rayleigh_certificate(H, v0).upper

    # Variance  r² = ⟨v0|H²|v0⟩ − E0_ritz²
    Hv0 = H @ v0
    r_sq = float(v0 @ (H @ Hv0)) - E0_ritz**2

    # Temple formula with μ_1 = E1_ritz  ← PROBLEM: E1_ritz is a Ritz
    # upper bound on E_1, not a lower bound.  The Temple formula requires
    # μ_1 ≤ E_1; using E1_ritz ≥ E_1 can make the denominator too large,
    # giving E0_lower < true E_0 (pessimistic, not a false lower bound).
    # However, if E1_ritz is only slightly above E_1, the result can still
    # be a useful estimate.
    temple_condition_met = (E1_ritz > E0_upper)
    if temple_condition_met and r_sq >= 0:
        E0_lower = E0_ritz - r_sq / (E1_ritz - E0_ritz)
    else:
        E0_lower = float('-inf')

    return TwoSidedBounds(
        E0_upper=E0_upper,
        E0_lower=E0_lower,
        E1_ritz=E1_ritz,
        temple_condition_met=temple_condition_met,
        k_lanczos=k,
        width=E0_upper - E0_lower,
        notes=(
            "E0_upper: rigorous Rayleigh-Ritz certified bound. "
            "E0_lower: Temple HEURISTIC — rigorous lower bound ONLY when "
            "E1_ritz is a true lower bound on E_1 (which it is not here; "
            "E1_ritz is a Ritz upper bound). "
            "temple_condition_met checks E1_ritz > E0_upper only."
        ),
    )
```

### The mathematical issue

The Temple formula requires μ_1 ≤ E_1 (a lower bound on E_1).
The code uses μ_1 = E1_ritz, which is a **Ritz upper bound** (≥ E_1, not ≤ E_1).

**Consequence analysis:**
- If E1_ritz ≥ E_1 (the typical case), the denominator (E1_ritz − E0_ritz) ≥
  (E_1 − E_0) > 0.  A larger denominator makes the Temple term smaller,
  so E0_lower = E0_ritz − r²/(E1_ritz − E0_ritz) may be **smaller** than
  the true Temple lower bound.
- This means E0_lower is **pessimistic** (further below E_0 than necessary),
  not falsely above E_0.
- BUT: near degeneracy (E_1 ≈ E_0), E1_ritz can be close to E0_ritz, making
  the denominator small and E0_lower potentially a very large negative number —
  not a false lower bound but an uninformative one.

The current code labels E0_lower as "heuristic" and the assurance is documented.

---

## The Gate-A questions

### Q1 — `first_excited_upper`: is the heuristic label sufficient?

The function is labelled [heuristic] with a docstring warning and counter-example.
It is used only to produce a "trial energy difference" (not a certified gap bound).

Is the current disposition correct?  Specifically:
- (a) Is there a simple, rigorous alternative — e.g. restricting to exact
  eigenvectors, or using a different orthogonalization — that would make the
  function produce a true E_1 upper bound?
- (b) If no simple fix exists, is the [heuristic] label sufficient to prevent
  misuse, or should the function be deprecated/removed from the public API?
- (c) Is the counter-example in the docstring mathematically correct?

### Q2 — Temple lower bound: is pessimism sufficient?

When E1_ritz ≥ E_1 (Ritz upper bound used as μ_1):
- (a) Is the resulting E0_lower always ≤ E_0 (a valid, if pessimistic, lower bound)?
  Or is there a case where it exceeds E_0?
- (b) Specifically: does E0_ritz − r²/(E1_ritz − E0_ritz) ≤ E_0 hold whenever
  E1_ritz ≥ E_1, r² ≥ 0, and temple_condition_met = True?
  (Hint: compare with the exact Temple formula using E_1 instead of E1_ritz.)
- (c) If the answer to (b) is NO — give the minimal counter-example.
  If YES — give the algebraic proof (one or two lines).

### Q3 — Combined gap estimate: is the trial difference safe to report?

The code reports `trial_energy_diff = E0_upper_var − E0_upper_var` (both
variational upper bounds), labelled as [heuristic] and NOT as a gap bound.

Is this adequate, or does the combination of two upper bounds create a
misleading signal (e.g. trial_energy_diff < 0 when gap > 0)?

### Q4 — Gate-A verdict

Given Q1–Q3: for each function, state PASS / CONDITIONAL / BLOCKED with the
minimal required change (if any).

---

## Acceptance criteria

1. **PASS:** all claims confirmed, [heuristic] labels are sufficient.
2. **CONDITIONAL:** labelling is insufficient; give the exact required docstring
   or code change.
3. **BLOCKED:** a mathematical error exists that the heuristic label does not
   cover; give the counter-example and the minimal repair (redesign or removal).
