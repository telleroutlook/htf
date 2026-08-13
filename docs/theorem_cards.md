# HTF Theorem Cards

One card per claim type used in HTF.  Each card states the theorem invoked,
the assumptions that must hold, the known failure modes, and the verification
algorithm that the independent verifier runs.

Evidence grammar used throughout:
- `[engineering]` — buildable with known tools
- `[research]` — genuine open research
- `[heuristic]` — interpretive analogy, not an established method
- `[OUT]` — explicitly not claimed by HTF

---

## TC-1 · Rayleigh-Ritz Upper Bound

**Claim form:**  `E0 ≤ upper`  where `upper = Re(⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩)`.

**Status:** `[engineering]` — standard linear-algebra result; no open questions.

### Theorem

**Rayleigh-Ritz.**  Let H be a self-adjoint operator on a finite-dimensional
Hilbert space, with ground-state energy E0 (smallest eigenvalue).  For any
non-zero vector |ψ⟩:

```
E0 ≤ Re(⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩)
```

*Reference:* Courant & Hilbert, *Methods of Mathematical Physics*, Vol. 1, §VI.

### Assumptions (machine-checked by `rayleigh_certificate`)

| # | Assumption | Check |
|---|---|---|
| A1 | H is real symmetric or complex Hermitian: `max|H − H†| ≤ 1e-10` | `_check_preconditions` |
| A2 | |ψ⟩ is a vector of dimension matching H: `len(ψ) == H.shape[0]` | `_check_preconditions` |
| A3 | |ψ⟩ is non-zero: `⟨ψ|ψ⟩ > 1e-30` | `_check_preconditions` |
| A4 | Arithmetic is interval arithmetic (Arb/Acb): the certificate records `backend` | `_arb_rayleigh` / `_acb_rayleigh` |

If any assumption fails, `rayleigh_certificate` raises `ValueError` before
issuing a certificate.

### Failure modes

| Mode | Symptom | Root cause |
|---|---|---|
| **F1. H not symmetric** | `ValueError: not symmetric` at issue time | Rounding or asymmetric construction; fix source |
| **F2. ψ = 0** | `ValueError: zero norm` | All-zero trial state |
| **F3. Complex H, real-only backend** | `backend = "numpy-float"`, `radius = 0` | `python-flint` absent; bound has no certified rounding |
| **F4. Trial state far from GS** | Large `upper − E0` gap | Variational: `upper` is a valid but loose bound; not a failure of the theorem |
| **F5. Bond-dimension truncation** | `upper` includes truncation error in `⟨ψ|H|ψ⟩` | `[OUT]`: truncation error is outside Rayleigh-Ritz scope; must be bounded separately |
| **F6. Modeling error** | `upper` is for the wrong H | `[OUT]`: model selection is outside HTF scope |

### Verification algorithm  (`verify_rayleigh_certificate` / `htf-verify`)

1. Load the full certificate JSON (produced by `cert.to_full_json()`).
2. Reconstruct H and ψ from the `canonical` section using `_decode_canonical`.
3. Confirm `SHA-256(input_bytes) == input_digest`; abort if mismatch.
4. Re-check all assumptions A1–A3 independently.
5. Recompute `⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩` with Arb (real) or Acb (complex) interval arithmetic.
6. Confirm `recomputed_upper ≤ stored_upper + tol` where `tol = max(|upper|×1e-14, 1e-15)`.
7. If all checks pass, return `verified = True`.

### What is NOT certified

- The tightness of the bound (upper − E0 gap is not bounded from above by this certificate).
- Bond-dimension truncation error in computing ψ from an MPS: `[OUT]`.
- The continuum limit (χ → ∞): `[OUT]`.

---

## TC-2 · Variational Upper Bound on Ground-State Energy

**Claim form:**  `E0 ≤ E_var`  where `E_var` is the energy of an optimised
trial state (e.g., MERA or MPS).

**Status:** `[engineering]` — a corollary of TC-1 applied after variational
optimisation.

### Theorem

Same as TC-1 (Rayleigh-Ritz).  The variational principle adds that after
minimisation over a parameterised family:

```
E0 ≤ min_{θ} Re(⟨ψ(θ)|H|ψ(θ)⟩ / ⟨ψ(θ)|ψ(θ)⟩)
```

### Assumptions (beyond TC-1)

| # | Assumption | Check |
|---|---|---|
| V1 | Optimisation converges to a local minimum (L-BFGS-B tolerance) | `optimize_mera` / `dmrg_sweep_mpo_2site` convergence flag |
| V2 | MERA isometry constraints hold: `max defect ≤ 1e-10` | `check_isometry` called after each update |

### Failure modes (beyond TC-1)

| Mode | Symptom | Root cause |
|---|---|---|
| **V-F1. Local minimum** | `E_var ≫ E0` | Optimiser trapped; run `dmrg_multistart` |
| **V-F2. Isometry defect** | `E_var < true_E0` (apparent violation) | Broken isometry constraint; fix with `enforce_isometry` |
| **V-F3. χ too small** | Large `E_var − E0` that doesn't decrease with χ | Area-law violation or large entanglement; `[research]` |

### What is NOT certified

- The gap to the second excited state: TC-1/V only gives a one-sided bound.
- That the local minimum is the global minimum.

---

## TC-3 · Spectral Gap Certified Upper Bound

**Claim form:**  `gap ≤ E1_var − E0_var`  where `E0_var`, `E1_var` are
variational bounds on the ground and first-excited state.

**Status:** `[heuristic]` — `E1_var − E0_var` is NOT a rigorous spectral gap
upper bound.  See P0-2.

### Attempted theorem

`gap = E1 − E0`.  If `E0 ≤ E0_var` and `E1 ≤ E1_var`, then `gap ≤ E1_var − E0_var`.

### Why this does NOT hold as implemented (P0-2)

`certified_gap_upper` in `htf/gap.py` computes `E1_var − E0_var` where:
- `E0_var` = variational upper bound on E0 (valid, ≥ E0).
- `E1_var` = variational upper bound on E1 (valid, ≥ E1).

The subtraction `E1_var − E0_var` satisfies:

```
E1_var − E0_var ≥ E1 − E0 = gap       (NOT guaranteed)
```

Because `E1_var ≥ E1` and `E0_var ≥ E0`, the subtraction can go either way:
it is a **heuristic estimate**, not a certified upper bound.

A rigorous spectral gap upper bound requires `E0_var − E0_lower_bound`, and
a rigorous lower bound on E0 is harder to obtain (see TC-4).

### Notes in the current implementation

`htf/gap.py` contains a warning:
> `certified_gap_upper` is NOT a rigorous spectral gap upper bound:
> E1_var − E0_var ≠ E1_upper − E0_lower.  Use as a heuristic estimate only.

### Verification algorithm

None: this claim is `[heuristic]` and does not issue a `RayleighCertificate`.
Do not trust as a rigorous bound.

### Path to rigorous TC-3  `[research]`

Requires either:
(a) A certified upper bound on E1 (e.g., via a second Rayleigh certificate
    with a trial vector orthogonal to the ground state), combined with a
    certified lower bound on E0 (see TC-4).
(b) A Lanczos-based two-sided bound (see TC-4) for both E0 and E1.

---

## TC-4 · Temple / Lanczos Heuristic Lower Bound

**Claim form:**  `E0 ≥ E_temple`  (heuristic; NOT a rigorous lower bound).

**Status:** `[heuristic]` — the current implementation uses the second Ritz
value as the denominator, which can produce pseudo-lower-bounds.  See P0-1.

### Intended theorem (Temple's inequality)

Given Rayleigh quotient R = ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ and variance
Δ² = ⟨ψ|H²|ψ⟩/⟨ψ|ψ⟩ − R², if E1 > R:

```
E0 ≥ R − Δ² / (E1 − R)
```

where E1 is the exact second eigenvalue (NOT a Ritz approximation).

### Why this does NOT hold as implemented (P0-1)

`temple_lower_bound` / `temple_lanczos` in `htf/gap.py` and `htf/lanczos.py`
use the second Ritz value ε1 as a proxy for E1.  Since ε1 ≥ E1 (Ritz values
are upper bounds), the denominator `ε1 − R` is larger than the true
`E1 − R`, making the lower bound less tight — or, if ε1 is a poor
approximation, potentially invalid as a strict lower bound.

### Notes in the current implementation

`temple_lower_bound` is annotated:
> Heuristic label: uses second Ritz value as denominator.  Not a certified
> lower bound.  P0-1.

### Verification algorithm

None: this claim is `[heuristic]` and should not be presented as a
machine-verified bound.

### Path to rigorous TC-4  `[research]`

Requires one of:
(a) A certified upper bound on E1 (from an independent Rayleigh certificate
    for a trial state ⊥ to the GS), substituted as the denominator in
    Temple's inequality.
(b) Anderson's two-sided Lanczos bounds, which use the tridiagonal structure
    to certify both upper and lower eigenvalue estimates.
(c) Eigenvalue enclosure methods (interval arithmetic on the characteristic
    polynomial or Krylov residuals).

---

## TC-5 · OS-Positivity Diagnostic

**Claim form:**  "The transfer matrix and reflection operator satisfy the
finite-lattice OS axioms."

**Status:** `[heuristic]` for the current implementation — the checks pass
for all real symmetric H and do not distinguish genuine OS-positive systems
from others.  See P0-5.

### Intended theorem (Osterwalder-Schrader, 1973–1975)

A quantum field theory reconstructed from a Euclidean path integral satisfies
unitarity if and only if the Euclidean measure is OS-positive (reflection
positive).  On the lattice: the transfer matrix T is PSD and commutes with
the reflection operator R.

### Why this does NOT work as intended (P0-5)

The three checks in `finite_lattice_reflection_diagnostics` are:

1. Transfer matrix T = exp(−δH) is PSD.
2. [H, R] = 0 (H commutes with R).
3. OS-Gram matrix G = T + R·T·R ≽ 0.

For any **real symmetric** H, all three checks pass automatically:
- T = exp(−δH) is PSD for any real symmetric H.
- Any permutation matrix R commutes with any real diagonal matrix, and a
  conjugated real matrix is still symmetric; the check as written is
  insufficient to detect non-commuting structure.
- G = T + RTR ≽ 0 follows from T PSD.

The checks are therefore **vacuous** for real symmetric H — they confirm
existence of a PSD transfer matrix, not genuine OS-positivity.

### Notes in the current implementation

The function was renamed from `os_positivity_report` to
`finite_lattice_reflection_diagnostics` and carries a deprecation warning:
> These diagnostics pass for ALL real symmetric H; they are not a genuine
> OS-positivity certificate.  P0-5.

### Verification algorithm

None: this claim is `[heuristic]` and does not issue a machine-verified certificate.

### Path to rigorous TC-5  `[research]`

Genuine OS-positivity checking requires:
(a) A non-trivial reflection operator R that does not commute with H by
    construction (e.g., spatial reflection on a non-diagonal H).
(b) Explicit verification that the Schwinger functions satisfy the OS
    positivity condition: sum of |⟨Oi Oj⟩| ≥ 0 for reflected pairs.
(c) For lattice gauge theory: checking Wightman axioms is `[OUT]`.

---

## TC-6 · ZX-Calculus Rewrite Soundness

**Claim form:**  "The ZX rewrite log constitutes a proof that the simplified
circuit is semantically equivalent to the original."

**Status:** `[research]` — rewrite rules are implemented and checked against
matrix semantics; the log is a proof sketch.

### Theorem

Each ZX rewrite rule in `htf/zx.py` is an axiom of the ZX-calculus (van de
Wetering, 2020).  A sequence of sound rules applied to a valid ZX-graph
preserves the represented linear map.

### Assumptions (machine-checked)

| # | Assumption | Check |
|---|---|---|
| Z1 | Input graph is a valid ZX-graph (nodes have declared types) | `ZXGraph` constructor |
| Z2 | Each applied rule matches its syntactic precondition | `spider_fusion`, `identity_removal`, etc. |
| Z3 | `zx_to_matrix` computes the correct dense unitary by tensor-network contraction | 19 regression tests comparing to known unitaries |

### Failure modes

| Mode | Symptom | Root cause |
|---|---|---|
| **Z-F1. Phase accumulation error** | Rewritten matrix ≠ original | Phase arithmetic in `spider_fusion` off by 2π |
| **Z-F2. Boundary port mismatch** | Wrong tensor contraction order | Input/output ports not tracked correctly |
| **Z-F3. Non-Clifford rules applied** | Unsound simplification | Rules outside the Clifford fragment applied to non-Clifford angles |

### Verification algorithm

1. Compute the dense unitary of the original circuit: `U_orig = zx_to_matrix(original)`.
2. Apply rewrite rules to get `simplified`.
3. Compute `U_simp = zx_to_matrix(simplified)`.
4. Check `max|U_orig − U_simp| < 1e-10`.

The `ZXRewriteLog` records each rule application.  Independent verification
replays steps 1–4.

### What is NOT certified

- Completeness of the rule set (rules may not find the globally minimal form).
- Soundness of the Clifford fragment beyond what regression tests cover.

---

## TC-7 · Interval Arithmetic Rounding

**Claim form:**  The arithmetic backend propagates floating-point rounding
exactly, so `radius` is a certified bound on the floating-point error in the
Rayleigh quotient.

**Status:** `[engineering]` when `backend = "flint-arb"` or `"flint-acb"`.
`[OUT]` (no certification) when `backend` contains `"numpy-float"`.

### Theorem

**Arb ball arithmetic (Johansson, 2017).**  `python-flint` wraps FLINT's Arb
library, which implements rigorous real-ball arithmetic: every operation on
`arb` (or `acb` for complex) objects produces an output interval that is
guaranteed to contain the true mathematical result, accounting for all
floating-point rounding.

### Assumptions

| # | Assumption | Check |
|---|---|---|
| I1 | `python-flint` ≥ 0.5 is installed | `from flint import arb` succeeds |
| I2 | Input arrays are finite (no NaN/Inf) | Implicit in `_check_preconditions` (would manifest as Arb exceptions) |
| I3 | Working precision ≥ 53 bits (default for Arb) | Arb default; explicitly documented |

### Failure modes

| Mode | Symptom | Root cause |
|---|---|---|
| **I-F1. flint absent** | `radius = 0`, `backend = "numpy-float ..."` | `python-flint` not installed; install with `pip install python-flint` |
| **I-F2. Large n** | `radius` grows with matrix size | Accumulated ball widths; for n ≫ 100, `radius` may exceed useful precision |
| **I-F3. Ill-conditioned H** | Wide interval (large `radius`) | H has large condition number; the certified bound is wide but valid |

### Verification algorithm

Check that `cert.backend` starts with `"flint-arb"` or `"flint-acb"`.  If
not, treat the certificate as discovery-tier only.

---

## TC-8 · Input Integrity (SHA-256 Digest)

**Claim form:**  The certificate is bound to the exact (H, ψ) that produced
it; any substitution is detectable.

**Status:** `[engineering]`.

### Mechanism

`input_digest = SHA-256(H_f64_bytes ‖ ψ_f64_bytes)`
(for complex: real and imag parts concatenated separately).

Stored in the certificate.  The full certificate (`to_full_json()`) also
embeds the canonical arrays so a verifier can reconstruct and check.

### Failure modes

| Mode | Symptom | Root cause |
|---|---|---|
| **D-F1. Input substitution** | Digest mismatch in verifier | Certificate re-used with different H or ψ |
| **D-F2. Precision loss in serialisation** | Digest mismatch | JSON serialisation with insufficient float precision; HTF uses `tolist()` (full float64 precision) |
| **D-F3. HTF version mismatch** | `htf_version` field differs | Certificate produced by a different HTF version; re-issue if API changed |

### Verification algorithm

1. Reconstruct H, ψ from `canonical` section.
2. Compute `SHA-256(H_bytes ‖ ψ_bytes)`.
3. Compare to `input_digest`.  Any mismatch → reject.

---

## Summary Table

| Card | Claim | Status | Certified |
|---|---|---|---|
| TC-1 | `E0 ≤ upper` (Rayleigh-Ritz) | `[engineering]` | ✅ Yes — `RayleighCertificate` |
| TC-2 | `E0 ≤ E_var` (variational) | `[engineering]` | ✅ Yes — corollary of TC-1 |
| TC-3 | Spectral gap upper bound | `[heuristic]` | ❌ No — P0-2 |
| TC-4 | Temple / Lanczos lower bound | `[heuristic]` | ❌ No — P0-1 |
| TC-5 | OS-positivity diagnostics | `[heuristic]` | ❌ No — P0-5 |
| TC-6 | ZX rewrite soundness | `[research]` | ⚠️ Partial — regression tested, not formally proved |
| TC-7 | Interval arithmetic rounding | `[engineering]` | ✅ Yes — when `backend = "flint-arb/acb"` |
| TC-8 | Input integrity (SHA-256) | `[engineering]` | ✅ Yes — all certificates |

Cards TC-3, TC-4, TC-5 correspond to P0 defects.  They remain in HTF as
labelled heuristics.  Rigorous replacements require additional research (paths
described in each card).  No certificate is issued for `[heuristic]` claims.
