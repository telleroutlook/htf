# outsource/ — self-contained review requests

Each file is a standalone task extracted from this HTF repository.
A reviewer needs **nothing else from this repo** to evaluate the problem.

## Format contract

- **Self-contained:** all definitions, code excerpts, claims, and verification steps are in the file.
- **Falsifiable:** if a step is wrong, the reviewer should return an explicit counterexample
  or gap description, not just "cannot verify."
- **No circularity:** no problem assumes the claim it is asking to verify.

## Status board

| # | File | Topic | Status |
|---|---|---|---|
| HTF-01 | `HTF-01-rayleigh-cert-audit.md` | Rayleigh certificate code+math correctness | BLOCKED → IMPLEMENTED — external referee verdict received 2026-08-13 (R1–R6 all blocked); all fixes applied (v2 schema, exact preconditions, `_outward_upper`/`_outward_lower`, domain-separated digest, component-preserving decode, strict verifier); see `solutions/HTF-01-referee-verdict.md` + `solutions/HTF-01-independent-audit.py` |
| HTF-02 | `HTF-02-adapter-semantics.md` | quimb/TeNPy adapter data semantics | BLOCKED → IMPLEMENTED — external referee verdict received 2026-08-13 (R1–R4 all blocked); all fixes applied (`bc='finite'` enforcement, `undo_sort_charge=True`, no silent `imag_tol` projection, `_preserve_state_dtype`); see `solutions/HTF-02-referee-verdict.md` |
| HTF-03 | `HTF-03-spectral-gap-math.md` | Temple lower bound + `first_excited_upper` mathematical correctness | IMPLEMENTED — verdict received 2026-08-14; all required changes applied (2D Ritz, temple_denominator_positive, heuristic_width) |
| HTF-04 | `HTF-04-acb-imaginary-check.md` | Acb imaginary-part containment: soundness of `q.imag.contains(0)` | IMPLEMENTED — verdict received 2026-08-14; backend string stabilised, error message updated |
| HTF-05 | `HTF-05-rayleigh-external-review.md` | Full Rayleigh certificate pipeline: external independent re-audit | IMPLEMENTED — verdict received 2026-08-14; M1–M6 all applied (dtype enforcement, NaN fail-closed, assurance required) |
| HTF-06 | `HTF-06-post-p0-comprehensive-review.md` | Post-P0 comprehensive review: full pipeline after all blockers resolved (G5 gate) | BLOCKED → FIXED — Gate-A review received 2026-08-15; B3/B4/B5 applied (conditional nextafter, stable midpoint, assumption text); see `solutions/HTF-06-Gate-A-independent-review.md` |

## What a returned review should contain

- Verdict: PASS / CONDITIONAL (exact required edit) / BLOCKED (exact gap + minimal repair).
- For each link: CONFIRMED / PARTIAL / REFUTED + explanation.
- For each question: a direct answer.
- Any numerical re-checks with independently computed values.
