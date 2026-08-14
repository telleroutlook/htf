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
| HTF-01 | `HTF-01-rayleigh-cert-audit.md` | Rayleigh certificate code+math correctness | CONDITIONAL (internal review; see `solutions/HTF-01-verdict.md`) |
| HTF-02 | `HTF-02-adapter-semantics.md` | quimb/TeNPy adapter data semantics | CONDITIONAL (internal review; see `solutions/HTF-02-verdict.md`) |
| HTF-03 | `HTF-03-spectral-gap-math.md` | Temple lower bound + `first_excited_upper` mathematical correctness | IMPLEMENTED — verdict received 2026-08-14; all required changes applied (2D Ritz, temple_denominator_positive, heuristic_width) |
| HTF-04 | `HTF-04-acb-imaginary-check.md` | Acb imaginary-part containment: soundness of `q.imag.contains(0)` | IMPLEMENTED — verdict received 2026-08-14; backend string stabilised, error message updated |
| HTF-05 | `HTF-05-rayleigh-external-review.md` | Full Rayleigh certificate pipeline: external independent re-audit | IMPLEMENTED — verdict received 2026-08-14; M1–M6 all applied (dtype enforcement, NaN fail-closed, assurance required) |

## What a returned review should contain

- Verdict: PASS / CONDITIONAL (exact required edit) / BLOCKED (exact gap + minimal repair).
- For each link: CONFIRMED / PARTIAL / REFUTED + explanation.
- For each question: a direct answer.
- Any numerical re-checks with independently computed values.
