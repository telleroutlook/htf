# Security Invariants

This document maps each certification security invariant to its production code
location and test function. A failing test here is a blocking invariant violation,
not a quality issue.

Adopted from the proofctl pattern: each invariant is explicit, tested, and referenced
here so that reviewers can verify coverage without reading the full test suite.

## Invariant Status

| INV | Description | Code Location | Test Function | Status |
|-----|-------------|---------------|---------------|--------|
| INV-01 | `certified` mode raises `ImportError` if python-flint unavailable — never silently degrades to float | `htf/engine.py:212-213` | `test_phase2.py::TestCertifiedAmplitude::test_returns_certificate` (import-error path) | ✅ ACTIVE |
| INV-02 | `assurance="heuristic"` certificates always return `verified=False` — must never be treated as a proof | `htf/rayleigh_cert.py:595-640` | `test_verify.py::TestVerifyRayleighCertificate::test_heuristic_cert_returns_verified_false` | ✅ ACTIVE |
| INV-03 | `input_digest` must be a 64-char lowercase hex string; malformed digest rejected on load | `htf/rayleigh_cert.py:252-255` | `test_verify.py::TestVerifyRayleighCertificate::test_raises_on_missing_input_digest` | ✅ ACTIVE |
| INV-04 | Tampered `input_digest` → `verify_rayleigh_certificate` fails with `verified=False` | `htf/verify.py` | `test_verify.py::TestVerifyRayleighCertificate::test_tampered_digest_fails` | ✅ ACTIVE |
| INV-05 | Tampered H or ψ in canonical section → verify fails | `htf/verify.py` | `test_verify.py::TestVerifyRayleighCertificate::test_tampered_H_in_canonical_fails`, `test_tampered_psi_in_canonical_fails` | ✅ ACTIVE |
| INV-06 | Tampered `backend`, `claim`, or `theorem` fields → verify fails | `htf/verify.py` | `test_verify.py::TestVerifyRayleighCertificate::test_tampered_backend_fails`, `test_tampered_claim_fails`, `test_tampered_theorem_fails` | ✅ ACTIVE |
| INV-07 | `_outward_upper`/`_outward_lower` never produce `inf` for finite DBL_MAX input (B3 fix) | `htf/_rayleigh_primitives.py:170-185` | `test_cleanroom_verify.py::TestHTF06GateARegressions` | ✅ ACTIVE |
| INV-08 | Midpoint computed as `lo + (up−lo)/2`, not `(lo+up)/2`, preventing overflow for large inputs (B4 fix) | `htf/rayleigh_cert.py:429`, `htf/verify.py:250` | `test_cleanroom_verify.py::TestHTF06GateARegressions` | ✅ ACTIVE |
| INV-09 | Assumption text in `_check_preconditions` must not embed float64 computation results (no format specs or `float()` vars) | `htf/_rayleigh_primitives.py:52` | `test_cert_hygiene.py::test_assumption_text_is_exact_logical`, `test_no_float_format_spec_in_return_list` | ✅ ACTIVE |
| INV-10 | `outsource/` docs must not contain bare `...` ellipses in code blocks (no silent placeholder truncation) | `outsource/**/*.md` | `test_outsource_hygiene.py` | ✅ ACTIVE |
| INV-11 | Inflated `radius` field → verify rejects (stored interval must match recomputed interval) | `htf/verify.py` | `test_verify.py::TestVerifyRayleighCertificate::test_tampered_radius_inflated_fails` | ✅ ACTIVE |
| INV-12 | `replay_mode` provenance: `rayleigh_certificate()` always sets `"from_scratch"`; `verify_rayleigh_certificate()` always sets `"self_consistency"` | `htf/rayleigh_cert.py` (producer line ~454, verifier line ~582) | `test_rayleigh_cert.py::TestReplayMode` | ✅ ACTIVE |
| INV-13 | Cleanroom exact check: `verify_from_dict` computes the exact rational Rayleigh quotient via stdlib `Fraction` (no flint/mpmath) and asserts it ≤ `Fraction.from_float(stored_upper)` | `htf/verify.py:286` | `test_verify.py::TestINV13CleanroomCheck` | ✅ ACTIVE |

## How to add a new invariant

1. Assign the next INV-XX number.
2. Implement the enforcement in `htf/` (preferred in `verify.py` or `_rayleigh_primitives.py`).
3. Add at least one test that directly triggers the invariant (tamper the relevant field).
4. Add the row to this table with code location (file:line) and test function name.
5. `python -m pytest -q` must pass before merging.

## PR invariant checklist

Every PR that touches `htf/engine.py`, `htf/rayleigh_cert.py`, `htf/verify.py`,
`htf/_rayleigh_primitives.py`, `htf/mps_cert.py`, or `htf/certificate.py` must answer:

- Which invariant does this change affect?
- Does the change introduce a new trust input (new field, new digest, new assurance level)? If so, is it validated?
- What tamper / regression test covers it?
