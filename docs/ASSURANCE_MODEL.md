# Assurance Model

Adopted from the proofctl assurance model pattern.

## Overview

HTF tracks the epistemic category of each verification result explicitly.
Assurance levels are never collapsed into a single numeric score. A certificate
carries exactly one assurance level; the consumer must check it before treating
the result as a rigorous bound.

The guiding principle is: **different assurance levels answer different questions**
and are not interchangeable.

## Assurance Levels

### `rigorous`

An interval-arithmetic bound was computed using python-flint's Arb/Acb library
with outward-rounded arithmetic. The result is a mathematically rigorous upper
bound on E₀ in the sense that the true Rayleigh quotient lies within the
reported interval.

**Suitable for:** Published bounds, release certificates, Gate-A sign-off.
**Requires:** python-flint installed; `assurance="rigorous"` in the certificate.
**Verification:** Independent re-computation via `verify_rayleigh_certificate`
must reproduce an interval that contains the stored bound within the stored
radius; `verified=True` is returned only if this check passes.

### `reproducible`

A floating-point computation was performed and the result is bound to its
canonical inputs via `input_digest` (SHA-256 of H and ψ). The result is
reproducible in the sense that the same inputs will produce the same output,
but it carries no rigorous error bound.

**Suitable for:** Development-phase checks, regression tests, intermediate steps.
**Not suitable for:** Release gate sign-off or published bounds.

### `heuristic`

A floating-point estimate was computed without digest binding. No `input_digest`
is recorded; the result cannot be independently replayed from the certificate
alone.

**Always forbidden in:** Any context where `verified=True` would be load-bearing.
`verify_rayleigh_certificate` always returns `verified=False` for heuristic
certificates.

**Suitable for:** Quick exploratory estimates during development only.

## Why No Single Score

A single score (e.g. "confidence: 0.95") would:

1. Allow trading off assurance levels — e.g. replacing one `rigorous` certificate
   with many `heuristic` estimates to reach the same threshold.
2. Obscure the epistemic basis — a score of 0.9 could mean "interval arithmetic"
   or "repeated float estimate" with very different implications.
3. Make policy violations harder to detect.

HTF instead maintains a strict per-certificate assurance level that is checked
independently by `verify_rayleigh_certificate`.

## Assurance Level Hierarchy

```
rigorous  ──▶  reproducible  ──▶  heuristic
  (best)                            (worst)
```

A consumer that requires `rigorous` must reject certificates at lower levels.
There is no upcast: a `reproducible` certificate cannot be promoted to `rigorous`
without re-running the computation with Arb/Acb arithmetic.

## Default Forbidden Levels

| Level | Reason for prohibition in release context |
|-------|-------------------------------------------|
| `heuristic` | No digest binding; result is not independently verifiable from the certificate. |

`verify_rayleigh_certificate` enforces this: `assurance="heuristic"` → `verified=False`
unconditionally (see INV-02 in `SECURITY-INVARIANTS.md`).

## MPS/MPO Extension

`rayleigh_certificate_mps` supports the same assurance hierarchy:

| Level | MPS path |
|-------|----------|
| `rigorous` | Arb/Acb transfer-matrix contraction (`_arb_rayleigh_mps`) |
| `reproducible` | NumPy float contraction with digest binding |

See `htf/mps_cert.py` for details.
