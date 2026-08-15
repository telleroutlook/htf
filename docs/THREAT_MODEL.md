# Threat Model — HTF Rayleigh Certificate Pipeline

Adapted from the proofctl threat-model pattern for the HTF certificate chain.

## Scope

This document covers the Rayleigh certificate pipeline:
`rayleigh_certificate()` → `RayleighCertificate` → `verify_rayleigh_certificate()` / `verify_from_dict()`.

It does **not** cover the broader HTF tensor engine, MPS/MPO modules, or ZX module.

---

## Trust Boundaries

| Boundary | Trust level | Rationale |
|---|---|---|
| HTF Python package (installed, unmodified) | **Trusted** | In-process; digest + precondition checks run within this boundary |
| Canonical inputs `(H, ψ)` at call time | **Trusted** | Caller asserts these are the intended matrices |
| Serialised certificate JSON on filesystem | **Untrusted** | May be read from disk, network, or third-party storage; treated as adversarial input |
| `canonical` section of a full certificate | **Untrusted** | Digest must be recomputed from it before use |
| python-flint / Arb library | **Trusted** | External library; assumed correct for interval arithmetic |
| mpmath, numpy | **Trusted for cross-check only** | Used as independent arithmetic paths; systematic errors in either would not undermine Arb's rigour |

---

## Attacker Capabilities

The threat model assumes an attacker who:

1. **Can write or modify files on the filesystem** — including certificate JSON files.
2. **Can replace one certificate file with another** (swap attack).
3. **Cannot modify the in-process HTF binary** during a single Python session.
4. **Cannot break SHA-256 preimage resistance** — the attacker cannot craft a different `(H, ψ)` that collides with a stored `input_digest`.
5. **Cannot forge flint-Arb interval arithmetic** — the attacker cannot produce a float upper bound lower than the true Rayleigh quotient without the digest also failing.

---

## Attack Vectors and Mitigations

### A1 · Certificate swap

**Description:** Attacker replaces a certificate with one computed for a different `(H, ψ)`.

**Mitigation:** `input_digest` is a domain-separated SHA-256 of the canonical `(H, ψ)` (see INV-04 in `SECURITY-INVARIANTS.md`). `verify_from_dict` recomputes the digest from the `canonical` section and rejects any mismatch. `verify_rayleigh_certificate` does the same from `cert._H_canonical`.

**Residual risk:** None, assuming SHA-256 preimage resistance.

---

### A2 · Upper-bound inflation (weakened claim)

**Description:** Attacker increases `interval.upper` to make the certificate appear to certify a weaker (larger) bound.

**Mitigation:** `verify_from_dict` independently recomputes `upper` via Arb arithmetic and checks `recomputed_upper ≤ stored_upper` (strict, no tolerance). Inflated `stored_upper` passes this check — but is not a stronger claim, only a weaker one. The verifier also checks `claim` text encodes `upper` exactly (step 2), and the cleanroom exact path independently confirms the rational Rayleigh quotient is ≤ `stored_upper`.

**Residual risk:** An inflated upper bound is a valid but weaker certificate; HTF does not prevent weakening.

---

### A3 · Upper-bound deflation (forged tighter claim)

**Description:** Attacker decreases `interval.upper` to make the certificate claim a tighter bound than was actually computed.

**Mitigation:** `verify_from_dict` checks `recomputed_upper ≤ stored_upper`. If `stored_upper` is deflated below `recomputed_upper`, this check fails and the certificate is rejected.

**Residual risk:** None under the assumption that the attacker cannot find `H, ψ` with a lower Rayleigh quotient without triggering a digest failure.

---

### A4 · NaN / Inf injection

**Description:** Attacker sets `interval.upper = NaN` or `interval.upper = Inf` in the certificate.

**Mitigation:** `validate_certificate_dict` calls `math.isfinite` on all four interval fields and raises `ValueError` on non-finite values (INV-06 context). `verify_from_dict` also checks `math.isfinite(stored_upper)` and `math.isfinite(recomputed_upper)` before the comparison.

**Residual risk:** None for NaN/Inf in the interval dict fields.

---

### A5 · Tolerance bypass (claim passing with enlarged tolerance)

**Description:** Attacker exploits a tolerance in the upper-bound comparison to pass a slightly-inflated recomputed upper.

**Mitigation:** The comparison `recomputed_upper ≤ stored_upper` uses **no tolerance** (strict `<=`). See INV-09 in `SECURITY-INVARIANTS.md`.

**Residual risk:** None; tolerance-free comparison is enforced by the strict check.

---

### A6 · Backend substitution

**Description:** Attacker modifies the `backend` field to claim a different arithmetic backend was used.

**Mitigation:** `verify_from_dict` recomputes the backend string from the inputs and compares it with `stored_backend` (step 4). `verify_rayleigh_certificate` does the same.

**Residual risk:** None; the backend string is derived from the same Arb computation that produces `upper`.

---

### A7 · Theorem / assumption text tamper

**Description:** Attacker modifies `theorem` or `assumptions` to change the mathematical claim without altering the numerical values.

**Mitigation:** `verify_from_dict` checks that `theorem` matches `EXPECTED_THEOREM` exactly (step 2). `verify_rayleigh_certificate` does the same. The `assumptions` list is re-derived by `_check_preconditions` during verification and is not taken at face value. See INV-10 in `SECURITY-INVARIANTS.md`.

**Residual risk:** None for the theorem field. The `assumptions` list is informational; the actual preconditions are re-checked independently.

---

### A8 · `replay_mode` spoofing

**Description:** Attacker sets `replay_mode="from_scratch"` on a certificate that was actually produced by a self-consistency re-check, to make it appear as a fresh computation.

**Mitigation:** `replay_mode` is set by the producer (`rayleigh_certificate` → `"from_scratch"`, `verify_rayleigh_certificate` → `"self_consistency"`) and is carried in the certificate dict. The release policy `rayleigh-release-v1.json` requires `replay_mode="from_scratch"`; this check must be enforced by the release CI gate.

**Residual risk:** `replay_mode` is not cryptographically bound to the computation. An attacker who can write the certificate JSON can spoof it. Enforcement depends on the CI gate reading the policy, not on in-process arithmetic.

---

## Explicit Non-Mitigations

The following are **outside scope** — HTF does not mitigate them:

| Non-mitigation | Reason |
|---|---|
| **Cryptographic signing** of certificates | No signing key infrastructure exists; certificates are integrity-checked by digest + arithmetic, not by signature |
| **External timestamp authority** | Certificate issuance time (`htf_version`, `git_commit`) is informational; no trusted timestamp service is used |
| **Modeling error** | HTF certifies numerical/truncation error only; whether `H` correctly models the physical system is outside scope (`[OUT]` in evidence grammar) |
| **Volume-law entanglement** | Bond-dimension truncation is a regulator, not a cure; HTF does not certify area-law vs. volume-law regimes |
| **Continuum / infinite-volume limit** | Finite-lattice estimates only; `χ → ∞` is a wall HTF does not cross |
| **Side-channel attacks on the Python process** | Not in scope; HTF is a library, not a hardened binary |

---

## References

- `SECURITY-INVARIANTS.md` — code-location map for each invariant
- `docs/ASSURANCE_MODEL.md` — rigorous / reproducible / heuristic hierarchy
- `policies/rayleigh-release-v1.json` — machine-readable release gate policy
- `policies/rayleigh-dev-v1.json` — machine-readable development policy
- `htf/_cleanroom_verify.py` — independent stdlib-only Fraction verification path
