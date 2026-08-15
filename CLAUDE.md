# CLAUDE.md — HTF (Holographic Tensor Framework)

Project instructions for AI assistants (and humans) working in this repository.
These override default behaviour where they apply.

## 1. Project identity

HTF is a **certified, type-safe string-diagram / tensor-network framework**. The
distinguishing value is **certification** (rigorous error bounds) and
**proof-carrying structure** (physical properties enforced by types + machine
checks) — not "another tensor library".

**One-line honest scope:** HTF is a *certified model engine, not a world engine*.
It certifies numerical/truncation error, not modeling error; the continuum limit
(`χ → ∞`) is a wall HTF does not cross.

## 2. Evidence grammar (do not violate)

Every load-bearing claim in docs/issues/PRs carries exactly one tier:

- `[engineering]` — buildable with known tools, has precedent.
- `[research]` — genuine open research; feasible, valuable, not guaranteed.
- `[heuristic]` — an interpretive analogy, not an established method.
- `[OUT]` — explicitly **not** claimed (e.g. a proof of the continuum YM mass gap).

Status is derived by checks/tests, **never self-declared**. No "PASS" is trusted
without a replayable certificate.

## 3. Hard boundaries — never reintroduce these overclaims

Prior drafts were corrected to remove exactly these. Do **not** put them back:

- ❌ "immune to UV divergence" — the bond dimension `χ` is a **regulator**;
  UV divergence is traded for a controllable **truncation error**, not cured.
- ❌ "reads off / proves the Yang–Mills mass gap" — MERA gives a variational /
  finite-lattice estimate; the continuum, non-perturbative 4D gauge gap is `[OUT]`.
- ❌ "a MERA *is* a 5D AdS metric / holography is a graphics game" — the
  MERA↔AdS link is `[heuristic]` (Swingle), debated; present it as such.
- ❌ "abandons PDEs / absolute stability" — a difference operator represented as
  a box **is** a finite-difference operator; stability depends on the scheme.
- ❌ "a world engine that predicts reality" — certification bounds numerical
  error only; modeling error is outside scope; tensor networks are limited to
  **area-law** entanglement (volume-law / real-time dynamics / sign problem are
  hard boundaries, not engineering gaps).

When in doubt, state the error bar and the boundary; never drop them.

## 4. Honest scope & limitations

State the tool's limits concretely; do not dress them up as a research
philosophy. HTF certifies finite/local structure and maps how estimates degrade,
but it does not by itself cross continuum/global walls. Regularization (`χ`
truncation) ≠ solving the continuum; "reading off" a gap ≠ proving it. Put effort
into the certified finite/local layer and the tooling, and label continuum claims
`[OUT]`.

## 5. Architecture (keep the layers separate)

1. `htf/topology.py` — symbolic topology (`Wire`, `Box`, `Diagram`, `>>`, `@`,
   type checking). **No numerics here.**
2. `htf/functor.py` — assign & validate concrete tensors.
3. `htf/engine.py` — contraction. `mode="float"` (discovery-tier, no error bound)
   / `mode="certified"` (Phase 2; must **raise** until real interval bounds exist
   — never fake certification).
4. `htf/certificate.py` — replayable provenance; `htf/cli.py` — agent-drivable,
   JSON I/O.

Convention: a box `b: dom -> cod` has a tensor of shape `dims(cod) + dims(dom)`.

## 6. Engineering conventions

- **Run the tests** before claiming anything works: `python -m pytest -q`.
  Zero failures is the bar; fix pre-existing failures in the same change.
- **No fake certification.** `certified` mode raises until interval bounds are
  real. Floats are discovery-tier; label them so.
- **Language:** most of the repo is in **English** (code, README, CLAUDE, API,
  commit messages). `PLAN.md` may be in Chinese. Docstrings/comments English.
- **Privacy / portability:** never write personal absolute paths, usernames,
  company names, or internal network addresses into any file. Use relative paths.
- **Commits in English.** Run `git status` before committing.
- **No silent truncation:** if a computation caps `χ`, resolution, or order, say
  so in the output/certificate.

## 7. PR invariant checklist (adopted from proofctl)

Every PR that touches `htf/engine.py`, `htf/rayleigh_cert.py`, `htf/verify.py`,
`htf/_rayleigh_primitives.py`, `htf/mps_cert.py`, or `htf/certificate.py` must answer:

- Which invariant in `SECURITY-INVARIANTS.md` does this change affect?
- Does the change introduce a new trust input (new field, new digest, new assurance level)? If so, is it validated?
- What tamper / regression test covers it?

See `SECURITY-INVARIANTS.md` for the full invariant table.
See `docs/ASSURANCE_MODEL.md` for the assurance level hierarchy.

## 8. What this repository does NOT do

Does not prove the mass gap (`[OUT]`), does not claim UV immunity, does not treat
a finite-lattice estimate as a continuum theorem, does not present the MERA↔AdS
analogy as an established method, and does not claim to predict physical reality.
