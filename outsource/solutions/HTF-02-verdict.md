# HTF-02 Verdict — Adapter Data Semantics Review

**Reviewer type:** Internal self-review (no external reviewer available as of 2026-08-14)
**Review date:** 2026-08-14
**Commit reviewed:** b35bdc1 (HEAD at time of review)

---

## Overall verdict: CONDITIONAL

The adapter extraction logic is correct for the common case (uniform, real,
left-to-right ordered MPS with a matching H matrix).  Two documentation gaps
must be closed before the adapters should be used in a context where basis
ordering is not trivially obvious.

---

## Link-by-link assessment

### Link 1 — quimb `to_dense()` semantics

**CONFIRMED** for standard quimb MPS.

`MatrixProductState.to_dense()` returns the state vector in computational basis
order with the **last site varying fastest** (row-major, standard convention).
A `nn_hamiltonian` built with the same left-to-right site ordering matches.

**CONDITIONAL-1:** The adapter docstring does not state this ordering assumption.
Add: "Requires H to be built in the same computational basis ordering as
`mps.to_dense()` (standard left-to-right, last site fastest)."

### Link 2 — TeNPy `get_theta(0, L)` semantics

**CONFIRMED** for standard TeNPy MPS (uniform, no charge sorting).

`get_theta(0, L).to_ndarray().ravel()` gives C-order raveling of shape
`(1, d, d, …, d, 1)`, which is standard left-to-right computational basis
(first site slowest, last site fastest) — matching `nn_hamiltonian`.

For TeNPy MPS with non-trivial charge sorting (`undo_sort_charge` parameter),
the extracted vector may be in a permuted basis.  The adapter correctly passes
`undo_sort_charge=True` when available.

**CONDITIONAL-2:** Document that for systems with U(1)/SU(2) symmetry, the
caller must ensure `get_theta` is called in a way that removes charge sorting,
or the extracted ψ may not be in the standard computational basis.

### Link 3 — MPS gauge / normalization invariance

**CONFIRMED.**

`rayleigh_certificate(H, α·ψ)` produces the same `upper` as
`rayleigh_certificate(H, ψ)` because the Rayleigh quotient divides by ⟨ψ|ψ⟩.
Different input_digest values are correct (the actual bytes differ).

`get_theta(0, L)` contracts all bond matrices into the full state vector,
removing all gauge freedom.  The result is the unique physical state vector
regardless of left/right/mixed canonical form.

### Link 4 — Complex state handling

**CONFIRMED** with documentation note.

The `imag_tol=1e-10` default is appropriate for numerically real states where
imaginary parts come only from floating-point noise.  The adapters raise
`ValueError` on genuinely complex states, which is correct behaviour — they
route only to the real Arb path.

The adapter docstring should state: "For genuinely complex physical states,
pass the complex numpy array directly to `rayleigh_certificate()` rather than
using these adapters."  This prevents silent wrong certificates if a user
passes `imag_tol=1` on a complex state.

---

## Q1–Q4 answers

**Q1 (basis ordering):** Consistent for standard MPS, but not stated explicitly
in the docstring.  CONDITIONAL-1 closes this.

**Q2 (TeNPy non-uniform):** `get_theta` is correct for uniform systems;
charge-sorted systems require `undo_sort_charge=True`.  CONDITIONAL-2 closes.

**Q3 (gauge invariance):** Confirmed — `get_theta(0, L)` gives the physical
state vector independent of canonical form.

**Q4 (complex handling):** Confirmed — real adapter raises on complex state;
complex states should use `rayleigh_certificate()` directly.  Documentation
should make this explicit.

---

## Acceptance criteria outcome

**CONDITIONAL** — two documentation gaps (CONDITIONAL-1 and CONDITIONAL-2).
No code changes required.  The adapters are semantically correct for their
documented use case; the conditions only require docstring additions to prevent
misuse for non-standard MPS configurations.
