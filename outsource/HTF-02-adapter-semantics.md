# HTF-02 — Adapter Data Semantics Review

**Type:** Gate-A independent review (adapter interface correctness)

**What this is.** A request to independently verify that the HTF adapters for quimb and
TeNPy correctly extract the quantum state vector from an MPS object, and that the resulting
`RayleighCertificate` is semantically meaningful: `cert.upper` is a genuine upper bound on
the ground-state energy of the Hamiltonian H **for the state that the MPS represents**.

The reviewer needs nothing from the HTF repository beyond what is quoted in this file.

**Scope.** This review covers the *adapter interface contract* — what the adapter promises,
what the backend API actually delivers, and whether these align.  It does NOT re-verify
the Rayleigh-Ritz theorem itself (that is HTF-01).

---

## All definitions (self-contained)

### The semantic guarantee (what each adapter promises)

Both adapters promise:

> Given an MPS object `mps_like` representing a quantum state |ψ_MPS⟩, and a full
> Hamiltonian matrix H (real symmetric or complex Hermitian, shape n×n), produce a
> `RayleighCertificate` certifying `E0 ≤ ⟨ψ_MPS|H|ψ_MPS⟩ / ⟨ψ_MPS|ψ_MPS⟩`.

In other words: `cert.upper` is a rigorous upper bound on the ground-state energy of H
**in the Hilbert space of dimension n**, for the specific state encoded in the MPS.

### Adapter 1 — quimb (`rayleigh_from_quimb_mps`)

**Interface:** `mps_like.to_dense()` → numpy array, then `.ravel()` → 1-D state vector.

```python
def _extract_state_vector(mps_like, *, imag_tol):
    if not hasattr(mps_like, "to_dense"):
        raise TypeError(...)
    raw = np.asarray(mps_like.to_dense()).ravel()
    if np.iscomplexobj(raw):
        max_imag = float(np.abs(raw.imag).max())
        if max_imag > imag_tol:
            raise ValueError("MPS to_dense() returned complex ...")
        raw = raw.real
    return raw.astype(np.float64)

def rayleigh_from_quimb_mps(mps_like, H, *, imag_tol=1e-10, notes=""):
    psi = _extract_state_vector(mps_like, imag_tol=imag_tol)
    return rayleigh_certificate(np.asarray(H), psi, notes=f"quimb-adapter: ...; {notes}")
```

**quimb convention (from quimb docs):** `MatrixProductState.to_dense()` returns the full
state vector as a numpy array of shape `(2**n,)` for n qubits (d=2), or `(d**n,)` in
general, in the **computational basis** ordering (|000⟩, |001⟩, ..., |111⟩).

### Adapter 2 — TeNPy (`rayleigh_from_tenpy_mps`)

**Interface:** `mps_like.L` (int, number of sites) + `mps_like.get_theta(0, L)` → tensor,
then `.to_ndarray()` (if TeNPy Array) or direct `.ravel()`, giving the 1-D state vector.

```python
def _extract_tenpy_state_vector(mps_like, *, imag_tol):
    if hasattr(mps_like, "get_theta") and hasattr(mps_like, "L"):
        n_sites = int(mps_like.L)
        theta = mps_like.get_theta(0, n_sites)
        if hasattr(theta, "to_ndarray"):
            raw = theta.to_ndarray().ravel()
        else:
            raw = np.asarray(theta).ravel()
    elif hasattr(mps_like, "to_dense"):
        raw = np.asarray(mps_like.to_dense()).ravel()
    else:
        raise TypeError(...)
    if np.iscomplexobj(raw):
        max_imag = float(np.abs(raw.imag).max())
        if max_imag > imag_tol:
            raise ValueError("MPS get_theta() returned complex ...")
        raw = raw.real
    return raw.astype(np.float64)

def rayleigh_from_tenpy_mps(mps_like, H, *, imag_tol=1e-10, notes=""):
    psi = _extract_tenpy_state_vector(mps_like, imag_tol=imag_tol)
    return rayleigh_certificate(np.asarray(H), psi, notes=f"tenpy-adapter: ...; {notes}")
```

**TeNPy convention:** `MPS.get_theta(i, n)` returns the n-site theta tensor starting at
site i, of shape `(chi_L, d_i, d_{i+1}, ..., d_{i+n-1}, chi_R)` where `chi_L = chi_R = 1`
for the full-chain theta (i=0, n=L).  Raveling this shape gives the state vector in the
computational basis ordering `(chi_L=1) × d_0 × d_1 × ... × d_{L-1} × (chi_R=1)`.

---

## The four links under review

**Link 1 — quimb `to_dense()` semantics.**
Does `mps.to_dense().ravel()` always return the amplitude vector ⟨x|ψ⟩ in the
**standard computational basis** (matching the ordering of H)?  Specifically:
- In quimb, the local physical dimension ordering for `to_dense()` is row-major
  (last site varies fastest by default, i.e. `|00…0⟩, |00…1⟩, …`).
- The H matrix must be built with the same ordering.
- Are there quimb MPS types or site orderings where `to_dense()` uses a different ordering?

**Link 2 — TeNPy `get_theta(0, L)` semantics.**
Does `theta.to_ndarray().ravel()` give the same computational-basis ordering as a
conventionally-built H?  Specifically:
- The theta tensor has shape `(1, d, d, …, d, 1)` for a uniform system.
- After `.to_ndarray()`, the layout is C-order (row-major): the first physical index
  varies slowest.  This is `|0 0 … 0⟩, |0 0 … 1⟩, …` (last site fastest).
- TeNPy's `MPS` can have a site ordering (`sites` attribute) that may not be the
  standard left-to-right convention.  Does `get_theta` always respect the ordering
  assumed by the H matrix?

**Link 3 — MPS gauge / normalization invariance.**
The Rayleigh quotient is invariant under overall normalization of ψ (since it divides
by ⟨ψ|ψ⟩).  Both adapters pass the extracted ψ (which may not be normalized) to
`rayleigh_certificate`, which correctly handles unnormalized states.
- Confirm: `rayleigh_certificate(H, α·ψ) == rayleigh_certificate(H, ψ)` for any α ≠ 0.
  (They produce different `input_digest` values but the same `upper`.)
- MPS gauge freedom: an MPS in left/right/mixed canonical form has the same physical state
  |ψ_MPS⟩ but different individual tensors.  `get_theta(0, L)` contracts all tensors into
  the full state vector, so the gauge does NOT affect the extracted ψ.  Confirm this is
  correct for TeNPy's `get_theta` implementation.

**Link 4 — Complex state handling.**
Both adapters raise `ValueError` if the extracted ψ has `max|imag| > imag_tol = 1e-10`.
This rejects complex physical states (e.g. time-evolved states, states under complex H).
- The `imag_tol` parameter allows the user to override this.
- For real-valued H and a physical state with only numerical imaginary parts (from
  floating-point, typically |imag| < 1e-14), the rejection threshold is appropriate.
- For genuinely complex states: the adapters' imaginary-part rejection is intentional
  (the real-only path is taken for these adapters).  A complex psi should be passed
  directly to `rayleigh_certificate(H, complex_psi)` bypassing the adapters.
- Confirm: is the `imag_tol` mechanism documented clearly enough that a user with a
  complex physical state will not silently get a wrong certificate by passing `imag_tol=1`?

---

## The Gate-A questions

### Q1 — Basis ordering consistency
For the certificate to be semantically valid, the basis ordering of the extracted ψ must
match the basis ordering of H.  Is this guaranteed by the adapter code, or is it an
**implicit assumption** placed on the caller (who must build H in the same basis as the MPS)?

If it is a caller assumption, should it be stated explicitly in the adapter docstring?
Propose the minimal documentation addition (or a runtime check) that would make this
requirement explicit.

### Q2 — TeNPy `get_theta` for non-uniform systems
For a TeNPy MPS with non-uniform local dimension (e.g. a mix of spin-1/2 and spin-1 sites),
does `get_theta(0, L).to_ndarray().ravel()` give the correct amplitude vector with the
expected dimension `d_0 × d_1 × … × d_{L-1}`?  If so, the caller must pass H of the same
dimension.  Is there a dimension mismatch check beyond `len(psi) == H.shape[0]`?

### Q3 — Silent complex truncation
The quimb adapter calls `raw = raw.real` (discarding the imaginary part) when
`max|imag| ≤ imag_tol`.  The resulting certificate has `input_digest` for the **real** ψ,
not the original complex ψ.  A caller who passed a complex MPS with tiny imaginary parts
(|imag| ≈ 1e-12) will get a certificate for a **slightly different state** than they
passed.  Is this:
(a) Documented and acceptable (numerical imaginary part = floating-point artefact; real
    part is the correct state)?
(b) A semantic gap that should be flagged more loudly?

### Q4 — Hamiltonian H: caller's responsibility
The adapters take H as a plain `numpy.ndarray`.  They do NOT verify that H is the correct
Hamiltonian for the physical system the MPS was computed for.  The certificate will be
valid as a Rayleigh-quotient certificate for H, but if the caller passes the wrong H
(e.g. a different coupling, different boundary condition), the certificate is meaningless
even though it is internally consistent.

Is this limitation clearly documented?  Should the adapters include a warning in the notes
field indicating that H was provided externally and its correctness is the caller's
responsibility?

### Q5 — Gate-A verdict
Given Q1–Q4 and the four links: do the quimb and TeNPy adapters correctly deliver on their
semantic promise — `cert.upper` is a genuine upper bound on E0(H) for the state |ψ_MPS⟩?
Identify any assumption that is implicit but should be explicit, or any failure mode not
currently caught.

Verdict: PASS / CONDITIONAL (with exact required documentation or code change) /
BLOCKED (with exact gap).

---

## Numerical anchor (sanity only)

For a 2-site spin-1/2 system (d=2, L=2), H = diag(0, 1, 1, 2) (Ising-like, standard basis
ordering `|00⟩, |01⟩, |10⟩, |11⟩`), MPS state |00⟩ = [1, 0, 0, 0]:

```
ψ = [1, 0, 0, 0]
⟨ψ|H|ψ⟩ = 0,  ⟨ψ|ψ⟩ = 1,  RQ = 0.0
E0(H) = 0.0  (exact ground state)
cert.upper ≤ 1e-9  (up to Arb rounding)
```

For a quimb mock:
```python
class MockMPS:
    def to_dense(self): return np.array([[1.0, 0.0], [0.0, 0.0]])  # shape (2,2)

cert = rayleigh_from_quimb_mps(MockMPS(), H)
assert cert.upper <= 1e-9   # passes in test suite
```

For a TeNPy mock:
```python
class MockTeNPyMPS:
    L = 4  # length 4 so get_theta(0,4) returns shape (1,4,1)
    def get_theta(self, i, n):
        return np.array([[[1.0, 0.0, 0.0, 0.0]]])  # shape (1,4,1)

cert = rayleigh_from_tenpy_mps(MockTeNPyMPS(), H)
assert cert.upper <= 1e-9   # passes in test suite
```

Both pass the HTF test suite (1427 tests green, including 26 quimb adapter tests and
32 TeNPy adapter tests).

---

## Acceptance criteria

1. **PASS:** all four links confirmed, Q1–Q5 answered with no blocking gap.
2. **CONDITIONAL:** semantics are correct but a specific documentation or code change is
   required.  Give the exact required change (e.g. add a docstring note, add a warning to
   cert.notes).
3. **BLOCKED:** a genuine semantic gap exists — the extracted ψ does NOT correctly represent
   the MPS state, or the certificate is not a valid bound for a non-trivial reason.
   Give the minimal repair.

An honest "CONDITIONAL — add explicit basis-ordering documentation" is a valuable outcome.
