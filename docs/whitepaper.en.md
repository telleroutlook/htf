# HTF Design Whitepaper

**Holographic Tensor Framework — Certified, Type-Safe String-Diagram / Tensor-Network Engine**

Version 0.13.0 · Language: Python 3.10+ · License: MIT

---

## 1. What HTF Is

HTF is a **certified model engine** for tensor-network computations.

Its distinguishing property is **certification**: every numerical result carries a
machine-checked, replayable error bound (floating-point rounding, via Arb interval
arithmetic). This makes results *auditable* — not just fast or convenient.

**One-sentence honest scope:**
HTF certifies numerical / truncation error on a *finite lattice*; it does not certify
modeling error and it does not cross the continuum limit (`χ → ∞`).

---

## 2. What HTF Is Not

| Claim | Status |
|---|---|
| Immune to UV divergence | `[OUT]` — the bond dimension χ is a regulator; UV divergence is traded for a controllable **truncation error**, not cured. |
| Proves the Yang–Mills mass gap | `[OUT]` — MERA gives a variational / finite-lattice estimate; the continuum non-perturbative 4D gauge gap is out of scope. |
| A holographic 5D geometry engine | `[heuristic]` — the MERA ↔ AdS analogy (Swingle) is a debated interpretive frame, not an established method. |
| A world engine that predicts reality | `[OUT]` — certification bounds numerical rounding error, not modeling error; tensor networks are limited to area-law entanglement states. |

---

## 3. Architecture

HTF is structured in four layers. **Numerics never leak upward.**

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 4  Certified Results + Certificates                      │
│           Certificate(result, error_bound, mode, backend)       │
│           BenchmarkReport  StructureReport  DifficultyReport    │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3  Engine  (htf/engine.py)                               │
│           mode="float"     discovery-tier, no error bound       │
│           mode="certified" flint Arb interval arithmetic        │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2  Functor  (htf/functor.py)                             │
│           Assign & validate concrete tensors to Box labels.     │
│           Shape mismatch → TypeError at assignment time.        │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1  Topology  (htf/topology.py)                           │
│           Wire, Box, Diagram, >>, @                             │
│           Type checking at construction; NO numerics here.      │
└─────────────────────────────────────────────────────────────────┘
```

**Convention:** a box `b: dom → cod` has a tensor of shape `dims(cod) + dims(dom)`.

---

## 4. Core Capabilities

### 4.1 Type-Safe String Diagrams (Phase 1)

```python
from htf import Wire, Box, Id, TensorFunctor, contract

s = Wire("spin", 2)
psi = Box("psi", (), (s,))   # state:  () → spin
U   = Box("U",   (s,), (s,)) # gate:   spin → spin
phi = Box("phi", (s,), ())   # effect: spin → ()

diagram = psi >> U >> phi    # () → ()  (a scalar)
```

Type violations are caught at construction:

```python
t = Wire("time", 4)
Box("bad", (s,), (t,)) >> Box("also_bad", (s,), ())  # TypeError at >>
```

### 4.2 Certified Mode (Phase 2)

```python
cert = contract(diagram, F, mode="certified")
# cert.result      — float midpoint
# cert.error_bound — strict FP rounding bound (flint Arb)
# cert.mode        — "certified"
```

Error bounds for a 20-step heat equation: `error_bound ≈ 5.6 × 10⁻¹⁶` (≈1 machine ε).

### 4.3 Structure Verification / Proof-Carrying (Phase 3)

```python
from htf import check_isometry, check_unitary, check_reflection_positivity, enforce_isometry

r = check_isometry(tensor)   # StructureReport: defect = ||MM† − I||_max
r = check_unitary(tensor)    # defect = max(||MM† − I||, ||M†M − I||)
```

MERA layers are initialised via `random_mera` (SVD-based exact isometry) and can be
reprojected with `enforce_isometry` / `enforce_unitary`.

### 4.4 Variational Ground State (Phase 3)

```python
from htf import transverse_ising_ham, random_mera, optimize_mera, variational_bound

H     = transverse_ising_ham(n=4, J=1.0, h=0.5)
mera0 = random_mera(4, chi=2, seed=42)
mera_gs, history = optimize_mera(H, mera0, n_iter=100)
cert  = variational_bound(H, mera_gs)   # certified E_0 upper bound
```

The variational principle guarantees `cert.result ≥ E_0` exactly; `cert.error_bound`
bounds the floating-point rounding in computing `cert.result`.

### 4.5 Spectral Gap Bounds (Phase 4)

```python
from htf import spectral_gap_exact, gap_report, temple_lower_bound

gap_exact = spectral_gap_exact(H)
report    = gap_report(H, psi_gs, psi_es)
# report["gap_exact"]  — exact gap from full diagonalisation
# report["gap_var"]    — variational upper bound E1_var − E0_var
# report["temple_lb"]  — Temple's inequality lower bound on E_0
# report["gap_cert"]   — Certificate: certified FP rounding bound on gap_var
```

**Temple's inequality** provides a rigorous finite-lattice lower bound on `E_0`:

    E_0 ≥ E_var − (⟨H²⟩ − E_var²) / (E1_upper − E_var)

Valid when `E_var < E_1_exact`; returns `−∞` otherwise.

### 4.6 OS-Positivity Machine Check (Phase 4)

```python
from htf import os_positivity_report

rep = os_positivity_report(H, n_sites=4, beta=1.0)
# rep["transfer_positivity"]  — T = exp(−βH) is PSD
# rep["reflection_symmetry"]  — [H, R] = 0
# rep["os_gram_positivity"]   — G = T + RTR is PSD
# rep["all_passed"]           — bool
```

This is a **finite-lattice necessary condition** for OS-positivity; the continuum
statement is `[OUT]`.

### 4.7 Entanglement / Difficulty Map (Phase 4)

```python
from htf import difficulty_report

drep = difficulty_report(H, n_sites=4)
# drep.entanglement_profile  — S(cut) for cut = 1, 2, ..., n−1
# drep.likely_area_law       — heuristic (discovery-tier)
```

Area-law states (S ≈ const) are accessible to MERA with moderate χ. Volume-law /
critical states require exponentially larger χ; the continuum limit is `[OUT]`.

### 4.8 Open Systems / CPTP Maps (§4-H)

```python
from htf.open_systems import (
    density_matrix_from_pure, partial_trace, check_density_matrix,
    choi_matrix, check_kraus_completeness,
    lindblad_superoperator, lindblad_step, steady_state,
)

rho0   = density_matrix_from_pure(psi)
rho_A  = partial_trace(rho0, n_sites=2, keep_sites=[0])
dm_rep = check_density_matrix(rho0)     # Hermitian + PSD + unit-trace checks
rho_ss = steady_state(H, lindblad_ops)  # Lindblad steady state
```

### 4.9 Reproducibility Benchmark (§4-K)

```python
from htf.benchmark import run_benchmark

report = run_benchmark(n_sites=4, chi=2, n_iter=50, seed=0)
print(report.summary())
report.to_json()   # replayable JSON certificate
```

Every `BenchmarkReport` records `htf_version`, `seed`, `n_iter`, and all certified
results; given the same inputs and HTF version, output is bit-for-bit reproducible.

### 4.10 ZX-Calculus Diagram Rewriting (§4-E) `[研究]`

```python
from htf.zx import zx_from_circuit, simplify, zx_to_matrix, ZXRewriteLog
from htf.qasm import Gate

gates = [Gate("h", [0]), Gate("cx", [0, 1])]
g   = zx_from_circuit(gates, n_qubits=2)
log = ZXRewriteLog()
n   = simplify(g, log=log)  # applies 5 rules exhaustively
U   = zx_to_matrix(g)       # reconstruct unitary (circuit-topology only)
```

Five sound rewrite rules: `spider_fusion`, `identity_removal`, `hadamard_cancel`,
`color_change`, `pi_copy`. Every step is recorded in a proof-carrying `ZXRewriteLog`.

**`zx_to_matrix`** raises `NotImplementedError` for non-circuit-topology graphs
(cross-wire nodes, disconnected components) — no silent wrong results.

### 4.11 QASM 2.0 Interoperability (§4-F) `[工程]`

```python
from htf.qasm import circuit_to_qasm, qasm_to_circuit, circuit_unitary, circuit_to_diagram

gates  = qasm_to_circuit(open("bell.qasm").read())
U      = circuit_unitary(gates, n_qubits=2)
qasm   = circuit_to_qasm(gates, n_qubits=2)
diag   = circuit_to_diagram(gates, n_qubits=2)   # HTF Diagram bridge
```

`circuit_to_diagram` uses SWAP decomposition by default (`adjacent_only=False`) so
non-adjacent two-qubit gates are structurally exact rather than full-width fallbacks.

### 4.12 U(1) Block-Sparse Tensors (§4-G) `[研究]`

```python
from htf.symmetric import spin_half_basis, check_u1_invariance, u1_blocks, BlockSparseTensor

b   = spin_half_basis()          # charges ±1
res = check_u1_invariance(Z, dom_bases=[b], cod_bases=[b])
bst = u1_blocks(Z, dom_bases=[b], cod_bases=[b])
print(bst.sparsity())            # fraction of entries stored
```

`check_u1_invariance` uses numpy broadcasting (vectorized O(d²) check, no Python loop).
Only abelian U(1) is implemented; SU(N) is `[研究]`.

### 4.13 Lanczos Two-Sided Spectral Bounds (§4-I) `[研究]`

```python
from htf.lanczos import temple_lanczos, two_sided_bounds

bounds = temple_lanczos(H, k=30)
# bounds.E0_upper         — Ritz variational upper bound
# bounds.E0_lower         — Temple's inequality lower bound
# bounds.temple_condition_met — True if E_var < E_1_ritz
# bounds.width            — interval width (upper − lower)
```

Full Gram-Schmidt re-orthogonalization (`reorthogonalize=True`, default) prevents
ghost eigenvalues at large `k`. Temple's bound is a rigorous finite-lattice lower
bound on `E_0`; the continuum gap is `[OUT]`.

### 4.14 Inverse Design / Hamiltonian Learning (§4-J) `[工程]`/`[研究]`

```python
from htf.inverse import inverse_design, hamiltonian_learning

result  = inverse_design(target_e0=-1.5, model="ising", n_sites=4)
# result.E0_achieved   — energy at recovered parameters
# result.params_opt    — best-fit parameter vector
# result.converged     — bool

lresult = hamiltonian_learning(target_energies=[E0, E1], model="ising", n_sites=4)
```

Uses L-BFGS-B with random restarts. Local minima are possible; uniqueness is
model-dependent and not guaranteed `[研究]`.

### 4.15 Lean 4 Proof-Skeleton Export (§4-L) `[研究]`

```python
from htf.lean_export import LeanExporter, certificate_to_lean, gap_report_to_lean

exp = LeanExporter()
exp.add_certificate(cert, "energy_bound")
exp.add_gap_report(report, "ising_gap")
exp.write("htf_proofs.lean")   # valid Lean 4; every sorry = proof obligation
```

Generates valid Lean 4 syntax with `import Mathlib`. Every `theorem` has a `sorry`
stub that marks a proof obligation for a human Lean expert. HTF does not invoke the
Lean server; use `lake build` externally to type-check. Full formalisation is `[研究]`.

---

## 5. Agent-Drivable CLI

All major operations are available as subcommands with JSON output:

```bash
htf version
htf hello
htf variational  --n 4 --chi 2 --n-iter 50
htf gap          --n 4 --chi 2 --n-iter 50
htf difficulty   --n 4
htf os-check     --n 4 --beta 1.0
htf benchmark    --n 4 --chi 2 --models ising xx
htf lanczos      --n 4 --k 30
htf qasm-sim     --file bell.qasm
htf zx-simplify  --file bell.qasm
htf inverse      --n 4 --target-e0 -1.5
htf lean-export  --n 4 --output proofs.lean
```

The JSON output is designed for LLM agents: the certificate constrains the agent
from overstating precision.

### MCP Server (§7)

```bash
htf-mcp   # start stdio MCP server
```

Or add to an MCP client config:

```json
{
  "mcpServers": {
    "htf": { "command": "python3", "args": ["-m", "htf.mcp_server"] }
  }
}
```

Available tools: `htf_version`, `htf_variational`, `htf_gap`, `htf_os_check`,
`htf_benchmark`, `htf_lanczos`, `htf_qasm_simulate`, `htf_zx_simplify`, `htf_inverse`.

---

## 6. Evidence Grammar

Every load-bearing claim in docs, issues, and PRs carries exactly one tier:

| Tag | Meaning |
|---|---|
| `[工程]` / `[engineering]` | Buildable with known tools, has precedent. |
| `[研究]` / `[research]` | Genuine open research; feasible, not guaranteed. |
| `[启发]` / `[heuristic]` | Interpretive analogy, not an established method. |
| `[OUT]` | Explicitly **not** claimed. |

Status is derived by checks and tests — never self-declared. No "PASS" is trusted
without a replayable certificate.

---

## 7. Dependencies

| Package | Role | Required |
|---|---|---|
| `numpy>=1.23` | Core array operations | Yes |
| `scipy` | Matrix expm (open systems, transfer matrix) | Yes |
| `python-flint>=0.5` | Arb interval arithmetic (certified mode) | Optional (`pip install htf[certified]`) |
| `mcp>=1.0` | MCP server transport | Optional (`pip install htf[mcp]`) |
| `opt_einsum` | Optimised contraction-path selection (float mode) | Optional (auto-detected) |

---

## 8. Honest Limitations

HTF does **not**:

- Prove the Yang–Mills mass gap or any continuum result (`[OUT]`)
- Certify bond-dimension truncation bias (χ-bias is discoverable, not certified)
- Handle volume-law entanglement states efficiently (area-law only)
- Claim UV-immunity (χ is a regulator, not a cure)
- Certify Lindblad dynamics (float mode only)
- Cross the continuum limit (`χ → ∞`, `L → ∞`)
- Execute ZX `zx_to_matrix` on non-circuit-topology graphs (raises `NotImplementedError`)
- Provide complete ZX-calculus completeness (local soundness only; completeness is `[研究]`)
- Produce formal Lean proofs (generates `sorry`-stubs; full formalisation is `[研究]`)
- Guarantee uniqueness in inverse design (local minima possible; `[研究]`)

The framework's value is in the **finite/local certified layer**: provenance-carrying
certificates, type-enforced structural properties, and replayable benchmarks.
