# HTF — Holographic Tensor Framework

**A certified, type-safe string-diagram / tensor-network framework.**

HTF is a Python DSL in which physics models are built as **string diagrams**
(applied category theory), compiled by a **functor** to concrete tensors, and
executed by a **tensor engine** — with two features most tensor-network tooling
lacks:

1. **Certification** — results can carry a machine-checkable rigorous error
   bound (bond-dimension truncation + finite precision), turning a variational
   estimate into a *certified upper/lower bound*.
2. **Proof-carrying structure** — physical properties (reflection positivity,
   gauge invariance, unitarity) are enforced by the type system and
   machine-checked, so *a structurally illegal network does not compile*.

> **Status:** `v0.23.0` — all four layers implemented and tested (1212 tests,
> ≥98% coverage). The certified (interval) engine, proof-carrying structure
> checks, MPS/MPO/DMRG/TEBD, finite-temperature states, parallel multi-start
> DMRG, ZX-calculus Clifford pipeline, QASM 2.0 interop, Lanczos two-sided
> bounds, JAX autograd, and agent-drivable CLI + MCP server are all live.

## What HTF is — and is not

HTF is a **certified *model* engine, not a "world engine".** It certifies
*numerical / truncation* error, **not** *modeling* error, and the continuum
limit (`χ → ∞`) is **a wall the framework does not cross**.
Concretely:

- It **does** give: a type-safe, composable, reproducible DSL; a tensor engine
  with (roadmap) certified error bounds; certified *finite-lattice* spectral
  bounds; machine-checked structural properties; and honest "difficulty maps"
  of how estimates degrade toward the continuum.
- It **does not** claim: immunity to UV divergence (the bond dimension is a
  regulator, not a cure); a proof of the continuum Yang–Mills mass gap; that a
  MERA *is* an AdS geometry (that link is a heuristic); or to predict physical
  reality (modeling error is outside its scope, and tensor networks are limited
  to area-law entanglement).

See `docs/project-canvas.zh.md` for the full design canvas.

## Install

```bash
git clone https://github.com/telleroutlook/htf
cd htf
pip install -e .            # core (numpy)
pip install -e ".[dev]"     # + pytest
pip install -e ".[accel]"   # + jax / opt_einsum (optional)
```

## Quickstart

```python
import numpy as np
from htf import Wire, Box, TensorFunctor, contract

spin = Wire("spin", 2)
psi = Box("psi", (), (spin,))      # state
U   = Box("U", (spin,), (spin,))    # gate
phi = Box("phi", (spin,), ())       # effect
diagram = psi >> U >> phi            # () -> ()  (illegal wiring raises TypeError)

F = TensorFunctor({
    "psi": np.array([1.0, 0.0]),
    "U":   np.array([[0.0, 1.0], [1.0, 0.0]]),   # swap
    "phi": np.array([0.0, 1.0]),
})
print(float(contract(diagram, F)))   # 1.0  =  <phi| U |psi>
```

Command line (JSON output, agent-friendly):

```bash
htf version
htf hello        # runs the diagram above, prints a JSON provenance certificate
```

## Architecture (three layers)

| Layer | Module | Role |
|---|---|---|
| 1 · Symbolic topology | `htf/topology.py` | `Wire`, `Box`, `Diagram`; `>>` and `@`; type checking |
| 2 · Functorial mapping | `htf/functor.py` | assign & validate concrete tensors |
| 3 · Tensor engine | `htf/engine.py` | contract — `float` (discovery) and `certified` (flint Arb interval) modes |
|   · MPS / TEBD / DMRG | `htf/mps.py`, `htf/tebd.py` | MPS, TEBD, 1/2-site DMRG, TDVP, Heisenberg/Bose-Hubbard, periodic BC |
|   · MPO / MPO-DMRG | `htf/mpo.py` | MPO, MPO-DMRG (1/2-site), parallel multi-start DMRG, χ-convergence study |
|   · Finite temperature | `htf/thermal.py` | MPS purification, imaginary-time TEBD, parallel β-scan |
|   · Structure checks | `htf/structure.py` | isometry, unitarity, reflection positivity |
|   · Variational / MERA | `htf/mera.py`, `htf/variational.py` | binary MERA, L-BFGS-B optimisation, certified upper bound |
|   · Spectral gap | `htf/gap.py`, `htf/lanczos.py` | Temple bounds, Lanczos two-sided certified bounds |
|   · ZX-calculus | `htf/zx.py` | 8-rule Clifford simplification, proof-carrying rewrite log |
|   · QASM interop | `htf/qasm.py` | import/export QASM 2.0, circuit unitary, HTF diagram bridge |
|   · Open systems | `htf/open_systems.py` | CPTP maps, Lindblad, steady state |
|   · Inverse design | `htf/inverse.py` | Hamiltonian learning, JAX autograd (optional) |
|   · Symmetric tensors | `htf/symmetric.py` | U(1) block-sparse tensors |
|   · Provenance | `htf/certificate.py` | replayable certificate (no trusted PASS) |
|   · CLI / MCP | `htf/cli.py`, `htf/mcp_server.py` | agent-drivable JSON I/O; MCP server for LLM agents |

## Roadmap & scope

See `docs/whitepaper.en.md` for detailed API examples, architecture rationale,
honest scope, and evidence grammar. `docs/project-canvas.zh.md` contains the
full design canvas in Chinese.

## License

MIT — see `LICENSE`.
