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

> **Status:** early skeleton (`v0.0.1`). Layer 1 (topology), a minimal Layer 2
> (functor) and Layer 3 (float engine), a provenance certificate, and an
> agent-drivable CLI are implemented and tested. The certified (interval) engine
> and the proof-carrying checks are on the roadmap (`PLAN.md`).

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
| 3 · Tensor engine | `htf/engine.py` | contract (float now; certified = Phase 2) |
|   · Provenance | `htf/certificate.py` | replayable certificate (no trusted PASS) |
|   · CLI | `htf/cli.py` | agent-drivable, JSON I/O |

## Roadmap & scope

See `PLAN.md` (roadmap, value tracks, honest boundaries) and
`docs/project-canvas.zh.md` (full design rationale).

## License

MIT — see `LICENSE`.
