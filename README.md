# HTF — Holographic Tensor Framework

> **A certified, type-safe string-diagram / tensor-network framework for Python.**

[![CI](https://github.com/telleroutlook/htf/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/telleroutlook/htf/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-0.23.0-blue)](#)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#)
[![Tests](https://img.shields.io/badge/tests-1590%20passing-brightgreen)](#)
[![Coverage](https://img.shields.io/badge/coverage-93%25-yellowgreen)](#)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

HTF is a Python DSL in which physics models are built as **string diagrams**
(applied category theory), compiled by a **functor** to concrete tensors, and
executed by a **tensor engine**. Its current distinctive feature:

- **Interval arithmetic** — contraction results carry flint-Arb interval bounds
  (finite-precision rounding certified; bond-dimension truncation bias is a
  separate, unresolved research gate).
- **Runtime shape-checked DSL** — wiring errors (mismatched dimensions) raise
  `TypeError` at composition time; physical-property checks (isometry, unitarity,
  reflection symmetry) run as explicit validators, not compile-time constraints.

> **Status (2026-08-14):** An independent audit identified P0 issues in the
> spectral-bound and ZX modules — see [PLAN.md](PLAN.md) §0.5 for the full
> remediation history. All P0 issues (P0-1–P0-7) and P1 issues are now closed;
> G0–G6 release gates are all closed. Certified branding is restored.

---

## Table of Contents

- [What HTF is — and is not](#what-htf-is--and-is-not)
- [Install](#install)
- [Quickstart](#quickstart)
- [Architecture](#architecture)
- [Examples](#examples)
- [Roadmap & Docs](#roadmap--docs)
- [Contributing](#contributing)
- [License](#license)

---

## What HTF is — and is not

HTF is a **certified *model* engine, not a "world engine".** It certifies
*numerical / truncation* error — not *modeling* error — and the continuum
limit (`χ → ∞`) is a wall the framework does not cross.

| HTF **does** | HTF **does not** |
|---|---|
| Type-safe, composable DSL (dimension-checked at composition) | Claim immunity to UV divergence (bond dimension is a *regulator*, not a cure) |
| Tensor engine with flint-Arb rounding-error bounds | Prove the continuum Yang–Mills mass gap (`[OUT]`) |
| Variational finite-lattice energy estimates | Provide certified two-sided spectral-gap bounds (Temple/Lanczos labelled heuristic; P0-1/P0-2 fixed) |
| Runtime structural-property validators | Enforce physical constraints at the type level (wire semantics, not just dimensions) |
| "Difficulty maps" showing how estimates degrade toward the continuum | Handle volume-law entanglement, real-time dynamics, or the sign problem |

---

## Install

```bash
git clone https://github.com/telleroutlook/htf
cd htf
pip install -e .              # core (numpy only)
pip install -e ".[dev]"       # + pytest, ruff, mypy
pip install -e ".[certified]" # + python-flint (Arb interval arithmetic)
pip install -e ".[accel]"     # + jax, opt_einsum (optional GPU / path acceleration)
pip install -e ".[mcp]"       # + MCP server for LLM agent integration
```

**Requirements:** Python ≥ 3.10, NumPy ≥ 1.23, SciPy (required at import time; listed under core in `pyproject.toml`).

---

## Quickstart

### Python API

```python
import numpy as np
from htf import Wire, Box, TensorFunctor, contract

spin = Wire("spin", 2)
psi  = Box("psi", (), (spin,))         # state
U    = Box("U",   (spin,), (spin,))    # gate
phi  = Box("phi", (spin,), ())         # effect

diagram = psi >> U >> phi              # () → ()  (illegal wiring raises TypeError)

F = TensorFunctor({
    "psi": np.array([1.0, 0.0]),
    "U":   np.array([[0.0, 1.0], [1.0, 0.0]]),  # X gate
    "phi": np.array([0.0, 1.0]),
})

print(float(contract(diagram, F)))     # 1.0  →  ⟨φ|U|ψ⟩
```

### CLI (agent-friendly JSON output)

```bash
htf version          # print version info as JSON
htf hello            # run the example diagram, emit a provenance certificate
```

The CLI speaks JSON throughout — every result includes a provenance record.
The `Certificate` dataclass is result metadata; for a fully replayable proof
artifact use `RayleighCertificate` (schema v2) from `htf.rayleigh_cert` with
`verify_rayleigh_certificate()` or the `htf-verify` CLI.

---

## Architecture

| Layer | Module | Role |
|---|---|---|
| 1 · Symbolic topology | `htf/topology.py` | `Wire`, `Box`, `Diagram`; `>>` / `@` composition; type checking |
| 2 · Functorial mapping | `htf/functor.py` | assign & validate concrete tensors |
| 3 · Tensor engine | `htf/engine.py` | contract — `float` (discovery) and `certified` (Arb interval) modes |
| · MPS / TEBD / DMRG | `htf/mps.py`, `htf/tebd.py` | MPS, TEBD, 1/2-site DMRG, TDVP, Heisenberg / Bose-Hubbard, periodic BC |
| · MPO / MPO-DMRG | `htf/mpo.py` | MPO, MPO-DMRG (1/2-site), parallel multi-start DMRG, χ-convergence study |
| · Finite temperature | `htf/thermal.py` | MPS purification, imaginary-time TEBD, parallel β-scan |
| · Structure checks | `htf/structure.py` | isometry, unitarity, reflection positivity |
| · Variational / MERA | `htf/mera.py`, `htf/variational.py` | binary MERA, L-BFGS-B optimisation, certified upper bound |
| · Spectral gap | `htf/gap.py`, `htf/lanczos.py` | Variational energy estimates; Temple/Lanczos bounds (labelled heuristic; P0-1/P0-2 fixed) |
| · ZX-calculus | `htf/zx.py` | 8-rule Clifford simplification; rewrite log (P0-6 fixed: CX/CZ/SWAP/Ry semantics correct) |
| · QASM interop | `htf/qasm.py` | import / export QASM 2.0, circuit unitary, HTF diagram bridge |
| · Open systems | `htf/open_systems.py` | CPTP maps, Lindblad, steady state |
| · Inverse design | `htf/inverse.py` | Hamiltonian learning, JAX autograd (optional) |
| · Symmetric tensors | `htf/symmetric.py` | U(1) block-sparse tensors |
| · Provenance | `htf/certificate.py` | replayable certificate (no trusted PASS) |
| · CLI / MCP | `htf/cli.py`, `htf/mcp_server.py` | agent-drivable JSON I/O; MCP server for LLM agents |

**Layer invariant:** topology has no numerics; certified mode raises until real
interval bounds exist — it never fakes a certificate.
**All P0 issues (P0-1–P0-7) are closed.** See [PLAN.md §0.5](PLAN.md) for the full remediation history.

---

## Examples

Runnable scripts in [`examples/`](examples/):

| Script | Topic |
|---|---|
| [`hello_world.py`](examples/hello_world.py) | Basic diagram composition and contraction |
| [`bell_circuit_qasm.py`](examples/bell_circuit_qasm.py) | Bell state via QASM 2.0 interop |
| [`zx_simplify_demo.py`](examples/zx_simplify_demo.py) | ZX-calculus Clifford simplification |
| [`mera_variational.py`](examples/mera_variational.py) | Binary MERA variational optimisation |
| [`lanczos_bounds.py`](examples/lanczos_bounds.py) | Lanczos two-sided certified spectral bounds |
| [`hamiltonian_learning.py`](examples/hamiltonian_learning.py) | Inverse design / Hamiltonian learning with JAX |
| [`heat_equation.py`](examples/heat_equation.py) | Finite-difference PDE as a tensor-network box |
| [`phase4_certified_physics.py`](examples/phase4_certified_physics.py) | Certified physics pipeline end-to-end |

---

## Roadmap & Docs

| Document | Contents |
|---|---|
| [`docs/whitepaper.en.md`](docs/whitepaper.en.md) | Detailed API examples, architecture rationale, evidence grammar, honest scope |
| [`docs/project-canvas.zh.md`](docs/project-canvas.zh.md) | Full design canvas (Chinese) |
| [`OPTIMIZATION.md`](OPTIMIZATION.md) | Performance notes and contraction-path tuning |

---

## Contributing

1. Fork, branch, and open a pull request against `main`.
2. Run `python -m pytest -q` — zero failures is the merge bar.
3. Keep the [evidence grammar](CLAUDE.md#2-evidence-grammar-do-not-violate) (`[engineering]` / `[research]` / `[heuristic]` / `[OUT]`) intact in docs and comments.
4. Do not reintroduce the [hard-boundary overclaims](CLAUDE.md#3-hard-boundaries--never-reintroduce-these-overclaims).

---

## License

MIT — see [`LICENSE`](LICENSE).
