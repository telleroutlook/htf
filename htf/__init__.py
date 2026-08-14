"""HTF — certified model engine (htf-spec + htf-verify).

Architecture
------------
This top-level namespace exposes only the **certified core**:

**htf-spec** — symbolic topology, functor, engine, provenance certificate.
**htf-verify** — Rayleigh-Ritz certificates with rigorous Arb/Acb interval
arithmetic and an independent verifier.

For the full experimental/research suite (MPS, MPO, TEBD, DMRG, ZX, thermal,
variational, etc.) use :mod:`htf.labs`::

    import htf.labs as labs
    from htf.labs import MPS, tebd_evolve, dmrg_sweep_mpo, ...

Individual submodules remain directly importable regardless::

    from htf.mps import MPS
    from htf.gap import first_excited_upper

For backend adapters (quimb, TeNPy)::

    from htf.adapters.quimb_adapter import rayleigh_from_quimb_mps
    from htf.adapters.tenpy_adapter import rayleigh_from_tenpy_mps

Honest scope
------------
HTF is a *certified model engine*, not a "world engine".  It certifies
floating-point rounding error (via Arb/Acb), not modelling error; the
continuum limit (``chi → ∞``) is a wall the framework does not cross.
See ``PLAN.md`` and ``docs/theorem_cards.md``.
"""
from __future__ import annotations

# ── htf-spec: symbolic topology ───────────────────────────────────────────────
from .topology import Box, Diagram, Id, Wire

# ── htf-spec: functor + engine + certificate ──────────────────────────────────
from .certificate import Certificate
from .engine import contract
from .functor import TensorFunctor

# ── htf-verify: Rayleigh certificates + independent verifier ─────────────────
from .rayleigh_cert import (
    RayleighCertificate,
    rayleigh_certificate,
    rayleigh_estimate,
    verify_rayleigh_certificate,
)
from .verify import verify_from_dict

__version__ = "0.23.0"

__all__ = [
    # htf-spec: symbolic topology
    "Wire",
    "Box",
    "Id",
    "Diagram",
    # htf-spec: functor + engine + certificate
    "TensorFunctor",
    "contract",
    "Certificate",
    # htf-verify: certified Rayleigh certificates
    "RayleighCertificate",
    "rayleigh_certificate",
    "rayleigh_estimate",
    "verify_rayleigh_certificate",
    "verify_from_dict",
]
