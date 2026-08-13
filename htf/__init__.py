"""HTF — a certified, type-safe string-diagram / tensor-network framework.

Phase 1 (v0.0.1): topology, functor, float engine, certificate, CLI.
Phase 2 (v0.1.0): 1-D lattice operators; certified mode (flint Arb).
Phase 3 (v0.2.0): structure verification (proof-carrying); MERA tensor
  network; variational energy + certified upper bound.

Honest scope: HTF is a *certified model engine*, not a "world engine". It
certifies numerical/truncation error, not modeling error; the continuum
limit (``chi → ∞``) is a wall the framework does not cross. See ``PLAN.md``.
"""
from __future__ import annotations

from .certificate import Certificate
from .engine import contract
from .functor import TensorFunctor
from .lattice import effect_box, heat_step_box, laplacian_box, site_wire, state_box
from .mera import MERA, MERALayer, random_mera
from .structure import (
    StructureReport,
    check_box_isometry,
    check_box_unitary,
    check_isometry,
    check_reflection_positivity,
    check_unitary,
    enforce_isometry,
    enforce_unitary,
    gram_min_eig,
    isometry_defect,
    unitary_defect,
)
from .topology import Box, Diagram, Id, Wire
from .variational import (
    energy_expectation,
    optimize_mera,
    transverse_ising_ham,
    variational_bound,
    xx_model_ham,
)

__version__ = "0.2.0"
__all__ = [
    # topology
    "Wire", "Box", "Id", "Diagram",
    # functor
    "TensorFunctor",
    # engine
    "contract",
    # certificate
    "Certificate",
    # lattice (Phase 2)
    "site_wire", "laplacian_box", "heat_step_box", "state_box", "effect_box",
    # structure (Phase 3)
    "StructureReport",
    "isometry_defect", "unitary_defect",
    "check_isometry", "check_unitary",
    "check_box_isometry", "check_box_unitary",
    "gram_min_eig", "check_reflection_positivity",
    "enforce_isometry", "enforce_unitary",
    # mera (Phase 3)
    "MERALayer", "MERA", "random_mera",
    # variational (Phase 3)
    "transverse_ising_ham", "xx_model_ham",
    "energy_expectation", "variational_bound", "optimize_mera",
    # meta
    "__version__",
]
