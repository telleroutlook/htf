"""HTF — a certified, type-safe string-diagram / tensor-network framework.

Phase 1 (v0.0.1): topology, functor, float engine, certificate, CLI.
Phase 2 (v0.1.0): 1-D lattice operators; certified mode (flint Arb).
Phase 3 (v0.2.0): structure verification (proof-carrying); MERA tensor
  network; variational energy + certified upper bound.
Phase 4 (v0.4.0): spectral gap bounds (Temple's inequality); χ-convergence
  study; entanglement entropy / difficulty map; OS-positivity machine check;
  expanded agent-drivable CLI (gap, variational, difficulty, os-check).

Honest scope: HTF is a *certified model engine*, not a "world engine". It
certifies numerical/truncation error, not modeling error; the continuum
limit (``chi → ∞``) is a wall the framework does not cross. See ``PLAN.md``.
"""
from __future__ import annotations

from .certificate import Certificate
from .benchmark import BenchmarkReport, BenchmarkResult, run_benchmark
from .difficulty import (
    DifficultyReport,
    bipartite_entanglement_profile,
    difficulty_report,
    entanglement_entropy,
    entanglement_spectrum,
)
from .engine import contract
from .functor import TensorFunctor
from .gap import (
    certified_gap_upper,
    first_excited_upper,
    gap_report,
    h2_expectation,
    spectral_gap_exact,
    temple_lower_bound,
)
from .lattice import effect_box, heat_step_box, laplacian_box, site_wire, state_box
from .mera import MERA, MERALayer, random_mera
from .os_axioms import (
    check_reflection_symmetry,
    check_transfer_positivity,
    os_positivity_report,
    reflection_operator,
    transfer_matrix,
)
from .scaling import ChiPoint, ScalingReport, chi_convergence_study
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

__version__ = "0.5.0"
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
    # gap (Phase 4)
    "spectral_gap_exact", "h2_expectation",
    "temple_lower_bound", "first_excited_upper",
    "certified_gap_upper", "gap_report",
    # os_axioms (Phase 4 completion)
    "transfer_matrix", "reflection_operator",
    "check_transfer_positivity", "check_reflection_symmetry",
    "os_positivity_report",
    # scaling (Phase 4)
    "ChiPoint", "ScalingReport", "chi_convergence_study",
    # benchmark (§4-K)
    "BenchmarkResult", "BenchmarkReport", "run_benchmark",
    # difficulty (Phase 4)
    "entanglement_entropy", "entanglement_spectrum",
    "bipartite_entanglement_profile",
    "DifficultyReport", "difficulty_report",
    # meta
    "__version__",
]
