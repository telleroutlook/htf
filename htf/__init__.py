"""HTF — a certified, type-safe string-diagram / tensor-network framework.

Phase 1 (v0.0.1): topology, functor, float engine, certificate, CLI.
Phase 2 (v0.1.0): 1-D lattice operators; certified mode (flint Arb).
Phase 3 (v0.2.0): structure verification (proof-carrying); MERA tensor
  network; variational energy + certified upper bound.
Phase 4 (v0.4.0): spectral gap bounds (Temple's inequality); χ-convergence
  study; entanglement entropy / difficulty map; OS-positivity machine check;
  expanded agent-drivable CLI (gap, variational, difficulty, os-check).
Phase 5 (v0.5.0): certified reproducibility benchmark suite (§4-K); MCP
  server wrapper (§7); open systems / CPTP maps (§4-H).
Phase 6 (v0.6.0): open systems / CPTP (§4-H complete); English whitepaper.
Phase 7 (v0.7.0): Lanczos two-sided bounds (§4-I); QASM 2.0 interop (§4-F).

Honest scope: HTF is a *certified model engine*, not a "world engine". It
certifies numerical/truncation error, not modeling error; the continuum
limit (``chi → ∞``) is a wall the framework does not cross. See ``PLAN.md``.
"""
from __future__ import annotations

from .benchmark import BenchmarkReport, BenchmarkResult, run_benchmark
from .certificate import Certificate
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
from .inverse import (
    InverseDesignResult,
    LearningResult,
    ParametricHam,
    energy_gradient,
    hamiltonian_learning,
    inverse_design,
)
from .lanczos import (
    TwoSidedBounds,
    lanczos,
    lanczos_eigs,
    lanczos_ground_state,
    temple_lanczos,
    two_sided_bounds,
)
from .lattice import effect_box, heat_step_box, laplacian_box, site_wire, state_box
from .lean_export import (
    LeanExporter,
    certificate_to_lean,
    diagram_to_lean_type,
    export_lean,
    gap_report_to_lean,
    structure_report_to_lean,
)
from .mera import MERA, MERALayer, random_mera
from .mpo import (
    MPO,
    MPOChiPoint,
    MPODMRGResult,
    MPOScalingReport,
    MultiStartDMRGResult,
    dmrg_multistart,
    dmrg_sweep_mpo,
    dmrg_sweep_mpo_2site,
    identity_mpo,
    mpo_apply_mps,
    mpo_chi_convergence,
    mpo_expectation,
    mpo_from_matrix,
    mpo_hermitian_conjugate,
    mpo_to_matrix,
    nn_hamiltonian_mpo,
    random_mpo,
)
from .mps import (
    MPS,
    mps_add,
    mps_apply_gate,
    mps_expectation,
    mps_from_state,
    mps_inner,
    mps_norm,
    mps_normalise,
    mps_to_state,
    mps_truncate,
    random_mps,
)
from .open_systems import (
    check_density_matrix,
    check_kraus_completeness,
    choi_matrix,
    density_matrix_from_pure,
    lindblad_step,
    lindblad_superoperator,
    partial_trace,
    steady_state,
)
from .os_axioms import (
    check_reflection_symmetry,
    check_transfer_positivity,
    finite_lattice_reflection_diagnostics,
    os_positivity_report,
    reflection_operator,
    transfer_matrix,
)
from .rayleigh_cert import (
    RayleighCertificate,
    rayleigh_certificate,
    verify_rayleigh_certificate,
)
from .qasm import (
    Gate,
    circuit_to_diagram,
    circuit_to_qasm,
    circuit_unitary,
    get_gate_matrix,
    qasm_to_circuit,
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
from .symmetric import (
    BlockSparseTensor,
    ChargedBasis,
    block_sparse_matmul,
    check_u1_invariance,
    number_basis,
    project_to_u1,
    spin_half_basis,
    u1_blocks,
)
from .tebd import (
    DMRGResult,
    TEBDResult,
    bose_hubbard_bonds,
    dmrg_sweep,
    dmrg_sweep_2site,
    heisenberg_bonds,
    nn_hamiltonian,
    tdvp_evolve,
    tebd_evolve,
    tebd_step,
    tfim_bonds,
    xx_bonds,
)
from .thermal import (
    ThermalResult,
    ThermalScanPoint,
    ThermalScanResult,
    purification_bonds,
    purified_initial_mps,
    thermal_expectation,
    thermal_scan,
    thermal_state,
)
from .topology import Box, Diagram, Id, Wire
from .variational import (
    energy_expectation,
    optimize_mera,
    transverse_ising_ham,
    variational_bound,
    xx_model_ham,
)
from .viz import diagram_to_dict, diagram_to_html, save_diagram_html
from .zx import (
    ZXGraph,
    ZXNodeType,
    ZXRewriteLog,
    bialgebra,
    clifford_simplify,
    color_change,
    hadamard_cancel,
    identity_removal,
    local_complement,
    phase_gadget_fuse,
    pi_copy,
    simplify,
    spider_fusion,
    zx_from_circuit,
    zx_to_matrix,
)

__version__ = "0.23.0"
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
    # inverse (§4-J)
    "ParametricHam", "InverseDesignResult", "LearningResult",
    "inverse_design", "hamiltonian_learning", "energy_gradient",
    # lean_export (§4-L)
    "LeanExporter", "certificate_to_lean", "gap_report_to_lean",
    "structure_report_to_lean", "diagram_to_lean_type", "export_lean",
    # symmetric (§4-G)
    "ChargedBasis", "spin_half_basis", "number_basis",
    "check_u1_invariance", "project_to_u1", "u1_blocks",
    "BlockSparseTensor", "block_sparse_matmul",
    # viz (§7)
    "diagram_to_dict", "diagram_to_html", "save_diagram_html",
    # zx (§4-E / §8-B)
    "ZXNodeType", "ZXGraph", "ZXRewriteLog",
    "zx_from_circuit", "spider_fusion", "identity_removal",
    "hadamard_cancel", "color_change", "pi_copy", "simplify", "zx_to_matrix",
    "bialgebra", "local_complement", "phase_gadget_fuse", "clifford_simplify",
    # lanczos (§4-I)
    "lanczos", "lanczos_eigs", "lanczos_ground_state",
    "temple_lanczos", "two_sided_bounds", "TwoSidedBounds",
    # qasm (§4-F)
    "Gate", "get_gate_matrix",
    "circuit_to_qasm", "qasm_to_circuit",
    "circuit_unitary", "circuit_to_diagram",
    # open_systems (§4-H)
    "density_matrix_from_pure", "partial_trace",
    "check_density_matrix",
    "choi_matrix", "check_kraus_completeness",
    "lindblad_superoperator", "lindblad_step", "steady_state",
    # os_axioms (Phase 4)
    "transfer_matrix", "reflection_operator",
    "check_transfer_positivity", "check_reflection_symmetry",
    "finite_lattice_reflection_diagnostics",
    "os_positivity_report",  # deprecated alias, kept for backwards compat
    # rayleigh_cert — first closed-loop certificate
    "RayleighCertificate", "rayleigh_certificate", "verify_rayleigh_certificate",
    # scaling (Phase 4)
    "ChiPoint", "ScalingReport", "chi_convergence_study",
    # benchmark (§4-K)
    "BenchmarkResult", "BenchmarkReport", "run_benchmark",
    # difficulty (Phase 4)
    "entanglement_entropy", "entanglement_spectrum",
    "bipartite_entanglement_profile",
    "DifficultyReport", "difficulty_report",
    # mps (§8-A)
    "MPS", "mps_from_state", "mps_to_state", "mps_inner", "mps_norm",
    "mps_normalise", "mps_add", "mps_truncate", "mps_apply_gate",
    "mps_expectation", "random_mps",
    # tebd (§8-A / §9-A / §9-B / §9-C)
    "TEBDResult", "DMRGResult",
    "tebd_step", "tebd_evolve", "tdvp_evolve",
    "dmrg_sweep", "dmrg_sweep_2site",
    "nn_hamiltonian", "tfim_bonds", "xx_bonds",
    "heisenberg_bonds", "bose_hubbard_bonds",
    # thermal (§9-D / §9-J)
    "ThermalResult", "ThermalScanPoint", "ThermalScanResult",
    "purified_initial_mps", "purification_bonds",
    "thermal_state", "thermal_expectation", "thermal_scan",
    # mpo (§9-E / §9-F / §9-G / §9-H / §9-I)
    "MPO", "MPODMRGResult", "MPOChiPoint", "MPOScalingReport",
    "MultiStartDMRGResult",
    "identity_mpo", "random_mpo", "mpo_from_matrix",
    "mpo_to_matrix", "nn_hamiltonian_mpo",
    "mpo_apply_mps", "mpo_expectation", "mpo_hermitian_conjugate",
    "dmrg_sweep_mpo", "dmrg_sweep_mpo_2site",
    "dmrg_multistart", "mpo_chi_convergence",
]
