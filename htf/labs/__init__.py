"""HTF Labs — full experimental / research suite.

This module re-exports everything from the htf-lab modules so that users who
want the complete suite can do::

    import htf.labs as labs
    labs.MPS(...)

    from htf.labs import MPS, tebd_evolve, dmrg_sweep_mpo, ...

All symbols here are ``[工程]`` or ``[研究]`` tier.  They are NOT part of the
certified core (``htf-spec`` + ``htf-verify``).  Assurance level for all lab
outputs is ``"heuristic"`` unless a specific function explicitly returns a
:class:`~htf.rayleigh_cert.RayleighCertificate` with ``assurance="rigorous"``.

The individual submodules remain directly importable (e.g.
``from htf.mps import MPS``) — this namespace is a convenience re-export only.
"""
from __future__ import annotations

# ── benchmark ─────────────────────────────────────────────────────────────────
from ..benchmark import BenchmarkReport, BenchmarkResult, run_benchmark

# ── difficulty ────────────────────────────────────────────────────────────────
from ..difficulty import (
    DifficultyReport,
    bipartite_entanglement_profile,
    difficulty_report,
    entanglement_entropy,
    entanglement_spectrum,
)

# ── gap diagnostics ───────────────────────────────────────────────────────────
from ..gap import (
    certified_gap_upper,
    first_excited_upper,
    gap_report,
    h2_expectation,
    spectral_gap_exact,
    temple_lower_bound,
    trial_energy_difference,
)

# ── inverse design ────────────────────────────────────────────────────────────
from ..inverse import (
    InverseDesignResult,
    LearningResult,
    ParametricHam,
    energy_gradient,
    hamiltonian_learning,
    inverse_design,
)

# ── Lanczos ───────────────────────────────────────────────────────────────────
from ..lanczos import (
    TwoSidedBounds,
    lanczos,
    lanczos_eigs,
    lanczos_ground_state,
    temple_lanczos,
    two_sided_bounds,
)

# ── lattice operators ─────────────────────────────────────────────────────────
from ..lattice import effect_box, heat_step_box, laplacian_box, site_wire, state_box

# ── Lean 4 export ─────────────────────────────────────────────────────────────
from ..lean_export import (
    LeanExporter,
    certificate_to_lean,
    diagram_to_lean_type,
    export_lean,
    gap_report_to_lean,
    structure_report_to_lean,
)

# ── MERA ──────────────────────────────────────────────────────────────────────
from ..mera import MERA, MERALayer, random_mera

# ── MPO ───────────────────────────────────────────────────────────────────────
from ..mpo import (
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

# ── MPS ───────────────────────────────────────────────────────────────────────
from ..mps import (
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

# ── open systems / CPTP ───────────────────────────────────────────────────────
from ..open_systems import (
    check_density_matrix,
    check_kraus_completeness,
    choi_matrix,
    density_matrix_from_pure,
    lindblad_step,
    lindblad_superoperator,
    partial_trace,
    steady_state,
)

# ── OS-axiom diagnostics ──────────────────────────────────────────────────────
from ..os_axioms import (
    check_reflection_symmetry,
    check_transfer_positivity,
    finite_lattice_reflection_diagnostics,
    os_positivity_report,
    reflection_operator,
    transfer_matrix,
)

# ── quantum circuits / QASM ───────────────────────────────────────────────────
from ..qasm import (
    Gate,
    circuit_to_diagram,
    circuit_to_qasm,
    circuit_unitary,
    get_gate_matrix,
    qasm_to_circuit,
)

# ── scaling ───────────────────────────────────────────────────────────────────
from ..scaling import ChiPoint, ScalingReport, chi_convergence_study

# ── structure verification ────────────────────────────────────────────────────
from ..structure import (
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

# ── TEBD / DMRG ───────────────────────────────────────────────────────────────
from ..tebd import (
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

# ── thermal / finite-temperature ─────────────────────────────────────────────
from ..thermal import (
    ThermalResult,
    ThermalScanPoint,
    ThermalScanResult,
    purification_bonds,
    purified_initial_mps,
    thermal_expectation,
    thermal_scan,
    thermal_state,
)

# ── variational ───────────────────────────────────────────────────────────────
from ..variational import (
    energy_expectation,
    optimize_mera,
    transverse_ising_ham,
    variational_bound,
    xx_model_ham,
)

# ── visualisation ─────────────────────────────────────────────────────────────
from ..viz import diagram_to_dict, diagram_to_html, save_diagram_html

# ── ZX calculus ───────────────────────────────────────────────────────────────
from ..zx import (
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

# ── symmetric tensors ─────────────────────────────────────────────────────────
from ..symmetric import (
    BlockSparseTensor,
    ChargedBasis,
    block_sparse_matmul,
    check_u1_invariance,
    number_basis,
    project_to_u1,
    spin_half_basis,
    u1_blocks,
)
