"""HTF Claim Registry — single source of truth for all claim descriptions.

C6 gate: CLI, MCP, and schema descriptions all import from here so that
claim wording, assurance levels, and limitation disclaimers stay in sync
across every public surface.

Usage::

    from htf.claim_registry import CLAIM_REGISTRY, get_claim

    info = get_claim("rayleigh")
    print(info.mcp_description)   # used in mcp_server.py tool descriptions
    print(info.cli_help)          # used in cli.py sub-parser help strings
"""
from __future__ import annotations

from dataclasses import dataclass

_FP_ONLY = (
    "All certified bounds cover floating-point rounding only; "
    "bond-dimension truncation and continuum-limit effects are [OUT]."
)

_FINITE_LATTICE = (
    "Finite-lattice only; continuum gap and χ-truncation bias are [OUT]."
)


@dataclass(frozen=True)
class ClaimInfo:
    """Structured description of one HTF claim type."""

    claim_id: str
    title: str
    assurance: str        # "rigorous" | "heuristic" | "reproducible"
    evidence_tier: str    # "[engineering]" | "[research]" | "[heuristic]"
    mcp_description: str  # full description for MCP @server.tool()
    cli_help: str         # one-line help for argparse sub-parser
    limitations: str      # machine-readable limitation disclaimer


CLAIM_REGISTRY: dict[str, ClaimInfo] = {
    "rayleigh": ClaimInfo(
        claim_id="rayleigh",
        title="Rayleigh-Ritz upper bound on ground-state energy",
        assurance="rigorous",
        evidence_tier="[engineering]",
        mcp_description=(
            "Certified Rayleigh-Ritz upper bound E0 ≤ upper. "
            "Uses python-flint Arb/Acb interval arithmetic at 128-bit precision; "
            "the certificate is independently verifiable via htf-verify. "
            + _FP_ONLY
        ),
        cli_help="Compute a certified Rayleigh-Ritz upper bound on E0 (requires python-flint).",
        limitations=_FP_ONLY,
    ),
    "variational": ClaimInfo(
        claim_id="variational",
        title="Variational ground-state energy upper bound (MERA)",
        assurance="rigorous",
        evidence_tier="[engineering]",
        mcp_description=(
            "Certified variational ground-state energy upper bound via MERA optimisation. "
            "Returns a Certificate with result, error_bound (FP rounding), and mode='certified'. "
            + _FP_ONLY
        ),
        cli_help="Certified variational ground-state energy upper bound via MERA.",
        limitations=_FP_ONLY,
    ),
    "gap": ClaimInfo(
        claim_id="gap",
        title="Spectral gap diagnostics (heuristic)",
        assurance="heuristic",
        evidence_tier="[heuristic]",
        mcp_description=(
            "Spectral gap diagnostics for a finite-lattice model. "
            "Returns: exact gap (full diagonalisation), variational E0/E1 upper bounds, "
            "heuristic gap estimate (E1_var - E0_var, NOT a certified gap upper bound), "
            "Temple heuristic lower estimate (NOT a rigorous lower bound unless "
            "temple_condition_met=True and E1_lower is a true lower bound on E_1). "
            + _FINITE_LATTICE + " "
            "Assurance fields indicate the reliability level of each quantity."
        ),
        cli_help="Spectral gap diagnostics (heuristic; not a certified gap bound).",
        limitations=(
            "gap_var and trial_energy_diff are heuristic estimates (E1_var - E0_var), "
            "NOT certified gap upper bounds. Temple value is heuristic. "
            + _FINITE_LATTICE
        ),
    ),
    "lanczos": ClaimInfo(
        claim_id="lanczos",
        title="Lanczos two-sided spectral estimates",
        assurance="heuristic",
        evidence_tier="[research]",
        mcp_description=(
            "Lanczos two-sided spectral estimates on the ground-state energy E_0. "
            "E0_upper is a rigorous variational upper bound (Ritz value). "
            "E0_lower_heuristic is a Temple estimate — rigorous finite-lattice lower bound "
            "ONLY when temple_condition_met=True AND the E_1 input was a true lower bound; "
            "the current implementation uses a Ritz upper bound, so treat as heuristic. "
            + _FINITE_LATTICE
        ),
        cli_help="Lanczos ground-state bounds (Ritz upper + Temple heuristic lower).",
        limitations=(
            "Temple lower value is heuristic (Ritz upper bound used as E_1 input). "
            + _FINITE_LATTICE
        ),
    ),
    "benchmark": ClaimInfo(
        claim_id="benchmark",
        title="Certified reproducibility benchmark",
        assurance="reproducible",
        evidence_tier="[engineering]",
        mcp_description=(
            "Run the certified reproducibility benchmark suite: variational energy, "
            "gap bounds, OS-positivity diagnostics, and difficulty map for standard models. "
            "Output is a full JSON BenchmarkReport. "
            "Reproducible: same inputs produce identical outputs for a given HTF version."
        ),
        cli_help="Run the certified reproducibility benchmark suite.",
        limitations=_FP_ONLY,
    ),
    "difficulty": ClaimInfo(
        claim_id="difficulty",
        title="Entanglement / difficulty diagnostics",
        assurance="heuristic",
        evidence_tier="[research]",
        mcp_description=(
            "Entanglement entropy, entanglement spectrum, and bipartite entanglement "
            "profile for a model. Measures how hard the ground state is to represent "
            "as a low-bond-dimension MPS/MERA. Results are heuristic diagnostics, "
            "not certified bounds."
        ),
        cli_help="Compute entanglement / difficulty diagnostics (heuristic).",
        limitations="Entanglement measures are heuristic; not a certified bound.",
    ),
}


def get_claim(claim_id: str) -> ClaimInfo:
    """Return the ClaimInfo for *claim_id*, raising KeyError if unknown."""
    try:
        return CLAIM_REGISTRY[claim_id]
    except KeyError:
        known = sorted(CLAIM_REGISTRY)
        raise KeyError(
            f"Unknown claim_id {claim_id!r}. Known: {known}"
        ) from None


def registry_summary() -> dict:
    """Return a JSON-serialisable summary of all registered claims."""
    return {
        cid: {
            "title": info.title,
            "assurance": info.assurance,
            "evidence_tier": info.evidence_tier,
            "limitations": info.limitations,
        }
        for cid, info in CLAIM_REGISTRY.items()
    }
