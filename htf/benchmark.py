"""HTF §4-K — Certified reproducibility benchmark suite.

Runs the full Phase 4 analysis pipeline on a set of standard test
Hamiltonians and produces a comprehensive JSON report with all certified
results.  Every result is bit-for-bit reproducible given the same
``n_sites``, ``chi``, ``n_iter``, ``seed``, and HTF version.

Honest scope
------------
* ``E0_error_bound`` and ``gap_cert_error`` cover floating-point rounding only.
* Bond-dimension truncation bias, finite-size effects are ``[OUT]``.
* OS-positivity is a finite-lattice necessary condition; continuum is ``[OUT]``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class BenchmarkResult:
    """Certified outputs for one model run."""
    model: str
    n_sites: int
    chi: int
    seed: int
    E0_var: float           # variational energy (certified midpoint)
    E0_error_bound: float   # FP rounding bound on E0_var
    gap_exact: float        # exact gap from full diagonalisation
    gap_var: float          # variational gap E1_var - E0_var (upper bound)
    gap_cert_result: float  # certified gap midpoint
    gap_cert_error: float   # FP rounding bound on gap_cert
    temple_lb: float        # Temple lower bound on E0 (valid if E0_var < E1_exact)
    temple_condition_met: bool
    os_passed: bool         # True iff all three OS-positivity checks pass
    max_entropy: float      # max bipartite entanglement entropy [nats]
    likely_area_law: bool   # heuristic area-law classification
    n_iter_used: int        # actual L-BFGS-B iterations taken
    elapsed_s: float        # wall-clock time for this model

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkReport:
    """Aggregated benchmark report across all tested models."""
    htf_version: str
    n_sites: int
    chi: int
    n_iter: int
    seed: int
    results: list[BenchmarkResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "htf_version": self.htf_version,
            "n_sites": self.n_sites,
            "chi": self.chi,
            "n_iter": self.n_iter,
            "seed": self.seed,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        hdr = (
            f"HTF Benchmark Report — v{self.htf_version}\n"
            f"  n_sites={self.n_sites}  chi={self.chi}  "
            f"n_iter={self.n_iter}  seed={self.seed}\n"
            f"  {'Model':<20} {'E0_var':>12} {'gap_exact':>10} "
            f"{'gap_cert':>10} {'OS':>4} {'S_max':>7} {'area-law':>8}"
        )
        rows = []
        for r in self.results:
            rows.append(
                f"  {r.model:<20} {r.E0_var:>12.6f} {r.gap_exact:>10.6f} "
                f"{r.gap_cert_result:>10.6f} {'✓' if r.os_passed else '✗':>4} "
                f"{r.max_entropy:>7.4f} {'yes' if r.likely_area_law else 'no':>8}"
            )
        return "\n".join([hdr] + rows)


_HAM_BUILDERS = {
    "ising": lambda n: None,          # filled in at call time to avoid import at module level
    "ising_critical": lambda n: None,
    "xx": lambda n: None,
}


def _available_models() -> list[str]:
    return ["ising", "ising_critical", "xx"]


def run_benchmark(
    n_sites: int = 4,
    chi: int = 2,
    n_iter: int = 50,
    seed: int = 0,
    models: list[str] | None = None,
) -> BenchmarkReport:
    """Run the HTF certified reproducibility benchmark suite.

    For each model in *models* (default: ``["ising", "xx"]``), runs:

    * MERA variational optimisation → certified ``E_0`` upper bound.
    * Full gap report (exact, variational, Temple lb, certified Arb).
    * OS-positivity three-check verification.
    * Entanglement entropy / difficulty map.

    Parameters
    ----------
    n_sites : number of sites (power of 2; default 4).
    chi     : MERA bond dimension (default 2).
    n_iter  : L-BFGS-B iterations (default 50).
    seed    : RNG seed for deterministic reproducibility (default 0).
    models  : model names to benchmark; default ``["ising", "xx"]``.
              Available: ``"ising"``, ``"ising_critical"``, ``"xx"``.

    Returns
    -------
    :class:`BenchmarkReport`
    """
    import time

    from . import __version__
    from .difficulty import difficulty_report
    from .gap import gap_report
    from .mera import random_mera
    from .os_axioms import os_positivity_report
    from .variational import (
        optimize_mera,
        transverse_ising_ham,
        variational_bound,
        xx_model_ham,
    )

    if models is None:
        models = ["ising", "xx"]

    ham_fns = {
        "ising":          lambda n: transverse_ising_ham(n, J=1.0, h=0.5),
        "ising_critical": lambda n: transverse_ising_ham(n, J=1.0, h=1.0),
        "xx":             lambda n: xx_model_ham(n, J=1.0),
    }
    valid = set(ham_fns)
    for m in models:
        if m not in valid:
            raise ValueError(f"Unknown model {m!r}; choose from {sorted(valid)}")

    report = BenchmarkReport(
        htf_version=__version__,
        n_sites=n_sites,
        chi=chi,
        n_iter=n_iter,
        seed=seed,
    )

    for model_name in models:
        t0 = time.perf_counter()
        H = ham_fns[model_name](n_sites)
        evals = np.linalg.eigvalsh(H)

        mera0 = random_mera(n_sites, chi=chi, seed=seed)
        mera_gs, history = optimize_mera(H, mera0, n_iter=n_iter, tol=1e-6)
        cert = variational_bound(H, mera_gs)
        psi_gs = mera_gs.state_vector()

        psi_es_raw = random_mera(n_sites, chi=chi, seed=seed + 1).state_vector()
        g = gap_report(H, psi_gs, psi_es_raw)

        os_rep = os_positivity_report(H, n_sites, beta=1.0, d=2)
        drep = difficulty_report(H, n_sites, n_iter=n_iter, seed=seed)

        report.results.append(BenchmarkResult(
            model=model_name,
            n_sites=n_sites,
            chi=chi,
            seed=seed,
            E0_var=cert.result,
            E0_error_bound=float(cert.error_bound),
            gap_exact=g["gap_exact"],
            gap_var=g["gap_var"],
            gap_cert_result=g["gap_cert"].result,
            gap_cert_error=float(g["gap_cert"].error_bound),
            temple_lb=g["temple_lb"],
            temple_condition_met=bool(g["E0_var"] < evals[1]),
            os_passed=os_rep["all_passed"],
            max_entropy=drep.max_entropy,
            likely_area_law=drep.likely_area_law,
            n_iter_used=len(history),
            elapsed_s=time.perf_counter() - t0,
        ))

    return report
