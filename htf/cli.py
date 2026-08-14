"""Agent-drivable command-line interface for HTF.

Designed so an LLM agent (or a human) can operate HTF from structured commands
with machine-readable JSON output. Subcommands are verbs; every result is JSON
so an agent can parse it and relay the certified bounds faithfully in natural
language — the certificate constrains the agent and prevents overstatement.

Available subcommands
---------------------
version      Print the version.
hello        Phase-1 hello-world diagram.
gap          Spectral gap bounds (exact + variational + certified + Temple).
variational  Certified variational energy upper bound.
difficulty   Entanglement entropy / difficulty map.
os-check     Osterwalder-Schrader positivity machine check.
benchmark    Certified reproducibility benchmark suite.
lanczos      Lanczos bounds: heuristic Temple lower + Ritz upper [heuristic].
qasm-sim     Simulate a QASM 2.0 circuit file → unitary matrix JSON.
zx-simplify  Load a QASM circuit, convert to ZX, simplify, report stats.
inverse      Inverse / Hamiltonian-learning design.
lean-export  Generate a Lean 4 proof-skeleton for a Hamiltonian model.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from . import __version__
from .certificate import Certificate
from .engine import contract
from .functor import TensorFunctor
from .topology import Box, Wire

# ─────────────────────── helpers ─────────────────────────────────────────


def _hello_diagram():
    """The Phase-1 hello-world: <phi| U |psi> for a swap gate U on one qubit."""
    s = Wire("spin", 2)
    psi = Box("psi", (), (s,))
    U = Box("U", (s,), (s,))
    phi = Box("phi", (s,), ())
    diagram = psi >> U >> phi
    F = TensorFunctor(
        {
            "psi": np.array([1.0, 0.0]),
            "U": np.array([[0.0, 1.0], [1.0, 0.0]]),
            "phi": np.array([0.0, 1.0]),
        }
    )
    return diagram, F


def _add_model_args(sp: argparse.ArgumentParser) -> None:
    """Attach shared model / size arguments to subparser *sp*."""
    sp.add_argument(
        "--model", choices=["ising", "xx"], default="ising",
        help="Hamiltonian model: 'ising' (TFIM) or 'xx' (default: ising)",
    )
    sp.add_argument("--n", type=int, default=4, metavar="N",
                    help="number of sites (default: 4)")
    sp.add_argument("--J", type=float, default=1.0,
                    help="nearest-neighbour coupling J (default: 1.0)")
    sp.add_argument("--h", type=float, default=0.5,
                    help="transverse field h — ising model only (default: 0.5)")


def _build_ham(args) -> tuple[np.ndarray, str]:
    """Build Hamiltonian from parsed CLI args; return (H, label)."""
    from .variational import transverse_ising_ham, xx_model_ham
    if args.model == "xx":
        H = xx_model_ham(args.n, J=args.J)
        label = f"xx(n={args.n}, J={args.J})"
    else:
        H = transverse_ising_ham(args.n, J=args.J, h=args.h)
        label = f"ising(n={args.n}, J={args.J}, h={args.h})"
    return H, label


def _cert_to_dict(cert: Certificate) -> dict:
    return {
        "result": cert.result,
        "error_bound": cert.error_bound,
        "mode": cert.mode,
        "notes": cert.notes,
    }


def _report_to_dict(r) -> dict:
    return {
        "property": r.property_name,
        "passed": r.passed,
        "defect": r.defect,
        "tolerance": r.tolerance,
        "notes": r.notes,
    }


# ─────────────────────── subcommands ─────────────────────────────────────


def cmd_version(_args) -> None:
    print(json.dumps({"htf_version": __version__}))


def cmd_registry(_args) -> None:
    """Print the claim registry as JSON (C6: single source of truth)."""
    from .claim_registry import registry_summary
    print(json.dumps(registry_summary(), indent=2))


def cmd_hello(_args) -> None:
    diagram, F = _hello_diagram()
    val = contract(diagram, F, mode="float")
    cert = Certificate(
        result=float(val),
        mode="float",
        error_bound=None,
        notes="<phi|U|psi> for a swap gate; float / discovery-tier, no error bound",
    )
    print(cert.to_json(indent=2))


def cmd_variational(args) -> None:
    """Certified variational energy upper bound for the ground state."""
    from .mera import random_mera
    from .variational import optimize_mera, variational_bound

    H, model_label = _build_ham(args)
    mera0 = random_mera(args.n, chi=args.chi, seed=args.seed)
    mera_opt, history = optimize_mera(H, mera0, n_iter=args.n_iter, tol=1e-6)
    cert = variational_bound(H, mera_opt)
    out = {
        "model": model_label,
        "n_sites": args.n,
        "chi": args.chi,
        "n_iter_used": len(history),
        "certificate": _cert_to_dict(cert),
        "notes": "certified upper bound on E_0; certifies FP rounding only",
    }
    print(json.dumps(out, indent=2))


def cmd_gap(args) -> None:
    """Spectral gap bounds: exact, variational, Temple lower bound, certified."""
    from .gap import gap_report
    from .mera import random_mera
    from .variational import optimize_mera

    H, model_label = _build_ham(args)
    mera0 = random_mera(args.n, chi=args.chi, seed=args.seed)
    mera_gs, _ = optimize_mera(H, mera0, n_iter=args.n_iter, tol=1e-6)
    psi_gs = mera_gs.state_vector()
    psi_es_raw = random_mera(args.n, chi=args.chi, seed=args.seed + 1).state_vector()

    report = gap_report(H, psi_gs, psi_es_raw)
    evals = np.linalg.eigvalsh(H)
    temple_cond_met = report["E0_var"] < evals[1]

    out = {
        "model": model_label,
        "n_sites": args.n,
        "chi": args.chi,
        "gap_exact": report["gap_exact"],
        "E0_var": report["E0_var"],
        "E1_var": report["E1_var"],
        "gap_var": report["gap_var"],
        "temple_heuristic": report["temple_lb"],
        "temple_assurance": "heuristic",
        "temple_condition_met": bool(temple_cond_met),
        "trial_energy_diff": _cert_to_dict(report["gap_cert"]),
        "trial_energy_diff_assurance": "heuristic",
        "notes": (
            "gap_var and trial_energy_diff are heuristic estimates (NOT certified gap bounds); "
            "certified bounds cover FP rounding only; "
            "bond-dimension and finite-size bias are [OUT]"
        ),
    }
    print(json.dumps(out, indent=2))


def cmd_difficulty(args) -> None:
    """Entanglement entropy profile and computational difficulty map."""
    from .difficulty import difficulty_report

    H, model_label = _build_ham(args)
    rep = difficulty_report(H, args.n, n_iter=args.n_iter, seed=args.seed)
    out = {
        "model": model_label,
        "n_sites": rep.n_sites,
        "chi_used": rep.chi_used,
        "energy": rep.energy,
        "entanglement_profile": rep.entanglement_profile.tolist(),
        "max_entropy": rep.max_entropy,
        "area_law_limit": rep.area_law_limit,
        "likely_area_law": rep.likely_area_law,
        "notes": rep.notes,
    }
    print(json.dumps(out, indent=2))


def cmd_os_check(args) -> None:
    """Finite-lattice reflection diagnostics via transfer matrix."""
    from .os_axioms import finite_lattice_reflection_diagnostics

    H, model_label = _build_ham(args)
    report = finite_lattice_reflection_diagnostics(H, args.n, beta=args.beta, d=2)
    out = {
        "model": model_label,
        "n_sites": args.n,
        "beta": args.beta,
        "d": 2,
        "transfer_positivity": _report_to_dict(report["transfer_positivity"]),
        "reflection_symmetry": _report_to_dict(report["reflection_symmetry"]),
        "os_gram_positivity": _report_to_dict(report["os_gram_positivity"]),
        "all_passed": report["all_passed"],
        "notes": report["notes"],
    }
    print(json.dumps(out, indent=2))


def cmd_benchmark(args) -> None:
    """Certified reproducibility benchmark suite across standard models."""
    from .benchmark import run_benchmark

    rep = run_benchmark(
        n_sites=args.n,
        chi=args.chi,
        n_iter=args.n_iter,
        seed=args.seed,
        models=args.models if args.models else None,
    )
    print(rep.to_json(indent=2))


def cmd_lanczos(args) -> None:
    """Lanczos bounds: heuristic Temple lower + Ritz upper (NOT strict two-sided)."""
    from .lanczos import temple_lanczos

    H, model_label = _build_ham(args)
    bounds = temple_lanczos(H, k=args.k, seed=args.seed)
    out = {
        "model": model_label,
        "n_sites": args.n,
        "k_lanczos": bounds.k_lanczos,
        "E0_upper": bounds.E0_upper,
        "E0_upper_error": bounds.E0_upper_error,
        "E0_lower_heuristic": bounds.E0_lower,
        "E0_lower_assurance": "heuristic",
        "E1_ritz": bounds.E1_ritz,
        "interval_heuristic_width": bounds.heuristic_width if math.isfinite(bounds.heuristic_width) else None,
        "temple_condition_met": bounds.temple_condition_met,
        "notes": bounds.notes,
    }
    print(json.dumps(out, indent=2))


def cmd_qasm_sim(args) -> None:
    """Simulate a QASM 2.0 circuit file and output the unitary matrix."""
    from pathlib import Path

    from .qasm import circuit_unitary, qasm_to_circuit

    src = Path(args.file).read_text(encoding="utf-8")
    gates = qasm_to_circuit(src)
    n = args.n_qubits if args.n_qubits else (
        max((max(g.qubits) for g in gates if g.qubits), default=0) + 1
    )
    U = circuit_unitary(gates, n)
    out = {
        "file": args.file,
        "n_qubits": n,
        "n_gates": len(gates),
        "unitary_real": U.real.tolist(),
        "unitary_imag": U.imag.tolist(),
        "notes": "dense unitary simulation; float mode",
    }
    print(json.dumps(out, indent=2))


def cmd_zx_simplify(args) -> None:
    """Convert a QASM circuit to ZX, simplify, and report rewrite statistics."""
    from pathlib import Path

    from .qasm import qasm_to_circuit
    from .zx import ZXRewriteLog, simplify, zx_from_circuit

    src = Path(args.file).read_text(encoding="utf-8")
    gates = qasm_to_circuit(src)
    n = args.n_qubits if args.n_qubits else (
        max((max(g.qubits) for g in gates if g.qubits), default=0) + 1
    )
    g = zx_from_circuit(gates, n)
    n_before = len(g.nodes)
    log = ZXRewriteLog()
    total = simplify(g, log=log)
    n_after = len(g.nodes)
    rule_counts: dict[str, int] = {}
    for step in log.steps:
        rule_counts[step["rule"]] = rule_counts.get(step["rule"], 0) + 1
    out = {
        "file": args.file,
        "n_qubits": n,
        "n_gates_in": len(gates),
        "nodes_before": n_before,
        "nodes_after": n_after,
        "rewrites_total": total,
        "rule_counts": rule_counts,
        "notes": "ZX simplification; float / discovery-tier",
    }
    print(json.dumps(out, indent=2))


def cmd_inverse(args) -> None:
    """Inverse / Hamiltonian-learning design."""
    from .inverse import inverse_design

    result = inverse_design(
        target_e0=args.target_e0,
        model=args.model,
        n_sites=args.n,
        n_restarts=args.n_restarts,
        seed=args.seed,
    )
    out = {
        "model": args.model,
        "n_sites": args.n,
        "target_e0": args.target_e0,
        "E0_achieved": float(result.E0_achieved),
        "residual": float(result.residual),
        "params_opt": result.params_opt.tolist(),
        "param_names": result.param_names,
        "converged": bool(result.converged),
        "n_restarts": int(result.n_restarts),
        "notes": result.notes,
    }
    print(json.dumps(out, indent=2))


def cmd_lean_export(args) -> None:
    """Generate a Lean 4 proof-skeleton for a Hamiltonian model."""
    from .gap import gap_report
    from .lean_export import LeanExporter
    from .mera import random_mera
    from .variational import optimize_mera, variational_bound

    H, model_label = _build_ham(args)
    mera0 = random_mera(args.n, chi=2, seed=0)
    mera_opt, _ = optimize_mera(H, mera0, n_iter=30, tol=1e-6)
    cert = variational_bound(H, mera_opt)
    psi_gs = mera_opt.state_vector()
    psi_es = random_mera(args.n, chi=2, seed=1).state_vector()
    greport = gap_report(H, psi_gs, psi_es)

    exp = LeanExporter(preamble=f"HTF model: {model_label}")
    exp.add_certificate(cert, "variational_E0")
    exp.add_gap_report(greport, "spectral_gap")
    out_path = args.output or f"htf_{args.model}_n{args.n}.lean"
    exp.write(out_path)
    print(json.dumps({
        "output": out_path,
        "model": model_label,
        "n_sites": args.n,
        "E0_upper": cert.result,
        "gap_exact": greport["gap_exact"],
        "notes": "Lean 4 skeleton; every sorry is a proof obligation [研究]",
    }, indent=2))


def cmd_rayleigh(args) -> None:
    """Validated Rayleigh Certificate: certifies E0 ≤ upper for a trial state."""
    from .rayleigh_cert import rayleigh_certificate, verify_rayleigh_certificate

    H, model_label = _build_ham(args)
    rng = np.random.default_rng(args.seed)
    psi = np.asarray(rng.standard_normal(H.shape[0]))
    psi /= np.linalg.norm(psi)

    cert = rayleigh_certificate(H, psi, notes=f"model={model_label}, seed={args.seed}")
    verify_rayleigh_certificate(cert)

    if getattr(args, "full", False):
        print(cert.to_full_json(indent=2))
    else:
        print(json.dumps({
            "model": model_label,
            "n_sites": args.n,
            "claim": cert.claim,
            "theorem": cert.theorem,
            "assumptions": cert.assumptions,
            "interval": {"lower": cert.lower, "upper": cert.upper, "radius": cert.radius},
            "input_digest": cert.input_digest,
            "backend": cert.backend,
            "htf_version": cert.htf_version,
            "verified": cert.verified,
            "notes": cert.notes,
        }, indent=2))


# ─────────────────────── main ────────────────────────────────────────────


def main(argv=None) -> None:
    from .claim_registry import CLAIM_REGISTRY
    p = argparse.ArgumentParser(
        prog="htf",
        description=(
            "HTF — a certified, type-safe string-diagram / tensor-network framework."
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("version", help="print version as JSON").set_defaults(
        func=cmd_version
    )
    sub.add_parser(
        "hello", help="run the Phase-1 hello-world diagram and print a certificate"
    ).set_defaults(func=cmd_hello)
    sub.add_parser(
        "registry",
        help="print the claim registry (assurance levels and limitations) as JSON",
    ).set_defaults(func=cmd_registry)

    # ── variational ──────────────────────────────────────────────────────
    sp_var = sub.add_parser(
        "variational",
        help=CLAIM_REGISTRY["variational"].cli_help,
    )
    _add_model_args(sp_var)
    sp_var.add_argument("--chi", type=int, default=2, help="MERA bond dimension (default: 2)")
    sp_var.add_argument("--n-iter", type=int, default=50, dest="n_iter",
                        help="L-BFGS-B iterations (default: 50)")
    sp_var.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    sp_var.set_defaults(func=cmd_variational)

    # ── gap ──────────────────────────────────────────────────────────────
    sp_gap = sub.add_parser(
        "gap",
        help=CLAIM_REGISTRY["gap"].cli_help,
    )
    _add_model_args(sp_gap)
    sp_gap.add_argument("--chi", type=int, default=2, help="MERA bond dimension (default: 2)")
    sp_gap.add_argument("--n-iter", type=int, default=50, dest="n_iter",
                        help="L-BFGS-B iterations (default: 50)")
    sp_gap.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    sp_gap.set_defaults(func=cmd_gap)

    # ── difficulty ───────────────────────────────────────────────────────
    sp_dif = sub.add_parser(
        "difficulty",
        help=CLAIM_REGISTRY["difficulty"].cli_help,
    )
    _add_model_args(sp_dif)
    sp_dif.add_argument("--n-iter", type=int, default=50, dest="n_iter",
                        help="L-BFGS-B iterations (default: 50)")
    sp_dif.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    sp_dif.set_defaults(func=cmd_difficulty)

    # ── os-check ─────────────────────────────────────────────────────────
    sp_os = sub.add_parser(
        "os-check",
        help="Osterwalder-Schrader positivity machine check",
    )
    _add_model_args(sp_os)
    sp_os.add_argument(
        "--beta", type=float, default=1.0,
        help="imaginary-time step β for T = exp(−βH) (default: 1.0)",
    )
    sp_os.set_defaults(func=cmd_os_check)

    # ── benchmark ────────────────────────────────────────────────────────
    sp_bm = sub.add_parser(
        "benchmark",
        help=CLAIM_REGISTRY["benchmark"].cli_help,
    )
    sp_bm.add_argument("--n", type=int, default=4, metavar="N",
                       help="number of sites (default: 4)")
    sp_bm.add_argument("--chi", type=int, default=2,
                       help="MERA bond dimension (default: 2)")
    sp_bm.add_argument("--n-iter", type=int, default=50, dest="n_iter",
                       help="L-BFGS-B iterations (default: 50)")
    sp_bm.add_argument("--seed", type=int, default=0,
                       help="RNG seed for reproducibility (default: 0)")
    sp_bm.add_argument(
        "--models", nargs="+",
        choices=["ising", "ising_critical", "xx"],
        default=None,
        metavar="MODEL",
        help="models to benchmark (default: ising xx)",
    )
    sp_bm.set_defaults(func=cmd_benchmark)

    # ── lanczos ──────────────────────────────────────────────────────────
    sp_lan = sub.add_parser(
        "lanczos",
        help="Lanczos bounds: heuristic Temple lower + Ritz upper [heuristic]",
    )
    _add_model_args(sp_lan)
    sp_lan.add_argument("--k", type=int, default=30,
                        help="number of Lanczos steps (default: 30)")
    sp_lan.add_argument("--seed", type=int, default=0,
                        help="RNG seed (default: 0)")
    sp_lan.set_defaults(func=cmd_lanczos)

    # ── qasm-sim ─────────────────────────────────────────────────────────
    sp_qs = sub.add_parser(
        "qasm-sim",
        help="simulate a QASM 2.0 circuit file → unitary matrix JSON",
    )
    sp_qs.add_argument("--file", required=True, metavar="FILE",
                       help="path to a QASM 2.0 source file")
    sp_qs.add_argument("--n-qubits", type=int, default=0, dest="n_qubits",
                       help="override qubit count (default: inferred from circuit)")
    sp_qs.set_defaults(func=cmd_qasm_sim)

    # ── zx-simplify ──────────────────────────────────────────────────────
    sp_zx = sub.add_parser(
        "zx-simplify",
        help="load QASM circuit, convert to ZX, simplify, report rewrite stats",
    )
    sp_zx.add_argument("--file", required=True, metavar="FILE",
                       help="path to a QASM 2.0 source file")
    sp_zx.add_argument("--n-qubits", type=int, default=0, dest="n_qubits",
                       help="override qubit count (default: inferred from circuit)")
    sp_zx.set_defaults(func=cmd_zx_simplify)

    # ── inverse ──────────────────────────────────────────────────────────
    sp_inv = sub.add_parser(
        "inverse",
        help="inverse / Hamiltonian-learning design",
    )
    _add_model_args(sp_inv)
    sp_inv.add_argument("--target-e0", type=float, default=-1.5, dest="target_e0",
                        help="target ground-state energy E0 (default: -1.5)")
    sp_inv.add_argument("--n-restarts", type=int, default=5, dest="n_restarts",
                        help="L-BFGS-B random restarts (default: 5)")
    sp_inv.add_argument("--seed", type=int, default=0,
                        help="RNG seed (default: 0)")
    sp_inv.set_defaults(func=cmd_inverse)

    # ── lean-export ───────────────────────────────────────────────────────
    sp_le = sub.add_parser(
        "lean-export",
        help="generate a Lean 4 proof-skeleton for a Hamiltonian model",
    )
    _add_model_args(sp_le)
    sp_le.add_argument("--output", default="", metavar="FILE",
                       help="output .lean file path (default: htf_<model>_n<N>.lean)")
    sp_le.set_defaults(func=cmd_lean_export)

    sp_ray = sub.add_parser(
        "rayleigh",
        help="produce and verify a Rayleigh Certificate (E0 ≤ upper)",
    )
    _add_model_args(sp_ray)
    sp_ray.add_argument("--seed", type=int, default=42,
                        help="RNG seed for random trial state (default: 42)")
    sp_ray.add_argument("--full", action="store_true",
                        help="emit full certificate JSON (includes canonical H/ψ for htf-verify)")
    sp_ray.set_defaults(func=cmd_rayleigh)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
