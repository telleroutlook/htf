"""Agent-drivable command-line interface for HTF.

Designed so an LLM agent (or a human) can operate HTF from structured commands
with machine-readable JSON output. Subcommands are verbs; every result is JSON
so an agent can parse it and relay the certified bounds faithfully in natural
language — the certificate constrains the agent and prevents overstatement.

Available subcommands
---------------------
version     Print the version.
hello       Phase-1 hello-world diagram.
gap         Spectral gap bounds (exact + variational + certified + Temple).
variational Certified variational energy upper bound.
difficulty  Entanglement entropy / difficulty map.
os-check    Osterwalder-Schrader positivity machine check.
"""
from __future__ import annotations

import argparse
import json

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
        "temple_lb": report["temple_lb"],
        "temple_condition_met": bool(temple_cond_met),
        "gap_cert": _cert_to_dict(report["gap_cert"]),
        "notes": (
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
    """Osterwalder-Schrader positivity machine check via transfer matrix."""
    from .os_axioms import os_positivity_report

    H, model_label = _build_ham(args)
    report = os_positivity_report(H, args.n, beta=args.beta, d=2)
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


# ─────────────────────── main ────────────────────────────────────────────


def main(argv=None) -> None:
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

    # ── variational ──────────────────────────────────────────────────────
    sp_var = sub.add_parser(
        "variational",
        help="certified variational ground-state energy upper bound",
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
        help="spectral gap bounds (exact, variational, Temple, certified)",
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
        help="entanglement entropy profile and computational difficulty map",
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

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
