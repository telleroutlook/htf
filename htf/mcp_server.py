"""HTF MCP server — exposes HTF operations as MCP tools.

Allows external agents (LLM orchestrators, Claude Code, etc.) to call
HTF's certified-physics computations directly over the MCP protocol
(stdio transport by default).

Usage
-----
Start the server (stdio):

    python -m htf.mcp_server

Or add to an MCP client config::

    {
      "mcpServers": {
        "htf": {
          "command": "python3",
          "args": ["-m", "htf.mcp_server"]
        }
      }
    }

Available tools
---------------
htf_version      — HTF version string.
htf_variational  — Certified variational E_0 upper bound.
htf_gap          — Spectral gap bounds (exact + variational + Temple + certified).
htf_os_check     — Osterwalder-Schrader positivity machine check.
htf_benchmark    — Full certified reproducibility benchmark suite.

Honest scope
------------
All certified bounds cover floating-point rounding only.
Bond-dimension bias, finite-size effects, and the continuum limit are [OUT].
"""
from __future__ import annotations

import asyncio
import json
import types

try:
    from mcp.server.mcpserver import MCPServer
    HAS_MCP = True
except ImportError:
    HAS_MCP = False


def _build_server() -> MCPServer:
    """Construct and configure the HTF MCP server."""
    if not HAS_MCP:
        raise ImportError("htf.mcp_server requires 'mcp'. Install with: pip install mcp")

    import numpy as np
    from mcp.server.mcpserver import MCPServer

    from . import __version__

    # ── Resource limits (prevent accidental O(chi^n) memory blowup) ───────
    _MAX_N_SITES   = 16   # state vector has chi^n elements
    _MAX_CHI       = 16   # combined with n_sites: 16^16 would blow up, checked together
    _MAX_STATE_DIM = 65536  # chi^n_sites ≤ this (2^16)
    _MAX_LANCZOS_K = 200
    _MAX_QUBITS    = 12

    def _check_n_chi(n: int, chi: int) -> None:
        if n < 1:
            raise ValueError(f"n_sites must be ≥ 1, got {n}")
        if n > _MAX_N_SITES:
            raise ValueError(f"n_sites={n} exceeds MCP limit {_MAX_N_SITES}")
        if chi < 1:
            raise ValueError(f"chi must be ≥ 1, got {chi}")
        if chi > _MAX_CHI:
            raise ValueError(f"chi={chi} exceeds MCP limit {_MAX_CHI}")
        state_dim = chi ** n
        if state_dim > _MAX_STATE_DIM:
            raise ValueError(
                f"chi^n_sites = {chi}^{n} = {state_dim} exceeds MCP state-dim limit "
                f"{_MAX_STATE_DIM}; reduce n_sites or chi"
            )

    server = MCPServer(
        name="htf",
        version=__version__,
        description=(
            "HTF — certified, type-safe string-diagram / tensor-network framework. "
            "All certified bounds cover FP rounding only; "
            "bond-dimension and continuum-limit effects are [OUT]."
        ),
    )

    # ── htf_version ──────────────────────────────────────────────────────
    @server.tool(description="Return the HTF version.")
    def htf_version() -> str:
        return json.dumps({"htf_version": __version__})

    # ── htf_variational ───────────────────────────────────────────────────
    @server.tool(
        description=(
            "Certified variational ground-state energy upper bound. "
            "Returns a JSON Certificate with result, error_bound (FP rounding), "
            "and mode='certified'."
        )
    )
    def htf_variational(
        model: str = "ising",
        n: int = 4,
        J: float = 1.0,
        h: float = 0.5,
        chi: int = 2,
        n_iter: int = 50,
        seed: int = 42,
    ) -> str:
        from .cli import _build_ham, _cert_to_dict
        from .mera import random_mera
        from .variational import optimize_mera, variational_bound

        _check_n_chi(n, chi)
        a = types.SimpleNamespace(model=model, n=n, J=J, h=h)
        H, model_label = _build_ham(a)
        mera0 = random_mera(n, chi=chi, seed=seed)
        mera_opt, history = optimize_mera(H, mera0, n_iter=n_iter, tol=1e-6)
        cert = variational_bound(H, mera_opt)
        out = {
            "model": model_label,
            "n_sites": n,
            "chi": chi,
            "n_iter_used": len(history),
            "certificate": _cert_to_dict(cert),
            "notes": "certified upper bound on E_0; certifies FP rounding only",
        }
        return json.dumps(out, indent=2)

    # ── htf_gap ───────────────────────────────────────────────────────────
    @server.tool(
        description=(
            "Spectral gap diagnostics for a finite-lattice model. "
            "Returns: exact gap (full diagonalisation), variational E0/E1 upper bounds, "
            "heuristic gap estimate (E1_var - E0_var, NOT a certified gap upper bound), "
            "Temple heuristic lower estimate (NOT a rigorous lower bound unless "
            "temple_condition_met=True and E1_lower is a true lower bound). "
            "All bounds are finite-lattice only; continuum gap and χ-truncation bias are [OUT]. "
            "Assurance fields indicate the reliability level of each quantity."
        )
    )
    def htf_gap(
        model: str = "ising",
        n: int = 4,
        J: float = 1.0,
        h: float = 0.5,
        chi: int = 2,
        n_iter: int = 50,
        seed: int = 42,
    ) -> str:
        from .cli import _build_ham, _cert_to_dict
        from .gap import gap_report
        from .mera import random_mera
        from .variational import optimize_mera

        _check_n_chi(n, chi)
        a = types.SimpleNamespace(model=model, n=n, J=J, h=h)
        H, model_label = _build_ham(a)
        mera0 = random_mera(n, chi=chi, seed=seed)
        mera_gs, _ = optimize_mera(H, mera0, n_iter=n_iter, tol=1e-6)
        psi_gs = mera_gs.state_vector()
        psi_es_raw = random_mera(n, chi=chi, seed=seed + 1).state_vector()
        report = gap_report(H, psi_gs, psi_es_raw)
        evals = np.linalg.eigvalsh(H)
        out = {
            "model": model_label,
            "n_sites": n,
            "chi": chi,
            "gap_exact": report["gap_exact"],
            "E0_var": report["E0_var"],
            "E1_var": report["E1_var"],
            "gap_var": report["gap_var"],
            "gap_var_assurance": "heuristic",
            "temple_heuristic": report["temple_lb"],
            "temple_assurance": "heuristic",
            "temple_condition_met": bool(report["E0_var"] < evals[1]),
            "trial_energy_diff": _cert_to_dict(report["gap_cert"]),
            "trial_energy_diff_assurance": "heuristic",
            "notes": (
                "gap_var and trial_energy_diff are heuristic estimates (E1_var-E0_var), "
                "NOT certified gap upper bounds (P0-2); "
                "certified bounds cover FP rounding only; "
                "bond-dimension and finite-size bias are [OUT]"
            ),
        }
        return json.dumps(out, indent=2)

    # ── htf_os_check ──────────────────────────────────────────────────────
    @server.tool(
        description=(
            "Osterwalder-Schrader positivity machine check. "
            "Three independent checks: transfer matrix T=exp(-βH) is PSD, "
            "[H,R]=0 (reflection symmetry), OS-Gram G=T+RTR is PSD. "
            "All checks are finite-lattice; continuum OS-positivity is [OUT]."
        )
    )
    def htf_os_check(
        model: str = "ising",
        n: int = 4,
        J: float = 1.0,
        h: float = 0.5,
        beta: float = 1.0,
    ) -> str:
        from .cli import _build_ham, _report_to_dict
        from .os_axioms import os_positivity_report

        if n > _MAX_N_SITES:
            raise ValueError(f"n_sites={n} exceeds MCP limit {_MAX_N_SITES}")
        a = types.SimpleNamespace(model=model, n=n, J=J, h=h)
        H, model_label = _build_ham(a)
        rep = os_positivity_report(H, n, beta=beta, d=2)
        out = {
            "model": model_label,
            "n_sites": n,
            "beta": beta,
            "d": 2,
            "transfer_positivity": _report_to_dict(rep["transfer_positivity"]),
            "reflection_symmetry": _report_to_dict(rep["reflection_symmetry"]),
            "os_gram_positivity": _report_to_dict(rep["os_gram_positivity"]),
            "all_passed": rep["all_passed"],
            "notes": rep["notes"],
        }
        return json.dumps(out, indent=2)

    # ── htf_benchmark ─────────────────────────────────────────────────────
    @server.tool(
        description=(
            "Run the certified reproducibility benchmark suite: variational energy, "
            "gap bounds, OS-positivity, and difficulty map for standard models. "
            "Output is a full JSON BenchmarkReport. "
            "Reproducible: same inputs → same outputs for a given HTF version."
        )
    )
    def htf_benchmark(
        n: int = 4,
        chi: int = 2,
        n_iter: int = 50,
        seed: int = 0,
        models: list[str] | None = None,
    ) -> str:
        from .benchmark import run_benchmark
        _check_n_chi(n, chi)
        rep = run_benchmark(
            n_sites=n, chi=chi, n_iter=n_iter, seed=seed, models=models
        )
        return rep.to_json(indent=2)

    # ── htf_lanczos ───────────────────────────────────────────────────────
    @server.tool(
        description=(
            "Lanczos two-sided spectral estimates on the ground-state energy E_0. "
            "Returns Ritz upper bound (E0_upper, variational) and Temple heuristic "
            "lower estimate (E0_lower_heuristic). "
            "The Temple value is a rigorous finite-lattice lower bound ONLY when "
            "temple_condition_met=True AND the E1 input was a true lower bound on E_1 "
            "(the current implementation passes a Ritz upper bound, so treat as heuristic). "
            "Continuum gap and χ-truncation bias are [OUT]."
        )
    )
    def htf_lanczos(
        model: str = "ising",
        n: int = 4,
        J: float = 1.0,
        h: float = 0.5,
        k: int = 30,
        seed: int = 0,
    ) -> str:
        from .cli import _build_ham
        from .lanczos import temple_lanczos

        if n > _MAX_N_SITES:
            raise ValueError(f"n_sites={n} exceeds MCP limit {_MAX_N_SITES}")
        if k > _MAX_LANCZOS_K:
            raise ValueError(f"Lanczos k={k} exceeds MCP limit {_MAX_LANCZOS_K}")
        a = types.SimpleNamespace(model=model, n=n, J=J, h=h)
        H, model_label = _build_ham(a)
        bounds = temple_lanczos(H, k=k, seed=seed)
        out = {
            "model": model_label,
            "n_sites": n,
            "k_lanczos": bounds.k_lanczos,
            "E0_upper": bounds.E0_upper,
            "E0_upper_error": bounds.E0_upper_error,
            "E0_lower_heuristic": bounds.E0_lower,
            "E0_lower_assurance": "heuristic",
            "E1_ritz": bounds.E1_ritz,
            "interval_width": bounds.width,
            "temple_condition_met": bounds.temple_condition_met,
            "notes": bounds.notes,
        }
        return json.dumps(out, indent=2)

    # ── htf_qasm_simulate ─────────────────────────────────────────────────
    @server.tool(
        description=(
            "Simulate a QASM 2.0 circuit (supplied as a source string) and return "
            "the full unitary matrix. Float mode; exponential in qubit count. "
            "Returns real and imaginary parts as nested lists."
        )
    )
    def htf_qasm_simulate(
        qasm_src: str,
        n_qubits: int = 0,
    ) -> str:
        from .qasm import circuit_unitary, qasm_to_circuit

        gates = qasm_to_circuit(qasm_src)
        n = n_qubits or (
            max((max(g.qubits) for g in gates if g.qubits), default=0) + 1
        )
        if n > _MAX_QUBITS:
            raise ValueError(f"n_qubits={n} exceeds MCP limit {_MAX_QUBITS}")
        U = circuit_unitary(gates, n)
        out = {
            "n_qubits": n,
            "n_gates": len(gates),
            "unitary_real": U.real.tolist(),
            "unitary_imag": U.imag.tolist(),
            "notes": "dense unitary simulation; float mode",
        }
        return json.dumps(out, indent=2)

    # ── htf_zx_simplify ───────────────────────────────────────────────────
    @server.tool(
        description=(
            "Convert a QASM 2.0 circuit (source string) to a ZX diagram, "
            "simplify with spider_fusion / identity_removal / hadamard_cancel / "
            "color_change / pi_copy, and return rewrite statistics. "
            "The rewrite rules are sound (preserve the linear map) [研究]."
        )
    )
    def htf_zx_simplify(
        qasm_src: str,
        n_qubits: int = 0,
        rules: list[str] | None = None,
    ) -> str:
        from .qasm import qasm_to_circuit
        from .zx import ZXRewriteLog, simplify, zx_from_circuit

        gates = qasm_to_circuit(qasm_src)
        n = n_qubits or (
            max((max(g.qubits) for g in gates if g.qubits), default=0) + 1
        )
        g = zx_from_circuit(gates, n)
        n_before = len(g.nodes)
        log = ZXRewriteLog()
        total = simplify(g, rules=rules, log=log)
        n_after = len(g.nodes)
        rule_counts: dict[str, int] = {}
        for step in log.steps:
            rule_counts[step["rule"]] = rule_counts.get(step["rule"], 0) + 1
        out = {
            "n_qubits": n,
            "n_gates_in": len(gates),
            "nodes_before": n_before,
            "nodes_after": n_after,
            "rewrites_total": total,
            "rule_counts": rule_counts,
            "notes": "ZX simplification; locally sound rewrite rules [研究]",
        }
        return json.dumps(out, indent=2)

    # ── htf_inverse ───────────────────────────────────────────────────────
    @server.tool(
        description=(
            "Inverse design / Hamiltonian learning: find Hamiltonian parameters "
            "such that the ground-state energy matches a target value. "
            "Uses gradient descent on the parametric Hamiltonian family. "
            "Returns achieved E_0, parameters, and convergence status [工程]/[研究]."
        )
    )
    def htf_inverse(
        model: str = "ising",
        n: int = 4,
        target_e0: float = -1.5,
        n_restarts: int = 5,
        seed: int = 0,
    ) -> str:
        from .inverse import inverse_design

        result = inverse_design(
            target_e0=target_e0,
            model=model,
            n_sites=n,
            n_restarts=n_restarts,
            seed=seed,
        )
        out = {
            "model": model,
            "n_sites": n,
            "target_e0": target_e0,
            "E0_achieved": float(result.E0_achieved),
            "residual": float(result.residual),
            "params_opt": result.params_opt.tolist(),
            "param_names": result.param_names,
            "converged": bool(result.converged),
            "n_restarts": int(result.n_restarts),
            "notes": result.notes,
        }
        return json.dumps(out, indent=2)

    return server


def main() -> None:
    """Entry point: run the HTF MCP server over stdio."""
    if not HAS_MCP:
        raise SystemExit(
            "htf.mcp_server requires 'mcp'. Install with: pip install mcp"
        )
    server = _build_server()
    asyncio.run(server.run_stdio_async())


if __name__ == "__main__":
    main()
