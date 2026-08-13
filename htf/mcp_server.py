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

try:
    from mcp.server.mcpserver import MCPServer
    HAS_MCP = True
except ImportError:
    HAS_MCP = False


def _build_server() -> "MCPServer":
    """Construct and configure the HTF MCP server."""
    if not HAS_MCP:
        raise ImportError("htf.mcp_server requires 'mcp'. Install with: pip install mcp")

    import numpy as np
    from mcp.server.mcpserver import MCPServer

    from . import __version__

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

        class _Args:
            pass

        a = _Args()
        a.model, a.n, a.J, a.h = model, n, J, h
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
            "Spectral gap bounds: exact gap from full diagonalisation, "
            "variational upper bound, Temple's inequality lower bound on E_0, "
            "and a certified Arb bound on the gap. "
            "Certified bounds cover FP rounding only; bond-dimension bias is [OUT]."
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

        class _Args:
            pass

        a = _Args()
        a.model, a.n, a.J, a.h = model, n, J, h
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
            "temple_lb": report["temple_lb"],
            "temple_condition_met": bool(report["E0_var"] < evals[1]),
            "gap_cert": _cert_to_dict(report["gap_cert"]),
            "notes": (
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

        class _Args:
            pass

        a = _Args()
        a.model, a.n, a.J, a.h = model, n, J, h
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
        rep = run_benchmark(
            n_sites=n, chi=chi, n_iter=n_iter, seed=seed, models=models
        )
        return rep.to_json(indent=2)

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
