"""Agent-drivable command-line interface for HTF.

Designed so an LLM agent (or a human) can operate HTF from structured commands
with machine-readable JSON output. Subcommands are verbs; every result is JSON
so an agent can parse it and relay the certified bounds faithfully in natural
language — the certificate constrains the agent and prevents overstatement.
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


def _hello_diagram():
    """The Phase-1 hello-world: <phi| U |psi> for a swap gate U on one qubit."""
    s = Wire("spin", 2)
    psi = Box("psi", (), (s,))       # a state:  () -> spin
    U = Box("U", (s,), (s,))          # a gate:   spin -> spin
    phi = Box("phi", (s,), ())        # an effect: spin -> ()
    diagram = psi >> U >> phi         # () -> ()  (a scalar)
    F = TensorFunctor(
        {
            "psi": np.array([1.0, 0.0]),
            "U": np.array([[0.0, 1.0], [1.0, 0.0]]),  # swap
            "phi": np.array([0.0, 1.0]),
        }
    )
    return diagram, F


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


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="htf",
        description="HTF — a certified, type-safe string-diagram / tensor-network framework (skeleton).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("version", help="print the version as JSON").set_defaults(func=cmd_version)
    sub.add_parser(
        "hello", help="run the Phase-1 hello-world diagram and print a JSON certificate"
    ).set_defaults(func=cmd_hello)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
