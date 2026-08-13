"""HTF Layer 3 — Tensor Engine.

Executes a diagram by recursive tensor contraction. Two modes:

* ``"float"``  — fast, discovery-tier; **carries no error bound**.
* ``"certified"`` — interval-arithmetic with a rigorous error bound. This is a
  Phase-2 deliverable (see ``PLAN.md``) and currently raises ``NotImplementedError``
  rather than returning an uncertified number dressed up as certified. We do not
  fake certification.
"""
from __future__ import annotations

import numpy as np

from .topology import Box, Id, Then, Tensor, dims
from .functor import TensorFunctor


def _eval(d, F: TensorFunctor) -> np.ndarray:
    """Return the dense tensor of diagram ``d`` with layout dims(cod)+dims(dom)."""
    if isinstance(d, Box):
        return F.tensor(d)

    if isinstance(d, Id):
        ds = dims(d.ty)
        if not ds:
            return np.array(1.0)
        n = int(np.prod(ds))
        return np.eye(n, dtype=float).reshape(ds + ds)

    if isinstance(d, Then):
        Ff = _eval(d.f, F)  # shape: dims(f.cod) + dims(f.dom)
        Gg = _eval(d.g, F)  # shape: dims(g.cod) + dims(g.dom)
        nb = len(d.f.cod)   # shared wires (f.cod == g.dom)
        nc = len(d.g.cod)
        g_in = list(range(nc, nc + nb))   # g's input axes
        f_out = list(range(0, nb))        # f's output axes
        # result axes: dims(g.cod) + dims(f.dom) = dims(cod) + dims(dom)
        return np.tensordot(Gg, Ff, axes=(g_in, f_out))

    if isinstance(d, Tensor):
        Ff = _eval(d.f, F)
        Gg = _eval(d.g, F)
        nfo, nfi = len(d.f.cod), len(d.f.dom)
        ngo, ngi = len(d.g.cod), len(d.g.dom)
        outer = np.tensordot(Ff, Gg, axes=0)  # fcod+fdom+gcod+gdom
        perm = (
            list(range(0, nfo))                               # f.cod
            + list(range(nfo + nfi, nfo + nfi + ngo))         # g.cod
            + list(range(nfo, nfo + nfi))                     # f.dom
            + list(range(nfo + nfi + ngo, nfo + nfi + ngo + ngi))  # g.dom
        )
        return np.transpose(outer, perm) if perm else outer

    raise TypeError(f"unknown diagram node: {type(d)!r}")


def contract(diagram, functor: TensorFunctor, mode: str = "float") -> np.ndarray:
    """Contract ``diagram`` under ``functor``.

    ``mode="float"`` returns the dense result (discovery-tier, no error bound).
    ``mode="certified"`` is not yet implemented (Phase 2).
    """
    if mode == "float":
        return _eval(diagram, functor)
    if mode == "certified":
        raise NotImplementedError(
            "certified (interval-arithmetic) mode is a Phase-2 deliverable; see "
            "PLAN.md. Float mode is discovery-tier and carries no error bound."
        )
    raise ValueError(f"unknown mode {mode!r} (expected 'float' or 'certified')")
