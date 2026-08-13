"""HTF Layer 2 — Functorial Mapping.

A :class:`TensorFunctor` assigns a concrete tensor to each atomic :class:`Box`
and validates its shape against the box's declared type. It carries no
contraction logic (that is Layer 3, :mod:`htf.engine`); it only resolves and
type-checks the numerical data.

Convention: a box ``b: dom -> cod`` is given a tensor of shape
``dims(cod) + dims(dom)`` — output axes first, then input axes — so that a linear
map acts as ``out_i = sum_j T[i, j] in_j``.
"""
from __future__ import annotations

import numpy as np

from .topology import Box, dims


class TensorFunctor:
    def __init__(self, arrows: dict | None = None):
        self.arrows = dict(arrows or {})

    def tensor(self, box: Box) -> np.ndarray:
        if box.name not in self.arrows:
            raise KeyError(f"no tensor assigned to Box {box.name!r}")
        raw = np.asarray(self.arrows[box.name])
        if np.iscomplexobj(raw):
            raise TypeError(
                f"tensor for Box {box.name!r} has complex dtype {raw.dtype}; "
                "HTF currently supports real tensors only — "
                "complex support (Acb interval arithmetic) is a planned research gate"
            )
        arr = raw.astype(float)
        expected = dims(box.cod) + dims(box.dom)
        if arr.shape != expected:
            raise ValueError(
                f"tensor for Box {box.name!r} has shape {arr.shape}, "
                f"expected {expected} (= dims(cod) + dims(dom))"
            )
        return arr
