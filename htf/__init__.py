"""HTF — a certified, type-safe string-diagram / tensor-network framework.

Skeleton (v0.0.1): Layer 1 topology (:mod:`htf.topology`), Layer 2 functorial
mapping (:mod:`htf.functor`), Layer 3 tensor engine (:mod:`htf.engine`), plus a
provenance :class:`~htf.certificate.Certificate` and an agent-drivable CLI.

Honest scope: this is a *certified model engine*, not a "world engine". It
certifies numerical/truncation error, not modeling error; the continuum limit
(``chi -> inf``) is a wall the framework does not cross. See ``PLAN.md`` and
``docs/``.
"""
from __future__ import annotations

from .certificate import Certificate
from .engine import contract
from .functor import TensorFunctor
from .topology import Box, Diagram, Id, Wire

__version__ = "0.0.1"
__all__ = [
    "Wire",
    "Box",
    "Id",
    "Diagram",
    "TensorFunctor",
    "contract",
    "Certificate",
    "__version__",
]
