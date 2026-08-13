"""HTF Phase 2 — 1D lattice operators.

Factory functions that return ``(Box, matrix)`` pairs for use with
:class:`~htf.functor.TensorFunctor`.  Each operator encodes a standard
finite-difference scheme; the value of wrapping it in a Box is
composability and compile-time type safety, not a new numerical method.

Stability note
--------------
The explicit-Euler heat step is unconditionally stable only when
``dt * D / dx² ≤ 0.5``.  The framework does not enforce this; it is the
caller's responsibility (as with any finite-difference code).
"""
from __future__ import annotations

import numpy as np

from .topology import Box, Wire


def site_wire(n: int) -> Wire:
    """Canonical ``Wire("site", n)`` for a 1-D lattice with *n* sites."""
    return Wire("site", n)


def laplacian_box(n: int, dx: float = 1.0) -> tuple[Box, np.ndarray]:
    """Second-order central-difference Laplacian on *n* sites (Dirichlet BC).

    Returns ``(box, L)`` where ``L`` has shape ``(n, n)`` and encodes the
    standard tridiagonal operator ``[-2, 1, 0, …] / dx²``.

    Convention: ``box`` tensor shape = ``dims(cod) + dims(dom) = (n, n)``.
    """
    s = site_wire(n)
    box = Box("laplacian", (s,), (s,))
    L = (
        np.diag(np.full(n, -2.0))
        + np.diag(np.ones(n - 1), 1)
        + np.diag(np.ones(n - 1), -1)
    ) / dx ** 2
    return box, L


def heat_step_box(
    n: int, D: float, dt: float, dx: float = 1.0
) -> tuple[Box, np.ndarray]:
    """One explicit-Euler step of the 1-D heat equation ``u_t = D · u_xx``.

    Returns ``(box, M)`` where ``M = I + dt · D · L``, shape ``(n, n)``.

    Stability: ``dt · D / dx² ≤ 0.5``.
    """
    s = site_wire(n)
    box = Box("heat_step", (s,), (s,))
    _, L = laplacian_box(n, dx)
    M = np.eye(n) + dt * D * L
    return box, M


def state_box(name: str, u0: np.ndarray) -> tuple[Box, np.ndarray]:
    """State-preparation Box: ``() → (site,)``.

    *u0* must be a 1-D array of length *n*.
    """
    (n,) = u0.shape
    s = site_wire(n)
    return Box(name, (), (s,)), u0.copy()


def effect_box(name: str, v: np.ndarray) -> tuple[Box, np.ndarray]:
    """Effect (bra) Box: ``(site,) → ()``.

    *v* must be a 1-D array of length *n*.
    """
    (n,) = v.shape
    s = site_wire(n)
    return Box(name, (s,), ()), v.copy()
