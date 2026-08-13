"""HTF Layer 1 — Symbolic Topology.

Wires (objects) and Boxes (morphisms) of a strict monoidal category, with
``>>`` (sequential composition) and ``@`` (tensor / parallel). Types are checked
at construction: composing incompatible types raises ``TypeError``, so a
structurally illegal diagram cannot be built ("physically-illegal code won't
compile"). No numerics live here — only structure.
"""
from __future__ import annotations


class Wire:
    """An object of the category: a labelled vector space of dimension ``dim``."""

    __slots__ = ("dim", "name")

    def __init__(self, name: str, dim: int):
        if int(dim) <= 0:
            raise ValueError("Wire dim must be a positive integer")
        self.name = name
        self.dim = int(dim)

    def __repr__(self) -> str:
        return f"Wire({self.name!r}, {self.dim})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Wire):
            return NotImplemented
        return self.name == other.name and self.dim == other.dim

    def __hash__(self) -> int:
        return hash((self.name, self.dim))


Ty = tuple[Wire, ...]  # a type is a tuple of wires


def dims(ty: Ty) -> tuple[int, ...]:
    """The tuple of dimensions of a type (what the tensor layer actually sees)."""
    return tuple(w.dim for w in ty)


class Diagram:
    """Base morphism ``dom -> cod`` in the strict monoidal category of tensors."""

    dom: Ty = ()
    cod: Ty = ()

    def __rshift__(self, other: Diagram) -> Then:
        return Then(self, other)

    def __matmul__(self, other: Diagram) -> Tensor:
        return Tensor(self, other)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({dims(self.dom)} -> {dims(self.cod)})"


class Box(Diagram):
    """An atomic morphism, later assigned a concrete tensor by a functor."""

    def __init__(self, name: str, dom: Ty, cod: Ty):
        self.name = name
        self.dom = tuple(dom)
        self.cod = tuple(cod)

    def __repr__(self) -> str:
        return f"Box({self.name!r}: {dims(self.dom)} -> {dims(self.cod)})"


class Id(Diagram):
    """The identity morphism on a type."""

    def __init__(self, ty: Ty):
        self.ty = tuple(ty)
        self.dom = self.ty
        self.cod = self.ty


class Then(Diagram):
    """Sequential composition ``f >> g`` (data flows through ``f`` then ``g``)."""

    def __init__(self, f: Diagram, g: Diagram):
        if dims(f.cod) != dims(g.dom):
            raise TypeError(
                f"type mismatch in `>>`: cod {dims(f.cod)} does not match "
                f"dom {dims(g.dom)} — this diagram is not well-typed"
            )
        self.f = f
        self.g = g
        self.dom = f.dom
        self.cod = g.cod


class Tensor(Diagram):
    """Parallel composition ``f @ g`` (tensor product of morphisms)."""

    def __init__(self, f: Diagram, g: Diagram):
        self.f = f
        self.g = g
        self.dom = f.dom + g.dom
        self.cod = f.cod + g.cod
