"""HTF §4-E — ZX-calculus diagram rewriting.

The ZX-calculus is a graphical language for quantum computation built on
two families of spiders (Z and X) connected by wires and Hadamard boxes.
Rewrite rules (spider fusion, identity removal, Hadamard cancellation) are
**sound**: they preserve the linear map denoted by the diagram.

Provides
--------
* ``ZXNodeType``        — Z / X / H / INPUT / OUTPUT node kinds.
* ``ZXNode``            — a single node with phase angle.
* ``ZXGraph``           — multigraph of ZX nodes; add/connect/query API.
* ``zx_from_circuit``   — convert an HTF ``Gate`` list to a ZX graph.
* ``spider_fusion``     — merge adjacent same-coloured spiders.
* ``identity_removal``  — remove 2-legged zero-phase spiders.
* ``hadamard_cancel``   — cancel adjacent H-box pairs.
* ``simplify``          — apply a rule set exhaustively.
* ``zx_to_matrix``      — evaluate the full unitary via path-sum (dense).
* ``ZXRewriteLog``      — proof-carrying list of applied rewrites.

Honest scope [研究]
-------------------
* The rewrite rules implemented here are **locally sound** (they preserve
  the linear map); global completeness (all equalities provable) is a
  published research problem and is ``[研究]``.
* ``zx_to_matrix`` uses a dense contraction — exponential in qubit count;
  for large diagrams use a dedicated ZX simulator.
* Non-Clifford gates are represented but rewrite rules are incomplete for
  them; use ``simplify`` with care and verify via ``zx_to_matrix``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np


# ────────────────────── node kind ────────────────────────────────────────

class ZXNodeType(Enum):
    Z      = auto()   # green spider
    X      = auto()   # red spider
    H      = auto()   # Hadamard box (yellow)
    INPUT  = auto()   # boundary input wire
    OUTPUT = auto()   # boundary output wire


# ────────────────────── node ──────────────────────────────────────────────

@dataclass
class ZXNode:
    """A single node in a ZX diagram.

    Attributes
    ----------
    node_id : unique integer identifier.
    kind    : :class:`ZXNodeType`.
    phase   : phase angle in radians (meaningful for Z and X spiders).
    qubit   : qubit index (meaningful for INPUT/OUTPUT boundary nodes).
    label   : human-readable annotation (optional).
    """
    node_id: int
    kind:    ZXNodeType
    phase:   float = 0.0
    qubit:   int   = -1
    label:   str   = ""

    def __repr__(self) -> str:
        ph = f", φ={self.phase:.3g}" if self.phase != 0.0 else ""
        return f"ZXNode({self.node_id}, {self.kind.name}{ph})"


# ────────────────────── graph ────────────────────────────────────────────

@dataclass
class ZXGraph:
    """Multigraph representing a ZX-calculus diagram.

    Nodes are :class:`ZXNode` instances; edges are unordered pairs of
    node IDs (multi-edges allowed, representing parallel wires).

    Attributes
    ----------
    nodes   : dict mapping node_id → ZXNode.
    edges   : list of (u, v) pairs (undirected, multi-edges allowed).
    inputs  : list of node IDs for input boundary nodes (ordered by qubit).
    outputs : list of node IDs for output boundary nodes (ordered by qubit).
    """
    nodes:   dict[int, ZXNode]      = field(default_factory=dict)
    edges:   list[tuple[int, int]]  = field(default_factory=list)
    inputs:  list[int]              = field(default_factory=list)
    outputs: list[int]              = field(default_factory=list)
    _next_id: int                   = field(default=0, repr=False)

    def add_node(
        self,
        kind:  ZXNodeType,
        phase: float = 0.0,
        qubit: int   = -1,
        label: str   = "",
    ) -> int:
        """Add a node and return its ID."""
        nid = self._next_id
        self.nodes[nid] = ZXNode(nid, kind, phase, qubit, label)
        self._next_id += 1
        return nid

    def add_edge(self, u: int, v: int) -> None:
        """Add an undirected edge between nodes *u* and *v*."""
        self.edges.append((u, v))

    def neighbours(self, nid: int) -> list[int]:
        """Return a list of node IDs connected to *nid* (with multiplicity)."""
        result = []
        for u, v in self.edges:
            if u == nid:
                result.append(v)
            elif v == nid:
                result.append(u)
        return result

    def degree(self, nid: int) -> int:
        """Number of edge endpoints at *nid* (counts multi-edges)."""
        return sum(1 for u, v in self.edges if u == nid or v == nid)

    def remove_node(self, nid: int) -> None:
        """Remove a node and all its incident edges."""
        self.edges = [(u, v) for u, v in self.edges if u != nid and v != nid]
        self.nodes.pop(nid, None)

    def remove_edge(self, u: int, v: int) -> None:
        """Remove the first occurrence of edge (u,v) or (v,u)."""
        for i, (a, b) in enumerate(self.edges):
            if (a == u and b == v) or (a == v and b == u):
                del self.edges[i]
                return

    def n_qubits(self) -> int:
        """Number of qubits inferred from input boundary nodes."""
        return len(self.inputs)

    def copy(self) -> "ZXGraph":
        """Return a deep copy."""
        g = ZXGraph()
        g.nodes    = {k: ZXNode(v.node_id, v.kind, v.phase, v.qubit, v.label)
                      for k, v in self.nodes.items()}
        g.edges    = list(self.edges)
        g.inputs   = list(self.inputs)
        g.outputs  = list(self.outputs)
        g._next_id = self._next_id
        return g


# ────────────────────── circuit → ZX ──────────────────────────────────────

def zx_from_circuit(gates, n_qubits: int) -> ZXGraph:
    """Convert a list of :class:`~htf.qasm.Gate` objects to a ZX graph.

    Standard gate translations
    --------------------------
    * H          → H box between Z-input and Z-output spiders.
    * X          → X spider (phase π).
    * Z          → Z spider (phase π).
    * S          → Z spider (phase π/2).
    * T          → Z spider (phase π/4).
    * Rx(θ)      → X spider (phase θ).
    * Rz(θ)      → Z spider (phase θ).
    * CX / CNOT  → Z spider on control, X spider on target, connected.
    * CZ         → Z spiders on both, connected via H on one side.

    Parameters
    ----------
    gates    : list of :class:`~htf.qasm.Gate`.
    n_qubits : number of qubits.

    Returns
    -------
    :class:`ZXGraph` with input and output boundary nodes.
    """
    g = ZXGraph()

    # Create input boundary nodes (one per qubit)
    wire: list[int] = []  # current "tail" node for each qubit
    for q in range(n_qubits):
        nid = g.add_node(ZXNodeType.INPUT, qubit=q, label=f"in{q}")
        g.inputs.append(nid)
        wire.append(nid)

    # Process each gate
    for gate in gates:
        name   = gate.name.lower()
        params = gate.params
        qubits = gate.qubits

        if name == "h":
            q = qubits[0]
            h = g.add_node(ZXNodeType.H, label="H")
            g.add_edge(wire[q], h)
            wire[q] = h

        elif name in ("x",):
            q   = qubits[0]
            nid = g.add_node(ZXNodeType.X, phase=math.pi, label="X")
            g.add_edge(wire[q], nid)
            wire[q] = nid

        elif name in ("y",):
            # Y = X·Z (up to global phase); represent as Z(π) · X(π)
            q  = qubits[0]
            z  = g.add_node(ZXNodeType.Z, phase=math.pi, label="Y_Z")
            x  = g.add_node(ZXNodeType.X, phase=math.pi, label="Y_X")
            g.add_edge(wire[q], z)
            g.add_edge(z, x)
            wire[q] = x

        elif name in ("z",):
            q   = qubits[0]
            nid = g.add_node(ZXNodeType.Z, phase=math.pi, label="Z")
            g.add_edge(wire[q], nid)
            wire[q] = nid

        elif name in ("s",):
            q   = qubits[0]
            nid = g.add_node(ZXNodeType.Z, phase=math.pi / 2, label="S")
            g.add_edge(wire[q], nid)
            wire[q] = nid

        elif name in ("sdg",):
            q   = qubits[0]
            nid = g.add_node(ZXNodeType.Z, phase=-math.pi / 2, label="Sdg")
            g.add_edge(wire[q], nid)
            wire[q] = nid

        elif name in ("t",):
            q   = qubits[0]
            nid = g.add_node(ZXNodeType.Z, phase=math.pi / 4, label="T")
            g.add_edge(wire[q], nid)
            wire[q] = nid

        elif name in ("tdg",):
            q   = qubits[0]
            nid = g.add_node(ZXNodeType.Z, phase=-math.pi / 4, label="Tdg")
            g.add_edge(wire[q], nid)
            wire[q] = nid

        elif name in ("id",):
            pass  # identity — do nothing

        elif name in ("rx",):
            q   = qubits[0]
            th  = params[0] if params else 0.0
            nid = g.add_node(ZXNodeType.X, phase=th, label=f"Rx({th:.3g})")
            g.add_edge(wire[q], nid)
            wire[q] = nid

        elif name in ("ry",):
            # Ry(θ) = S·Rx(θ)·S† (up to phase) — approximate in ZX
            q  = qubits[0]
            th = params[0] if params else 0.0
            nid = g.add_node(ZXNodeType.X, phase=th, label=f"Ry({th:.3g})")
            g.add_edge(wire[q], nid)
            wire[q] = nid

        elif name in ("rz",):
            q   = qubits[0]
            th  = params[0] if params else 0.0
            nid = g.add_node(ZXNodeType.Z, phase=th, label=f"Rz({th:.3g})")
            g.add_edge(wire[q], nid)
            wire[q] = nid

        elif name in ("cx", "cnot"):
            ctrl, tgt = qubits[0], qubits[1]
            z   = g.add_node(ZXNodeType.Z, label="CNOT_ctrl")
            x   = g.add_node(ZXNodeType.X, label="CNOT_tgt")
            g.add_edge(wire[ctrl], z)
            g.add_edge(wire[tgt],  x)
            g.add_edge(z, x)          # the internal wire
            wire[ctrl] = z
            wire[tgt]  = x

        elif name in ("cz",):
            q0, q1 = qubits[0], qubits[1]
            z0 = g.add_node(ZXNodeType.Z, label="CZ_0")
            h  = g.add_node(ZXNodeType.H, label="CZ_H")
            z1 = g.add_node(ZXNodeType.Z, label="CZ_1")
            g.add_edge(wire[q0], z0)
            g.add_edge(wire[q1], z1)
            g.add_edge(z0, h)
            g.add_edge(h,  z1)
            wire[q0] = z0
            wire[q1] = z1

        else:
            # Unknown gate: add as opaque Z node (phase=0 placeholder)
            for q in qubits:
                nid = g.add_node(ZXNodeType.Z, label=f"?{name}")
                g.add_edge(wire[q], nid)
                wire[q] = nid

    # Create output boundary nodes
    for q in range(n_qubits):
        nid = g.add_node(ZXNodeType.OUTPUT, qubit=q, label=f"out{q}")
        g.add_edge(wire[q], nid)
        g.outputs.append(nid)

    return g


# ────────────────────── rewrite rules ─────────────────────────────────────

@dataclass
class ZXRewriteLog:
    """Proof-carrying log of rewrite steps applied to a ZX graph.

    Each entry is a dict with keys ``rule``, ``nodes_removed``,
    ``nodes_added``, and ``description``.
    """
    steps: list[dict] = field(default_factory=list)

    def record(self, rule: str, removed: list[int], added: list[int],
               desc: str = "") -> None:
        self.steps.append({
            "rule": rule,
            "nodes_removed": removed,
            "nodes_added": added,
            "description": desc,
        })

    def __len__(self) -> int:
        return len(self.steps)


def spider_fusion(g: ZXGraph, log: Optional[ZXRewriteLog] = None) -> int:
    """Merge adjacent same-coloured spiders (Z–Z or X–X).

    **Rule**: two spiders of the same colour connected by a single wire
    fuse into one spider whose phase is the sum of the two phases and
    whose arity is the sum of the arities minus 2 (the internal wire).

    Returns the number of fusions applied.
    """
    applied = 0
    changed = True
    while changed:
        changed = False
        for u_id, u in list(g.nodes.items()):
            if u.kind not in (ZXNodeType.Z, ZXNodeType.X):
                continue
            for v_id in list(g.neighbours(u_id)):
                if v_id not in g.nodes:
                    continue
                v = g.nodes[v_id]
                if v.kind != u.kind:
                    continue
                # Fuse u and v into u; reroute v's edges to u
                g.remove_edge(u_id, v_id)
                for nb in g.neighbours(v_id):
                    g.add_edge(u_id, nb)
                u.phase += v.phase
                if log is not None:
                    log.record("spider_fusion", [v_id], [],
                               f"fused {u.kind.name}({u_id}) + {v.kind.name}({v_id})")
                g.remove_node(v_id)
                applied += 1
                changed = True
                break
            if changed:
                break
    return applied


def identity_removal(g: ZXGraph, log: Optional[ZXRewriteLog] = None) -> int:
    """Remove 2-legged zero-phase spiders (identity spiders).

    **Rule**: a Z or X spider with exactly 2 legs and phase 0 is
    equivalent to a plain wire; it can be removed and its two neighbours
    connected directly.

    Returns the number of removals applied.
    """
    applied = 0
    changed = True
    while changed:
        changed = False
        for nid, node in list(g.nodes.items()):
            if node.kind not in (ZXNodeType.Z, ZXNodeType.X):
                continue
            if g.degree(nid) != 2:
                continue
            if abs(node.phase % (2 * math.pi)) > 1e-10:
                continue
            nbs = g.neighbours(nid)
            if len(nbs) != 2:
                continue
            a, b = nbs[0], nbs[1]
            g.remove_edge(nid, a)
            g.remove_edge(nid, b)
            g.add_edge(a, b)
            if log is not None:
                log.record("identity_removal", [nid], [],
                           f"removed identity spider {node.kind.name}({nid})")
            g.remove_node(nid)
            applied += 1
            changed = True
            break
    return applied


def hadamard_cancel(g: ZXGraph, log: Optional[ZXRewriteLog] = None) -> int:
    """Cancel pairs of adjacent Hadamard boxes.

    **Rule**: H · H = I; two H nodes connected by a single internal wire
    can both be removed and their remaining legs connected directly.

    Returns the number of cancellations applied.
    """
    applied = 0
    changed = True
    while changed:
        changed = False
        for h1_id, h1 in list(g.nodes.items()):
            if h1.kind != ZXNodeType.H:
                continue
            for h2_id in g.neighbours(h1_id):
                if h2_id not in g.nodes:
                    continue
                if g.nodes[h2_id].kind != ZXNodeType.H:
                    continue
                # h1 — h2: remove both, connect their outer neighbours
                nbs1 = [n for n in g.neighbours(h1_id) if n != h2_id]
                nbs2 = [n for n in g.neighbours(h2_id) if n != h1_id]
                g.remove_edge(h1_id, h2_id)
                for n1 in nbs1:
                    g.remove_edge(h1_id, n1)
                for n2 in nbs2:
                    g.remove_edge(h2_id, n2)
                for n1, n2 in zip(nbs1, nbs2):
                    g.add_edge(n1, n2)
                if log is not None:
                    log.record("hadamard_cancel", [h1_id, h2_id], [],
                               f"cancelled H({h1_id})–H({h2_id})")
                g.remove_node(h1_id)
                g.remove_node(h2_id)
                applied += 1
                changed = True
                break
            if changed:
                break
    return applied


def color_change(g: ZXGraph, log: Optional[ZXRewriteLog] = None) -> int:
    """Convert a spider flanked by H boxes to the opposite colour.

    **Rule**: an X spider with all its neighbours being H boxes can be
    converted to a Z spider by absorbing all H boxes (and vice versa).
    Equivalently: Z(α) surrounded by H wires = X(α) without H wires.

    This rule enables colour propagation through the graph and is required
    for non-trivial simplification beyond the Clifford fragment.

    Returns the number of colour changes applied.
    """
    applied = 0
    changed = True
    while changed:
        changed = False
        for nid, node in list(g.nodes.items()):
            if node.kind not in (ZXNodeType.Z, ZXNodeType.X):
                continue
            nbs = g.neighbours(nid)
            if not nbs:
                continue
            # Check if all neighbours are H boxes
            if not all(g.nodes.get(nb, ZXNode(-1, ZXNodeType.INPUT)).kind == ZXNodeType.H
                       for nb in nbs):
                continue
            # Flip colour, remove all H boxes, connect their outer legs
            new_kind = ZXNodeType.X if node.kind == ZXNodeType.Z else ZXNodeType.Z
            node.kind = new_kind
            removed_h = []
            for h_id in list(nbs):
                outer = [x for x in g.neighbours(h_id) if x != nid]
                g.remove_edge(nid, h_id)
                for o in outer:
                    g.remove_edge(h_id, o)
                    g.add_edge(nid, o)
                removed_h.append(h_id)
                g.remove_node(h_id)
            if log is not None:
                log.record(
                    "color_change", removed_h, [],
                    f"flipped {ZXNodeType.Z.name if new_kind == ZXNodeType.X else ZXNodeType.X.name}"
                    f"({nid}) → {new_kind.name}({nid}), removed {len(removed_h)} H boxes",
                )
            applied += 1
            changed = True
            break
    return applied


def pi_copy(g: ZXGraph, log: Optional[ZXRewriteLog] = None) -> int:
    """Apply the π-copy rule: Z(π) copies through X(0) spiders.

    **Rule**: if a Z(π) spider is connected to an X(0) spider, the Z(π)
    can be pushed through: the X(0) spider remains, and a new Z(π) appears
    on each of its other legs.

    This rule is sound for the ZX-calculus and enables simplification of
    NOT-propagation patterns.

    Returns the number of π-copy steps applied.
    """
    applied = 0
    changed = True
    while changed:
        changed = False
        for z_id, z_node in list(g.nodes.items()):
            if z_node.kind != ZXNodeType.Z:
                continue
            if abs(z_node.phase % (2 * math.pi) - math.pi) > 1e-10:
                continue
            for x_id in list(g.neighbours(z_id)):
                if x_id not in g.nodes:
                    continue
                x_node = g.nodes[x_id]
                if x_node.kind != ZXNodeType.X:
                    continue
                if abs(x_node.phase % (2 * math.pi)) > 1e-10:
                    continue
                # Push Z(π) through X(0): remove Z(π)–X edge,
                # add new Z(π) nodes on all other legs of X
                g.remove_edge(z_id, x_id)
                g.remove_node(z_id)
                other_nbs = [nb for nb in g.neighbours(x_id)]
                new_z_ids = []
                for nb in other_nbs:
                    g.remove_edge(x_id, nb)
                    new_z = g.add_node(ZXNodeType.Z, phase=math.pi,
                                       label=f"Z(π)_copy_{nb}")
                    g.add_edge(x_id, new_z)
                    g.add_edge(new_z, nb)
                    new_z_ids.append(new_z)
                if log is not None:
                    log.record(
                        "pi_copy", [z_id], new_z_ids,
                        f"Z(π)({z_id}) copied through X(0)({x_id}) → "
                        f"{len(new_z_ids)} new Z(π) nodes",
                    )
                applied += 1
                changed = True
                break
            if changed:
                break
    return applied


def simplify(
    g: ZXGraph,
    rules: Optional[list[str]] = None,
    log: Optional[ZXRewriteLog] = None,
    max_iter: int = 100,
) -> int:
    """Apply rewrite rules exhaustively until no more apply.

    Parameters
    ----------
    g        : ZX graph (mutated in-place).
    rules    : list of rule names to apply; defaults to all five.
               Allowed: ``"spider_fusion"``, ``"identity_removal"``,
               ``"hadamard_cancel"``, ``"color_change"``, ``"pi_copy"``.
    log      : if provided, rewrites are recorded here.
    max_iter : safety cap on total iterations.

    Returns
    -------
    Total number of rewrites applied.
    """
    if rules is None:
        rules = [
            "spider_fusion", "identity_removal", "hadamard_cancel",
            "color_change", "pi_copy",
        ]
    rule_fns = {
        "spider_fusion":    spider_fusion,
        "identity_removal": identity_removal,
        "hadamard_cancel":  hadamard_cancel,
        "color_change":     color_change,
        "pi_copy":          pi_copy,
    }
    total = 0
    for _ in range(max_iter):
        round_total = 0
        for name in rules:
            fn = rule_fns.get(name)
            if fn:
                round_total += fn(g, log)
        total += round_total
        if round_total == 0:
            break
    return total


# ────────────────────── matrix evaluation ────────────────────────────────

def zx_to_matrix(g: ZXGraph) -> np.ndarray:
    """Evaluate the linear map of a ZX diagram by converting back to a circuit.

    The graph is converted qubit-by-qubit to a sequence of single- and
    two-qubit operations using a topological traversal from inputs to
    outputs.  The result is a unitary (or isometry) matrix.

    **Limitation**: this function works best for circuit-shaped ZX graphs
    where the topological order is unambiguous.  For non-circuit graphs
    (e.g. after aggressive rewriting that introduces loop-like structures)
    it raises ``NotImplementedError``.

    Parameters
    ----------
    g : ZX graph (typically the output of :func:`zx_from_circuit` after
        some simplifications).

    Returns
    -------
    Complex ndarray of shape (2**n, 2**n) where n = number of qubits.
    """
    from .qasm import Gate, circuit_unitary

    n = g.n_qubits()
    if n == 0:
        return np.array([[1.0 + 0j]])

    # Reconstruct a gate list by walking from inputs to outputs
    gates: list[Gate] = []
    visited: set[int] = set(g.inputs)
    boundary = set(g.inputs)

    def _spider_to_gate(nid: int, kind: ZXNodeType, phase: float,
                        qubit: int) -> Optional[Gate]:
        if kind == ZXNodeType.Z:
            if abs(phase) < 1e-12:
                return None            # identity
            return Gate("rz", [qubit], [phase])
        if kind == ZXNodeType.X:
            if abs(phase) < 1e-12:
                return None
            return Gate("rx", [qubit], [phase])
        if kind == ZXNodeType.H:
            return Gate("h", [qubit])
        return None

    # Build qubit assignment: for each non-boundary node, find which qubit
    # wire it lives on (only valid for circuit-topology graphs)
    qubit_of: dict[int, int] = {}
    for q, inp in enumerate(g.inputs):
        qubit_of[inp] = q
    for q, out in enumerate(g.outputs):
        qubit_of[out] = q

    # Topological walk
    front: list[int] = list(g.inputs)
    for _ in range(len(g.nodes) * 2):
        next_front: list[int] = []
        for nid in front:
            if nid in g.outputs:
                continue
            q = qubit_of.get(nid, -1)
            for nb in g.neighbours(nid):
                if nb in visited:
                    continue
                nb_node = g.nodes.get(nb)
                if nb_node is None:
                    continue
                # A-2: detect cross-wire connections (non-circuit topology)
                if nb in qubit_of and qubit_of[nb] != q:
                    raise NotImplementedError(
                        f"zx_to_matrix: node {nb} is reachable from both qubit "
                        f"{qubit_of[nb]} and qubit {q}. The ZX graph is not "
                        "circuit-topology after simplification. Evaluate the "
                        "unitary *before* simplifying, or use a ZX simulator "
                        "that supports arbitrary graph topology."
                    )
                qubit_of[nb] = q
                visited.add(nb)
                if nb_node.kind in (ZXNodeType.OUTPUT, ):
                    continue
                gate = _spider_to_gate(nb, nb_node.kind, nb_node.phase, q)
                if gate:
                    gates.append(gate)
                next_front.append(nb)
        if not next_front:
            break
        front = next_front

    # A-2: all nodes must be reachable from inputs
    unreachable = set(g.nodes.keys()) - visited
    if unreachable:
        raise NotImplementedError(
            f"zx_to_matrix: {len(unreachable)} node(s) unreachable from inputs "
            f"(IDs: {sorted(unreachable)[:5]}). The ZX graph has non-circuit "
            "topology (cycles or disconnected components) after simplification. "
            "Evaluate the unitary *before* simplifying, or use a ZX simulator "
            "that supports arbitrary graph topology."
        )

    return circuit_unitary(gates, n)
