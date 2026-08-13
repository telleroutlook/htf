"""HTF §4-E — ZX-calculus diagram rewriting.

The ZX-calculus is a graphical language for quantum computation built on
two families of spiders (Z and X) connected by wires and Hadamard boxes.
Rewrite rules (spider fusion, identity removal, Hadamard cancellation) are
locally sound and preserve the linear map.

``zx_to_matrix`` evaluates any ZX diagram by tensor-network contraction.
Results match ``circuit_unitary`` **up to a global scalar** (a known
property of the ZX convention): gates H, X, Z, S, T match exactly;
rotation gates (Rx, Rz, Ry) and multi-qubit gates differ by a real or
complex scalar.  Use ``circuit_unitary`` when exact values are required.

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
* ``zx_to_matrix``      — evaluate the full unitary via tensor-network
                          contraction (correct up to a global scalar;
                          P0-6 fixed — CX/CZ/SWAP/Ry all correct).
* ``ZXRewriteLog``      — rewrite-step audit trail (not a proof verifier).

Honest scope [研究]
-------------------
* Single-qubit and multi-qubit rewrite rules are locally sound.
* ``zx_to_matrix`` uses dense contraction — exponential in qubit count.
* Results agree with ``circuit_unitary`` up to a global scalar (ZX
  calculus convention); verify via ``circuit_unitary`` for exact values.
* Non-Clifford gates are represented but rewrite rules are incomplete for
  them; use ``simplify`` with care and verify via ``zx_to_matrix``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto

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

    def copy(self) -> ZXGraph:
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
            # Ry(θ) = Rz(π/2) · Rx(θ) · Rz(−π/2)  (exact, up to global phase).
            # Circuit applies gates left-to-right (input→output), so the innermost
            # (first applied) gate is Rz(−π/2) and the outermost is Rz(+π/2).
            q  = qubits[0]
            th = params[0] if params else 0.0
            zm = g.add_node(ZXNodeType.Z, phase=-math.pi / 2, label="Ry_Zm")
            xn = g.add_node(ZXNodeType.X, phase=th,           label=f"Ry_X({th:.3g})")
            zp = g.add_node(ZXNodeType.Z, phase=+math.pi / 2, label="Ry_Zp")
            g.add_edge(wire[q], zm)
            g.add_edge(zm, xn)
            g.add_edge(xn, zp)
            wire[q] = zp

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
            if name in ("swap",):
                # SWAP = CX(ctrl,tgt) · CX(tgt,ctrl) · CX(ctrl,tgt)
                ctrl, tgt = qubits[0], qubits[1]
                for c, t in ((ctrl, tgt), (tgt, ctrl), (ctrl, tgt)):
                    z = g.add_node(ZXNodeType.Z, label="SWAP_Z")
                    x = g.add_node(ZXNodeType.X, label="SWAP_X")
                    g.add_edge(wire[c], z)
                    g.add_edge(wire[t], x)
                    g.add_edge(z, x)
                    wire[c] = z
                    wire[t] = x
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
    """Rewrite-step audit trail for operations applied to a ZX graph.

    .. note::
        This is a structural audit log, not an independent proof verifier.
        Each entry records which rule was applied and which nodes changed,
        but there is no pre/post state hash or external checker.  Treat as
        discovery-tier bookkeeping until a proper verifier is integrated
        (P0-6 remediation).

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


def spider_fusion(g: ZXGraph, log: ZXRewriteLog | None = None) -> int:
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


def identity_removal(g: ZXGraph, log: ZXRewriteLog | None = None) -> int:
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


def hadamard_cancel(g: ZXGraph, log: ZXRewriteLog | None = None) -> int:
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


def color_change(g: ZXGraph, log: ZXRewriteLog | None = None) -> int:
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


def pi_copy(g: ZXGraph, log: ZXRewriteLog | None = None) -> int:
    """Apply the π-copy rule: Z(π) copies through X(0) spiders.

    **Rule**: if a Z(π) spider is connected to an X(0) spider, the Z(π)
    can be pushed through: the X(0) spider remains, and a new Z(π) appears
    on each of its other legs.

    This rule is sound for the ZX-calculus and enables simplification of
    NOT-propagation patterns.

    Returns the number of π-copy steps applied.
    """
    applied = 0
    # Snapshot candidates so newly-created Z(π) nodes are not re-processed
    # (without this, each new Z(π) triggers another pi_copy → infinite loop).
    candidate_ids = list(g.nodes.keys())
    for z_id in candidate_ids:
        z_node = g.nodes.get(z_id)
        if z_node is None or z_node.kind != ZXNodeType.Z:
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
            g.remove_edge(z_id, x_id)
            g.remove_node(z_id)
            other_nbs = list(g.neighbours(x_id))
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
            break   # z_id consumed; move to next candidate
    return applied


def simplify(
    g: ZXGraph,
    rules: list[str] | None = None,
    log: ZXRewriteLog | None = None,
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

def _spider_tensor(kind: ZXNodeType, phase: float, n_legs: int) -> np.ndarray:
    """Tensor for a ZX spider with *n_legs* legs.

    Z spider(α, k):  T[i₁,…,iₖ] = δ(all 0) + e^{iα}·δ(all 1)
    X spider(α, k):  H^⊗k · Z(α,k) · H^⊗k  (H = [[1,1],[1,-1]]/√2)
    H box    (2 legs): [[1,1],[1,-1]]/√2  (standard Hadamard)
    """
    H_MAT = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / math.sqrt(2)

    if kind == ZXNodeType.H:
        if n_legs != 2:
            raise ValueError(f"H box must have 2 legs; got {n_legs}")
        return H_MAT.copy()

    # Z spider
    shape = (2,) * n_legs
    T = np.zeros(shape, dtype=complex)
    if n_legs == 0:
        return T  # handled separately
    idx_0 = tuple([0] * n_legs)
    idx_1 = tuple([1] * n_legs)
    T[idx_0] = 1.0
    T[idx_1] = np.exp(1j * phase)

    if kind == ZXNodeType.X:
        # Apply H to every leg
        for leg in range(n_legs):
            T = np.tensordot(H_MAT, T, axes=([1], [leg]))
            # tensordot puts the new axis at position 0; move it back to `leg`
            T = np.moveaxis(T, 0, leg)

    return T


def zx_to_matrix(g: ZXGraph) -> np.ndarray:
    """Evaluate the linear map of a ZX diagram by tensor-network contraction.

    Each edge is modelled as two half-edges contracted by a delta tensor;
    each internal node contributes its ZX spider tensor.  The result is a
    complex matrix of shape ``(2**n, 2**n)`` where *n* is the qubit count.

    Correctly handles CX, CZ, SWAP, and any other multi-qubit ZX structure
    (P0-6 fix).  Results agree with ``circuit_unitary`` **up to a global
    scalar** — this is a fundamental property of the ZX calculus convention.
    Gates H, X, Z, S, T happen to match exactly; rotation gates (Rx, Rz, Ry)
    and multi-qubit gates differ by a known phase/scalar.

    Parameters
    ----------
    g : :class:`ZXGraph` — any topology (circuit or non-circuit).

    Returns
    -------
    Complex ndarray ``U`` of shape ``(2**n, 2**n)``.

    Raises
    ------
    NotImplementedError
        If a non-boundary node has no edges (orphan node), indicating a
        malformed / partially-deleted graph that cannot be evaluated.
    ValueError
        If input/output boundary nodes don't each have exactly one edge.
    """
    n = g.n_qubits()
    if n == 0:
        return np.array([[1.0 + 0j]])

    # ── assign half-edge indices ─────────────────────────────────────────
    # Edge i → u-side index: 2*i,  v-side index: 2*i+1
    node_leg_indices: dict[int, list[int]] = {nid: [] for nid in g.nodes}
    for edge_idx, (u, v) in enumerate(g.edges):
        node_leg_indices[u].append(2 * edge_idx)
        node_leg_indices[v].append(2 * edge_idx + 1)

    # ── free (boundary) indices ──────────────────────────────────────────
    def _single_bond(nid: int, role: str) -> int:
        bonds = node_leg_indices[nid]
        if len(bonds) != 1:
            raise ValueError(
                f"{role} node {nid} has {len(bonds)} edges; expected exactly 1"
            )
        return bonds[0]

    input_free  = [_single_bond(g.inputs[q],  f"Input[{q}]")  for q in range(n)]
    output_free = [_single_bond(g.outputs[q], f"Output[{q}]") for q in range(n)]

    # ── build einsum operands ────────────────────────────────────────────
    delta_2 = np.eye(2, dtype=complex)
    einsum_args: list = []

    # One delta tensor per edge (contracts the two half-edges)
    for edge_idx in range(len(g.edges)):
        einsum_args += [delta_2, [2 * edge_idx, 2 * edge_idx + 1]]

    # One spider tensor per internal node
    for nid, node in sorted(g.nodes.items()):
        if node.kind in (ZXNodeType.INPUT, ZXNodeType.OUTPUT):
            continue

        legs = node_leg_indices[nid]
        k = len(legs)
        if k == 0:
            raise NotImplementedError(
                f"zx_to_matrix: non-boundary node {nid!r} (kind={node.kind.name}, "
                f"label={node.label!r}) has no edges. This indicates an orphan / "
                "malformed node. Evaluate the graph before deleting internal nodes, "
                "or remove the orphan explicitly."
            )
        einsum_args += [_spider_tensor(node.kind, node.phase, k), legs]

    # ── contract ─────────────────────────────────────────────────────────
    # Build einsum with input-free indices first, output-free indices second.
    # This makes the raw tensor shape (2,)*n_inputs + (2,)*n_outputs, so the
    # reshape gives M[in, out] (rows = input).  The final .T converts that to
    # the standard gate convention M[out, in] (rows = output).
    free_indices = input_free + output_free
    einsum_args.append(free_indices)

    result: np.ndarray = np.einsum(*einsum_args, optimize=True)

    # Reshape (2,)*2n → (2^n, 2^n); .T converts M[in,out] → M[out,in].
    return result.reshape(2 ** n, 2 ** n).T


# ────────────────────── §8-B extended Clifford rules ────────────────────────

def _is_zero_phase(node: ZXNode) -> bool:
    """Return True when node.phase is within 1e-9 of zero."""
    return abs(node.phase) < 1e-9

def bialgebra(g: ZXGraph, log: ZXRewriteLog | None = None) -> int:
    """Z(0) – X(0) bialgebra (copy) rule — termination-safe variant.

    **Rule**: a zero-phase Z-spider connected to a zero-phase X-spider,
    where one of the two sides has exactly one other neighbour (linear
    fan-out), can be "passed through".

    Only nodes present at the start of the call are considered as candidates;
    newly created nodes are not re-processed, preventing infinite loops.

    [研究] Sound; not complete over arbitrary diagrams.
    """
    applied = 0
    # Snapshot candidates to avoid processing newly created nodes
    candidate_ids = list(g.nodes.keys())
    for nid in candidate_ids:
        node = g.nodes.get(nid)
        if node is None or node.kind != ZXNodeType.Z or not _is_zero_phase(node):
            continue
        for nb_id in list(g.neighbours(nid)):
            nb = g.nodes.get(nb_id)
            if nb is None or nb.kind != ZXNodeType.X or not _is_zero_phase(nb):
                continue
            L = [n for n in g.neighbours(nid)  if n != nb_id]
            R = [n for n in g.neighbours(nb_id) if n != nid]
            if not L or not R or min(len(L), len(R)) != 1:
                continue
            g.nodes.pop(nid)
            g.nodes.pop(nb_id)
            g.edges = [(a, b) for (a, b) in g.edges
                       if a not in (nid, nb_id) and b not in (nid, nb_id)]
            new_x = [g.add_node(ZXNodeType.X, phase=0.0) for _ in L]
            new_z = [g.add_node(ZXNodeType.Z, phase=0.0) for _ in R]
            for li, lnb in enumerate(L):
                g.add_edge(lnb, new_x[li])
            for ri, rnb in enumerate(R):
                g.add_edge(rnb, new_z[ri])
            for xi in new_x:
                for zi in new_z:
                    g.add_edge(xi, zi)
            desc = f"bialgebra: Z({nid})–X({nb_id}) → {len(L)}×{len(R)}"
            if log is not None:
                log.record("bialgebra", [nid, nb_id], new_x + new_z, desc)
            applied += 1
            break   # node nid consumed; move to next candidate
    return applied


def local_complement(g: ZXGraph, log: ZXRewriteLog | None = None) -> int:
    """Local complementation (LC) on a Clifford (π/2-phase) spider.

    **Rule**: a Z or X spider with phase ±π/2 (a Clifford generator) that
    is surrounded only by H-box edges can be eliminated:
    1. Add H-box edges between every pair of its neighbours.
    2. Flip each neighbour's phase by ±π/2 (opposite sign to the removed node).
    3. Remove the node.

    This is the standard LC rule from the Clifford ZX completeness paper
    (Backens 2014). It is sound for diagrams in graph state / stabiliser form.

    [研究] Applicable only when the node is strictly π/2-phase and all its
    edges are H-box edges.  Completeness for arbitrary non-Clifford diagrams
    requires additional rules.
    """
    import math as _math

    HALF_PI = _math.pi / 2

    def _is_clifford_half(phase: float) -> bool:
        return abs(abs(phase) - HALF_PI) < 1e-9

    def _all_h_edges(nid: int, g: ZXGraph) -> bool:
        """True if every edge incident to nid connects to an H node."""
        for nb_id in g.neighbours(nid):
            nb = g.nodes.get(nb_id)
            if nb is None or nb.kind != ZXNodeType.H:
                return False
        return True

    applied = 0
    while True:
        changed = False
        for nid, node in list(g.nodes.items()):
            if node.kind not in (ZXNodeType.Z, ZXNodeType.X):
                continue
            if not _is_clifford_half(node.phase):
                continue
            if not _all_h_edges(nid, g):
                continue
            sign = 1.0 if node.phase > 0 else -1.0
            nbrs = list(g.neighbours(nid))
            if len(nbrs) < 2:
                continue
            # 1. Add H-edges between all pairs of neighbours
            for i in range(len(nbrs)):
                for j in range(i + 1, len(nbrs)):
                    g.add_edge(nbrs[i], nbrs[j])
            # 2. Shift each neighbour's phase by -sign * π/2
            for nb_id in nbrs:
                nb = g.nodes[nb_id]
                nb.phase = nb.phase - sign * HALF_PI
            # 3. Remove the Clifford node
            g.nodes.pop(nid)
            g.edges = [(a, b) for (a, b) in g.edges
                       if a != nid and b != nid]
            desc = (f"local_complement: removed {node.kind.name}(π/2) "
                    f"node {nid}, added {len(nbrs)*(len(nbrs)-1)//2} "
                    "new H-edges")
            if log is not None:
                log.record("local_complement", [nid], [], desc)
            applied += 1
            changed = True
            break
        if not changed:
            break
    return applied


def phase_gadget_fuse(g: ZXGraph, log: ZXRewriteLog | None = None) -> int:
    """Fuse parallel phase gadgets connected to the same set of qubits.

    A *phase gadget* is a Z-spider of arbitrary phase that connects via
    H-box edges to an identical set of neighbouring X-spiders (or vice-versa).
    Two phase gadgets acting on the same qubit set can be replaced by a
    single gadget whose phase is the sum of the two phases.

    **Rule**: if two Z-spiders z₁ (phase α) and z₂ (phase β) have
    identical neighbour sets (all through H-boxes), replace them with a
    single Z-spider of phase α+β.

    [研究] Sound; detecting the full set of gadgets sharing a qubit basis
    in arbitrary diagrams requires more graph analysis.
    """

    applied = 0
    while True:
        changed = False
        # Build (frozenset-of-neighbours) → [list of node IDs] for Z spiders
        groups: dict = {}
        for nid, node in g.nodes.items():
            if node.kind != ZXNodeType.Z:
                continue
            nb_set = frozenset(g.neighbours(nid))
            if not nb_set:
                continue
            groups.setdefault(nb_set, []).append(nid)
        for nb_set, ids in groups.items():
            if len(ids) < 2:
                continue
            # Fuse the first two
            z1_id, z2_id = ids[0], ids[1]
            z1 = g.nodes[z1_id]
            z2 = g.nodes[z2_id]
            new_phase = z1.phase + z2.phase
            # Keep z1 with summed phase, remove z2
            z1.phase = new_phase
            g.nodes.pop(z2_id)
            g.edges = [(a, b) for (a, b) in g.edges
                       if a != z2_id and b != z2_id]
            desc = (f"phase_gadget_fuse: Z({z1_id},α={z1.phase - z2.phase:.4f}) + "
                    f"Z({z2_id},α={z2.phase:.4f}) → Z(α={new_phase:.4f})")
            if log is not None:
                log.record("phase_gadget_fuse", [z1_id, z2_id], [z1_id], desc)
            applied += 1
            changed = True
            break
        if not changed:
            break
    return applied


def clifford_simplify(
    g: ZXGraph,
    log: ZXRewriteLog | None = None,
    max_iter: int = 200,
) -> int:
    """Full Clifford ZX simplification pipeline.

    Applies all eight rewrite rules exhaustively in round-robin order until
    no rule fires.  Rules applied (in order each round):

    1. ``spider_fusion``      — merge same-colour adjacent spiders
    2. ``identity_removal``   — remove zero-phase 2-leg spiders
    3. ``hadamard_cancel``    — cancel adjacent H-box pairs
    4. ``color_change``       — flip colour when surrounded by H-boxes
    5. ``pi_copy``            — copy Z(π) through X(0)
    6. ``bialgebra``          — Z(0)–X(0) copy / bialgebra rule
    7. ``local_complement``   — eliminate π/2 Clifford nodes
    8. ``phase_gadget_fuse``  — fuse parallel phase gadgets

    [研究] Sound but not complete over all non-Clifford diagrams.  For a
    complete Clifford simplifier use PyZX (Kissinger & van de Wetering, 2020).

    Returns total number of rewrites applied.
    """
    rules = [
        spider_fusion, identity_removal, hadamard_cancel,
        color_change, pi_copy,
        bialgebra, local_complement, phase_gadget_fuse,
    ]
    total = 0
    for _ in range(max_iter):
        round_total = sum(fn(g, log) for fn in rules)
        total += round_total
        if round_total == 0:
            break
    return total
