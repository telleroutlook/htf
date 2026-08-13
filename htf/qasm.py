"""HTF §4-F — OpenQASM 2.0 quantum circuit interoperability.

Provides:
* Standard gate matrix library (H, X, Y, Z, S, T, CNOT, CZ, SWAP, Rx, Ry, Rz).
* ``circuit_to_qasm``  — export a gate list to a QASM 2.0 string.
* ``qasm_to_circuit``  — parse a QASM 2.0 string to a gate list.
* ``circuit_unitary``  — simulate a circuit by sequential matrix composition.
* ``circuit_to_diagram`` — wrap each gate as an HTF ``Box`` in a ``Diagram``.

Honest scope [工程]
-------------------
* Targets the QASM 2.0 grammar only; QASM 3.0 and OpenQASM extensions are out.
* ``gate`` definitions, ``if`` statements, and opaque gates are not parsed.
* Simulation is via dense matrix products (exponential cost); for large circuits
  use a dedicated simulator.
* No noise models; open-system circuits are ``[研究]``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .topology import Box, Diagram, Wire

# ─────────────────────── Gate matrix library ──────────────────────────────

_I2 = np.eye(2, dtype=complex)

def _rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

def _ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)

def _rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-0.5j * theta), 0],
                     [0, np.exp(0.5j * theta)]], dtype=complex)

# Fixed single-qubit gates
_SINGLE = {
    "h": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
    "s": np.array([[1, 0], [0, 1j]], dtype=complex),
    "sdg": np.array([[1, 0], [0, -1j]], dtype=complex),
    "t": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
    "tdg": np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex),
    "id": _I2.copy(),
}

# Fixed two-qubit gates (standard qubit order: q0 ⊗ q1)
_TWO: dict[str, np.ndarray] = {
    "cx": np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0]], dtype=complex),
    "cz": np.diag([1, 1, 1, -1]).astype(complex),
    "swap": np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1]], dtype=complex),
}

# Parameterised single-qubit gates
_PARAM_SINGLE = {"rx": _rx, "ry": _ry, "rz": _rz}


def get_gate_matrix(name: str, params: list[float] | None = None) -> np.ndarray:
    """Return the unitary matrix for a named gate.

    Parameters
    ----------
    name   : gate name (case-insensitive), e.g. ``"h"``, ``"cx"``, ``"rx"``.
    params : rotation angles in radians (required for ``rx``, ``ry``, ``rz``).

    Returns
    -------
    Unitary matrix as complex ndarray (2×2 for single-qubit, 4×4 for two-qubit).

    Raises
    ------
    ValueError
        If the gate name is unknown or the wrong number of parameters is given.
    """
    n = name.lower()
    if n in _SINGLE:
        return _SINGLE[n].copy()
    if n in _TWO:
        return _TWO[n].copy()
    if n in _PARAM_SINGLE:
        if not params:
            raise ValueError(f"Gate '{n}' requires exactly one angle parameter.")
        return _PARAM_SINGLE[n](float(params[0]))
    raise ValueError(f"Unknown gate '{name}'.")


# ─────────────────────── Gate dataclass ───────────────────────────────────

@dataclass
class Gate:
    """A single gate application in a circuit.

    Attributes
    ----------
    name    : gate name (lower-case), e.g. ``"h"``, ``"cx"``.
    qubits  : list of qubit indices this gate acts on.
    params  : rotation angles in radians (empty for fixed gates).
    """
    name:   str
    qubits: list[int]
    params: list[float] = field(default_factory=list)

    def matrix(self) -> np.ndarray:
        """Return the unitary matrix for this gate application."""
        return get_gate_matrix(self.name, self.params)


# ─────────────────────── QASM 2.0 export ──────────────────────────────────

def circuit_to_qasm(gates: list[Gate], n_qubits: int) -> str:
    """Serialize a gate list to a QASM 2.0 string.

    Parameters
    ----------
    gates    : ordered list of :class:`Gate` objects.
    n_qubits : number of qubits in the register.

    Returns
    -------
    A valid QASM 2.0 string (includes the standard header).

    Notes
    -----
    Only the standard gate set (h, x, y, z, s, sdg, t, tdg, id, cx, cz, swap,
    rx, ry, rz) is emitted.  Custom matrices are not serializable via this path.
    """
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{n_qubits}];",
        f"creg c[{n_qubits}];",
        "",
    ]
    for g in gates:
        qs = ", ".join(f"q[{qi}]" for qi in g.qubits)
        if g.params:
            ps = ", ".join(_fmt_angle(p) for p in g.params)
            lines.append(f"{g.name}({ps}) {qs};")
        else:
            lines.append(f"{g.name} {qs};")
    return "\n".join(lines) + "\n"


def _fmt_angle(theta: float) -> str:
    """Format an angle for QASM (use pi notation when close to rational multiples)."""
    for num in range(1, 9):
        for den in range(1, 9):
            v = np.pi * num / den
            if abs(theta - v) < 1e-10:
                return f"{num}*pi/{den}" if den > 1 else f"{num}*pi"
            if abs(theta + v) < 1e-10:
                return f"-{num}*pi/{den}" if den > 1 else f"-{num}*pi"
    return f"{theta:.10g}"


# ─────────────────────── QASM 2.0 import ──────────────────────────────────

# Regex patterns for parsing
_RE_GATE_NO_PARAM  = re.compile(
    r"^([a-z][a-z0-9_]*)\s+(q\[\d+\](?:\s*,\s*q\[\d+\])*)\s*;$"
)
_RE_GATE_WITH_PARAM = re.compile(
    r"^([a-z][a-z0-9_]*)\s*\(([^)]*)\)\s+(q\[\d+\](?:\s*,\s*q\[\d+\])*)\s*;$"
)
_RE_QUBIT = re.compile(r"q\[(\d+)\]")


def qasm_to_circuit(qasm_src: str) -> list[Gate]:
    """Parse a QASM 2.0 string and return a list of :class:`Gate` objects.

    Parameters
    ----------
    qasm_src : QASM 2.0 source string.

    Returns
    -------
    Ordered list of :class:`Gate` objects.  Header, register declarations,
    and measurement instructions are silently ignored.

    Raises
    ------
    ValueError
        If a gate line cannot be parsed.
    """
    gates: list[Gate] = []
    for raw in qasm_src.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        # Skip known non-gate lines
        if any(line.startswith(p) for p in (
            "OPENQASM", "include", "qreg", "creg", "measure", "barrier", "reset",
        )):
            continue

        m = _RE_GATE_WITH_PARAM.match(line)
        if m:
            name   = m.group(1).lower()
            params = [_parse_angle(s.strip()) for s in m.group(2).split(",")]
            qubits = [int(x) for x in _RE_QUBIT.findall(m.group(3))]
            gates.append(Gate(name=name, qubits=qubits, params=params))
            continue

        m = _RE_GATE_NO_PARAM.match(line)
        if m:
            name   = m.group(1).lower()
            qubits = [int(x) for x in _RE_QUBIT.findall(m.group(2))]
            gates.append(Gate(name=name, qubits=qubits, params=[]))
            continue

        # If we reach here, the line is unrecognised — raise
        raise ValueError(f"Cannot parse QASM gate line: {line!r}")

    return gates


def _parse_angle(expr: str) -> float:
    """Evaluate a QASM angle expression (e.g. ``"pi/2"``, ``"3*pi/4"``)."""
    expr = expr.replace("pi", str(np.pi))
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))
    except Exception as exc:
        raise ValueError(f"Cannot evaluate angle expression: {expr!r}") from exc


# ─────────────────────── Circuit simulation ───────────────────────────────

def circuit_unitary(gates: list[Gate], n_qubits: int) -> np.ndarray:
    """Return the full unitary matrix of the circuit by sequential composition.

    Parameters
    ----------
    gates    : ordered list of :class:`Gate` objects.
    n_qubits : total number of qubits.

    Returns
    -------
    Complex unitary matrix of shape (2**n_qubits, 2**n_qubits).

    Notes
    -----
    Two-qubit gates are embedded via tensor product with identity on all other
    qubits using the ``_embed`` helper.  Assumes big-endian qubit ordering
    consistent with HTF lattice conventions.
    """
    dim = 2 ** n_qubits
    U   = np.eye(dim, dtype=complex)
    for g in gates:
        Ug = _embed_gate(g, n_qubits)
        U  = Ug @ U
    return U


def _embed_gate(g: Gate, n_qubits: int) -> np.ndarray:
    """Embed a 1- or 2-qubit gate into the full n-qubit space."""
    mat = g.matrix()
    dim = 2 ** n_qubits

    if len(g.qubits) == 1:
        q = g.qubits[0]
        ops: list[np.ndarray] = []
        for i in range(n_qubits):
            ops.append(mat if i == q else np.eye(2, dtype=complex))
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    if len(g.qubits) == 2:
        q0, q1 = g.qubits
        # Build full unitary by row/column selection
        full = np.zeros((dim, dim), dtype=complex)
        for col in range(dim):
            bits = [(col >> (n_qubits - 1 - i)) & 1 for i in range(n_qubits)]
            b0, b1 = bits[q0], bits[q1]
            gate_col = b0 * 2 + b1

            for gate_row in range(4):
                amp = mat[gate_row, gate_col]
                if amp == 0:
                    continue
                r0 = (gate_row >> 1) & 1
                r1 = gate_row & 1
                new_bits = bits.copy()
                new_bits[q0] = r0
                new_bits[q1] = r1
                row = sum(b << (n_qubits - 1 - i) for i, b in enumerate(new_bits))
                full[row, col] = amp
        return full

    raise ValueError(f"Gates acting on {len(g.qubits)} qubits are not supported.")


# ─────────────────────── HTF diagram bridge ───────────────────────────────

def circuit_to_diagram(
    gates: list[Gate],
    n_qubits: int,
    adjacent_only: bool = False,
) -> Diagram:
    """Wrap each gate as an HTF ``Box`` and compose them sequentially.

    Each gate becomes a ``Box`` whose domain and codomain are the qubit wires
    it acts on, embedded in the full wire tensor via identity boxes on the
    remaining wires.  The result is a ``Diagram`` in which the gates are
    composed left-to-right (time flows right).

    Parameters
    ----------
    gates         : ordered list of :class:`Gate` objects.
    n_qubits      : total number of qubits (used to name wires ``q0``, …).
    adjacent_only : if True, non-adjacent 2-qubit gates fall back to a
                    full-width Box (old behaviour).  If False (default) they
                    are decomposed via SWAP gates so that the gate acts on
                    adjacent wires (structurally exact).

    Returns
    -------
    An HTF :class:`~htf.topology.Diagram`.
    """
    from .topology import Id

    wires = [Wire(f"q{i}", 2) for i in range(n_qubits)]
    all_wires = tuple(wires)

    def _swap_layer(i: int, suffix: str) -> Diagram:
        """SWAP box on wires i and i+1, identity elsewhere."""
        lbl = f"swap_{i}_{i+1}{suffix}"
        box = Box(lbl, (wires[i], wires[i + 1]), (wires[i], wires[i + 1]))
        return Id(tuple(wires[:i])) @ box @ Id(tuple(wires[i + 2:]))

    def _layer(g: Gate, idx: int) -> Diagram:
        label = g.name
        if g.params:
            label += "(" + ",".join(f"{p:.4g}" for p in g.params) + ")"
        label += f"_{idx}"

        if len(g.qubits) == 1:
            qi  = g.qubits[0]
            box = Box(label, (wires[qi],), (wires[qi],))
            return Id(tuple(wires[:qi])) @ box @ Id(tuple(wires[qi + 1:]))

        if len(g.qubits) == 2:
            qa, qb = min(g.qubits[0], g.qubits[1]), max(g.qubits[0], g.qubits[1])
            if qb == qa + 1:
                box = Box(label, (wires[qa], wires[qb]), (wires[qa], wires[qb]))
                return Id(tuple(wires[:qa])) @ box @ Id(tuple(wires[qb + 1:]))

            # Non-adjacent qubits
            if adjacent_only:
                return Box(label, all_wires, all_wires)

            # SWAP decomposition: bubble qb left to qa+1, apply, bubble back
            sfx = f"_{idx}"
            layers_out: list[Diagram] = []
            # Forward SWAPs: qb → qa+1
            for i in range(qb - 1, qa, -1):
                layers_out.append(_swap_layer(i - 1, sfx + "_f"))
            # Gate at (qa, qa+1)
            gbox = Box(label, (wires[qa], wires[qa + 1]), (wires[qa], wires[qa + 1]))
            layers_out.append(Id(tuple(wires[:qa])) @ gbox @ Id(tuple(wires[qa + 2:])))
            # Reverse SWAPs: qa+1 → qb
            for i in range(qa + 1, qb):
                layers_out.append(_swap_layer(i, sfx + "_r"))
            result = layers_out[0]
            for l in layers_out[1:]:
                result = result >> l
            return result

        raise ValueError(f"circuit_to_diagram: gates on {len(g.qubits)} qubits not supported.")

    if not gates:
        return Id(all_wires)

    layers = [_layer(g, i) for i, g in enumerate(gates)]
    result = layers[0]
    for layer in layers[1:]:
        result = result >> layer
    return result
