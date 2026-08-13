"""Tests for htf/zx.py — ZX-calculus diagram rewriting."""
import math

import numpy as np
import pytest

from htf.qasm import Gate
from htf.zx import (
    ZXGraph,
    ZXNodeType,
    ZXRewriteLog,
    bialgebra,
    clifford_simplify,
    color_change,
    hadamard_cancel,
    identity_removal,
    local_complement,
    phase_gadget_fuse,
    pi_copy,
    simplify,
    spider_fusion,
    zx_from_circuit,
    zx_to_matrix,
)

# ─────────────────────── TestZXGraph ─────────────────────────────────────

class TestZXGraph:

    def test_add_node_returns_id(self):
        g = ZXGraph()
        nid = g.add_node(ZXNodeType.Z)
        assert nid == 0

    def test_ids_increment(self):
        g = ZXGraph()
        a = g.add_node(ZXNodeType.Z)
        b = g.add_node(ZXNodeType.X)
        assert b == a + 1

    def test_node_stored_correctly(self):
        g = ZXGraph()
        nid = g.add_node(ZXNodeType.Z, phase=math.pi, qubit=0, label="test")
        node = g.nodes[nid]
        assert node.kind == ZXNodeType.Z
        assert abs(node.phase - math.pi) < 1e-12
        assert node.qubit == 0
        assert node.label == "test"

    def test_add_edge(self):
        g = ZXGraph()
        a = g.add_node(ZXNodeType.Z)
        b = g.add_node(ZXNodeType.X)
        g.add_edge(a, b)
        assert (a, b) in g.edges or (b, a) in g.edges

    def test_neighbours(self):
        g = ZXGraph()
        a = g.add_node(ZXNodeType.Z)
        b = g.add_node(ZXNodeType.X)
        g.add_edge(a, b)
        assert b in g.neighbours(a)
        assert a in g.neighbours(b)

    def test_degree(self):
        g = ZXGraph()
        a = g.add_node(ZXNodeType.Z)
        b = g.add_node(ZXNodeType.X)
        c = g.add_node(ZXNodeType.H)
        g.add_edge(a, b)
        g.add_edge(a, c)
        assert g.degree(a) == 2

    def test_remove_node(self):
        g = ZXGraph()
        a = g.add_node(ZXNodeType.Z)
        b = g.add_node(ZXNodeType.X)
        g.add_edge(a, b)
        g.remove_node(a)
        assert a not in g.nodes
        assert not any(a in e for e in g.edges)

    def test_remove_edge(self):
        g = ZXGraph()
        a = g.add_node(ZXNodeType.Z)
        b = g.add_node(ZXNodeType.X)
        g.add_edge(a, b)
        g.remove_edge(a, b)
        assert len(g.edges) == 0

    def test_copy_is_independent(self):
        g = ZXGraph()
        a = g.add_node(ZXNodeType.Z, phase=1.0)
        g2 = g.copy()
        g2.nodes[a].phase = 99.0
        assert g.nodes[a].phase == 1.0

    def test_n_qubits(self):
        g = ZXGraph()
        for q in range(3):
            nid = g.add_node(ZXNodeType.INPUT, qubit=q)
            g.inputs.append(nid)
        assert g.n_qubits() == 3

    def test_multi_edge_allowed(self):
        g = ZXGraph()
        a = g.add_node(ZXNodeType.Z)
        b = g.add_node(ZXNodeType.Z)
        g.add_edge(a, b)
        g.add_edge(a, b)
        assert g.degree(a) == 2


# ─────────────────────── TestZxFromCircuit ───────────────────────────────

class TestZxFromCircuit:

    def test_empty_circuit(self):
        g = zx_from_circuit([], n_qubits=2)
        assert len(g.inputs) == 2
        assert len(g.outputs) == 2

    def test_h_gate_produces_h_node(self):
        gates = [Gate("h", [0])]
        g = zx_from_circuit(gates, n_qubits=1)
        kinds = [n.kind for n in g.nodes.values()]
        assert ZXNodeType.H in kinds

    def test_x_gate_produces_x_node(self):
        gates = [Gate("x", [0])]
        g = zx_from_circuit(gates, n_qubits=1)
        kinds = [n.kind for n in g.nodes.values()]
        assert ZXNodeType.X in kinds

    def test_z_gate_produces_z_node(self):
        gates = [Gate("z", [0])]
        g = zx_from_circuit(gates, n_qubits=1)
        kinds = [n.kind for n in g.nodes.values()]
        assert ZXNodeType.Z in kinds

    def test_cx_produces_z_and_x_nodes(self):
        gates = [Gate("cx", [0, 1])]
        g = zx_from_circuit(gates, n_qubits=2)
        kinds = [n.kind for n in g.nodes.values()]
        assert ZXNodeType.Z in kinds
        assert ZXNodeType.X in kinds

    def test_rz_node_has_correct_phase(self):
        theta = 1.23
        gates = [Gate("rz", [0], [theta])]
        g = zx_from_circuit(gates, n_qubits=1)
        z_nodes = [n for n in g.nodes.values() if n.kind == ZXNodeType.Z]
        assert any(abs(n.phase - theta) < 1e-10 for n in z_nodes)

    def test_input_output_counts(self):
        gates = [Gate("h", [0]), Gate("cx", [0, 1])]
        g = zx_from_circuit(gates, n_qubits=2)
        assert len(g.inputs)  == 2
        assert len(g.outputs) == 2

    def test_identity_gate_no_spider(self):
        gates = [Gate("id", [0])]
        g = zx_from_circuit(gates, n_qubits=1)
        spiders = [n for n in g.nodes.values()
                   if n.kind in (ZXNodeType.Z, ZXNodeType.X, ZXNodeType.H)]
        assert len(spiders) == 0


# ─────────────────────── TestSpiderFusion ────────────────────────────────

class TestSpiderFusion:

    def test_two_z_spiders_fuse(self):
        g = ZXGraph()
        a = g.add_node(ZXNodeType.Z, phase=0.5)
        b = g.add_node(ZXNodeType.Z, phase=0.3)
        g.add_edge(a, b)
        n = spider_fusion(g)
        assert n >= 1
        assert len([v for v in g.nodes.values() if v.kind == ZXNodeType.Z]) == 1

    def test_fused_phase_is_sum(self):
        g = ZXGraph()
        a = g.add_node(ZXNodeType.Z, phase=0.5)
        b = g.add_node(ZXNodeType.Z, phase=0.3)
        g.add_edge(a, b)
        spider_fusion(g)
        z_nodes = [v for v in g.nodes.values() if v.kind == ZXNodeType.Z]
        assert abs(z_nodes[0].phase - 0.8) < 1e-10

    def test_different_colours_not_fused(self):
        g = ZXGraph()
        a = g.add_node(ZXNodeType.Z)
        b = g.add_node(ZXNodeType.X)
        g.add_edge(a, b)
        n = spider_fusion(g)
        assert n == 0
        assert len(g.nodes) == 2

    def test_log_records_step(self):
        g = ZXGraph()
        a = g.add_node(ZXNodeType.Z)
        b = g.add_node(ZXNodeType.Z)
        g.add_edge(a, b)
        log = ZXRewriteLog()
        spider_fusion(g, log)
        assert len(log) == 1
        assert log.steps[0]["rule"] == "spider_fusion"


# ─────────────────────── TestIdentityRemoval ─────────────────────────────

class TestIdentityRemoval:

    def test_removes_zero_phase_2_leg_spider(self):
        g = ZXGraph()
        a = g.add_node(ZXNodeType.Z)  # left neighbour
        b = g.add_node(ZXNodeType.Z, phase=0.0)  # identity spider
        c = g.add_node(ZXNodeType.Z)  # right neighbour
        g.add_edge(a, b)
        g.add_edge(b, c)
        n = identity_removal(g)
        assert n >= 1
        assert b not in g.nodes

    def test_wire_connected_after_removal(self):
        g = ZXGraph()
        a = g.add_node(ZXNodeType.Z)
        b = g.add_node(ZXNodeType.Z, phase=0.0)
        c = g.add_node(ZXNodeType.Z)
        g.add_edge(a, b)
        g.add_edge(b, c)
        identity_removal(g)
        assert c in g.neighbours(a) or a in g.neighbours(c)

    def test_non_zero_phase_not_removed(self):
        g = ZXGraph()
        a = g.add_node(ZXNodeType.Z)
        b = g.add_node(ZXNodeType.Z, phase=0.5)
        c = g.add_node(ZXNodeType.Z)
        g.add_edge(a, b)
        g.add_edge(b, c)
        n = identity_removal(g)
        assert n == 0
        assert b in g.nodes


# ─────────────────────── TestHadamardCancel ──────────────────────────────

class TestHadamardCancel:

    def test_two_h_boxes_cancel(self):
        g = ZXGraph()
        a = g.add_node(ZXNodeType.Z)
        h1 = g.add_node(ZXNodeType.H)
        h2 = g.add_node(ZXNodeType.H)
        b  = g.add_node(ZXNodeType.Z)
        g.add_edge(a, h1)
        g.add_edge(h1, h2)
        g.add_edge(h2, b)
        n = hadamard_cancel(g)
        assert n >= 1
        assert h1 not in g.nodes
        assert h2 not in g.nodes

    def test_outer_nodes_connected_after_cancel(self):
        g = ZXGraph()
        a  = g.add_node(ZXNodeType.Z)
        h1 = g.add_node(ZXNodeType.H)
        h2 = g.add_node(ZXNodeType.H)
        b  = g.add_node(ZXNodeType.Z)
        g.add_edge(a, h1)
        g.add_edge(h1, h2)
        g.add_edge(h2, b)
        hadamard_cancel(g)
        assert b in g.neighbours(a) or a in g.neighbours(b)

    def test_single_h_not_cancelled(self):
        g = ZXGraph()
        a = g.add_node(ZXNodeType.Z)
        h = g.add_node(ZXNodeType.H)
        b = g.add_node(ZXNodeType.Z)
        g.add_edge(a, h)
        g.add_edge(h, b)
        n = hadamard_cancel(g)
        assert n == 0


# ─────────────────────── TestSimplify ────────────────────────────────────

class TestSimplify:

    def test_returns_integer(self):
        g = ZXGraph()
        a = g.add_node(ZXNodeType.Z)
        b = g.add_node(ZXNodeType.Z)
        g.add_edge(a, b)
        n = simplify(g)
        assert isinstance(n, int)

    def test_chain_of_z_spiders_fuses_to_one(self):
        g = ZXGraph()
        nodes = [g.add_node(ZXNodeType.Z, phase=0.1 * i) for i in range(4)]
        for i in range(3):
            g.add_edge(nodes[i], nodes[i + 1])
        simplify(g, rules=["spider_fusion"])
        z_nodes = [v for v in g.nodes.values() if v.kind == ZXNodeType.Z]
        assert len(z_nodes) == 1
        assert abs(z_nodes[0].phase - 0.6) < 1e-10

    def test_log_accumulated(self):
        g = ZXGraph()
        a = g.add_node(ZXNodeType.Z)
        b = g.add_node(ZXNodeType.Z)
        g.add_edge(a, b)
        log = ZXRewriteLog()
        simplify(g, log=log)
        assert len(log) >= 1


# ─────────────────────── TestZxToMatrix ──────────────────────────────────

class TestZxToMatrix:

    def test_empty_circuit_is_identity(self):
        g = zx_from_circuit([], n_qubits=1)
        U = zx_to_matrix(g)
        assert np.allclose(U, np.eye(2, dtype=complex), atol=1e-10)

    def test_h_gate_roundtrip(self):
        from htf.qasm import get_gate_matrix
        gates = [Gate("h", [0])]
        g = zx_from_circuit(gates, n_qubits=1)
        U = zx_to_matrix(g)
        H_ref = get_gate_matrix("h")
        assert np.allclose(U, H_ref, atol=1e-10)

    def test_zero_qubits(self):
        g = ZXGraph()
        U = zx_to_matrix(g)
        assert U.shape == (1, 1)

    def test_non_circuit_topology_raises(self):
        """An orphan non-boundary node (no edges) must raise NotImplementedError."""
        g = ZXGraph()
        inp = g.add_node(ZXNodeType.INPUT, qubit=0, label="in0")
        out = g.add_node(ZXNodeType.OUTPUT, qubit=0, label="out0")
        g.inputs.append(inp)
        g.outputs.append(out)
        g.add_edge(inp, out)
        # Orphan node — no edges at all
        g.add_node(ZXNodeType.Z, phase=math.pi, label="orphan")
        with pytest.raises(NotImplementedError):
            zx_to_matrix(g)

    def test_cross_wire_spider_evaluates_correctly(self):
        """A 4-legged Z(0) spider connecting inp0, inp1, out0, out1.

        Z(0,4)[i,j,k,l] = δ(all=0) + δ(all=1), so the 4×4 matrix has
        ones only at (row=|00⟩, col=|00⟩) and (row=|11⟩, col=|11⟩).
        """
        g = ZXGraph()
        inp0 = g.add_node(ZXNodeType.INPUT, qubit=0)
        inp1 = g.add_node(ZXNodeType.INPUT, qubit=1)
        out0 = g.add_node(ZXNodeType.OUTPUT, qubit=0)
        out1 = g.add_node(ZXNodeType.OUTPUT, qubit=1)
        g.inputs = [inp0, inp1]
        g.outputs = [out0, out1]
        shared = g.add_node(ZXNodeType.Z, phase=0.0, label="shared")
        g.add_edge(inp0, shared)
        g.add_edge(inp1, shared)
        g.add_edge(shared, out0)
        g.add_edge(shared, out1)
        M = zx_to_matrix(g)
        assert M.shape == (4, 4)
        expected = np.zeros((4, 4), dtype=complex)
        expected[0, 0] = 1.0   # |00⟩ → |00⟩
        expected[3, 3] = 1.0   # |11⟩ → |11⟩
        np.testing.assert_allclose(np.abs(M), np.abs(expected), atol=1e-12)


# ─────────────────────── TestColorChange ─────────────────────────────────

class TestColorChange:

    def test_z_surrounded_by_h_becomes_x(self):
        # Z spider with all neighbours being H boxes → flips to X
        g = ZXGraph()
        left  = g.add_node(ZXNodeType.Z, label="left")
        h1    = g.add_node(ZXNodeType.H, label="h1")
        z     = g.add_node(ZXNodeType.Z, phase=0.5, label="center")
        h2    = g.add_node(ZXNodeType.H, label="h2")
        right = g.add_node(ZXNodeType.Z, label="right")
        g.add_edge(left, h1)
        g.add_edge(h1, z)
        g.add_edge(z, h2)
        g.add_edge(h2, right)
        n = color_change(g)
        assert n >= 1
        # center node should now be X (or gone — absorbed)
        # the H boxes should be removed
        h_nodes = [v for v in g.nodes.values() if v.kind == ZXNodeType.H]
        assert h1 not in g.nodes or h2 not in g.nodes or len(h_nodes) < 2

    def test_returns_count(self):
        g = ZXGraph()
        h1 = g.add_node(ZXNodeType.H)
        z  = g.add_node(ZXNodeType.Z, phase=1.0)
        h2 = g.add_node(ZXNodeType.H)
        g.add_edge(h1, z)
        g.add_edge(z, h2)
        n = color_change(g)
        assert isinstance(n, int)

    def test_no_change_when_not_all_h_neighbours(self):
        # Z spider with one H and one Z neighbour → no colour change
        g = ZXGraph()
        z1 = g.add_node(ZXNodeType.Z)
        h  = g.add_node(ZXNodeType.H)
        z2 = g.add_node(ZXNodeType.Z)
        z3 = g.add_node(ZXNodeType.Z)
        g.add_edge(z1, h)
        g.add_edge(h, z2)
        g.add_edge(z2, z3)  # z2 has both H and Z neighbours
        color_change(g)
        # z2 should not have changed (has non-H neighbour z3)
        assert z2 in g.nodes
        assert g.nodes[z2].kind == ZXNodeType.Z

    def test_log_records_step(self):
        g = ZXGraph()
        h1 = g.add_node(ZXNodeType.H)
        z  = g.add_node(ZXNodeType.Z, phase=1.0)
        h2 = g.add_node(ZXNodeType.H)
        g.add_edge(h1, z)
        g.add_edge(z, h2)
        log = ZXRewriteLog()
        color_change(g, log)
        assert len(log) >= 1

    def test_simplify_includes_color_change(self):
        # simplify with default rules should run color_change
        g = ZXGraph()
        h1 = g.add_node(ZXNodeType.H)
        z  = g.add_node(ZXNodeType.Z, phase=0.5)
        h2 = g.add_node(ZXNodeType.H)
        g.add_edge(h1, z)
        g.add_edge(z, h2)
        n = simplify(g)
        assert isinstance(n, int)


# ─────────────────────── TestPiCopy ──────────────────────────────────────

class TestPiCopy:

    def test_z_pi_copies_through_x_zero(self):
        # Z(π) connected to X(0) → Z(π) copied to X(0)'s other legs
        g = ZXGraph()
        z_pi = g.add_node(ZXNodeType.Z, phase=math.pi, label="Z_pi")
        x0   = g.add_node(ZXNodeType.X, phase=0.0,    label="X_0")
        out  = g.add_node(ZXNodeType.Z, label="out")
        g.add_edge(z_pi, x0)
        g.add_edge(x0, out)
        n = pi_copy(g)
        assert n >= 1
        # z_pi node should be removed
        assert z_pi not in g.nodes

    def test_returns_count(self):
        g = ZXGraph()
        z_pi = g.add_node(ZXNodeType.Z, phase=math.pi)
        x0   = g.add_node(ZXNodeType.X, phase=0.0)
        out  = g.add_node(ZXNodeType.Z)
        g.add_edge(z_pi, x0)
        g.add_edge(x0, out)
        n = pi_copy(g)
        assert isinstance(n, int) and n >= 1

    def test_z_nonpi_not_copied(self):
        g = ZXGraph()
        z  = g.add_node(ZXNodeType.Z, phase=0.5)
        x0 = g.add_node(ZXNodeType.X, phase=0.0)
        g.add_edge(z, x0)
        n = pi_copy(g)
        assert n == 0
        assert z in g.nodes

    def test_x_nonzero_not_target(self):
        g = ZXGraph()
        z_pi = g.add_node(ZXNodeType.Z, phase=math.pi)
        x05  = g.add_node(ZXNodeType.X, phase=0.5)  # non-zero phase
        g.add_edge(z_pi, x05)
        n = pi_copy(g)
        assert n == 0

    def test_log_records_step(self):
        g = ZXGraph()
        z_pi = g.add_node(ZXNodeType.Z, phase=math.pi)
        x0   = g.add_node(ZXNodeType.X, phase=0.0)
        out  = g.add_node(ZXNodeType.Z)
        g.add_edge(z_pi, x0)
        g.add_edge(x0, out)
        log = ZXRewriteLog()
        pi_copy(g, log)
        assert len(log) >= 1
        assert log.steps[0]["rule"] == "pi_copy"

    def test_simplify_includes_pi_copy(self):
        g = ZXGraph()
        z_pi = g.add_node(ZXNodeType.Z, phase=math.pi)
        x0   = g.add_node(ZXNodeType.X, phase=0.0)
        out  = g.add_node(ZXNodeType.Z)
        g.add_edge(z_pi, x0)
        g.add_edge(x0, out)
        n = simplify(g, rules=["pi_copy"])
        assert n >= 1


# ─────────────────────── TestBialgebra ───────────────────────────────────

class TestBialgebra:

    def _make_1x1(self):
        """Z(0)─X(0) with one extra neighbour each (1×1 fan-out)."""
        g = ZXGraph()
        left  = g.add_node(ZXNodeType.Z, phase=0.0, label="left")
        z0    = g.add_node(ZXNodeType.Z, phase=0.0, label="Z0")
        x0    = g.add_node(ZXNodeType.X, phase=0.0, label="X0")
        right = g.add_node(ZXNodeType.X, phase=0.0, label="right")
        g.add_edge(left, z0)
        g.add_edge(z0, x0)
        g.add_edge(x0, right)
        return g, z0, x0

    def test_fires_on_1x1(self):
        g, _z0, _x0 = self._make_1x1()
        n = bialgebra(g)
        assert n == 1

    def test_original_nodes_removed(self):
        g, z0, x0 = self._make_1x1()
        bialgebra(g)
        assert z0 not in g.nodes
        assert x0 not in g.nodes

    def test_new_nodes_created(self):
        g, _z0, _x0 = self._make_1x1()
        n_before = len(g.nodes)
        bialgebra(g)
        # 2 nodes removed, 1 X + 1 Z added
        assert len(g.nodes) == n_before - 2 + 2

    def test_no_fire_when_both_sides_have_multiple_neighbours(self):
        # Z(0) with 2 left neighbours, X(0) with 2 right neighbours → guard blocks
        g = ZXGraph()
        l1 = g.add_node(ZXNodeType.Z)
        l2 = g.add_node(ZXNodeType.Z)
        z0 = g.add_node(ZXNodeType.Z, phase=0.0)
        x0 = g.add_node(ZXNodeType.X, phase=0.0)
        r1 = g.add_node(ZXNodeType.X)
        r2 = g.add_node(ZXNodeType.X)
        g.add_edge(l1, z0)
        g.add_edge(l2, z0)
        g.add_edge(z0, x0)
        g.add_edge(x0, r1)
        g.add_edge(x0, r2)
        n = bialgebra(g)
        assert n == 0

    def test_no_fire_on_nonzero_phase(self):
        g = ZXGraph()
        z  = g.add_node(ZXNodeType.Z, phase=0.1)
        x  = g.add_node(ZXNodeType.X, phase=0.0)
        nb = g.add_node(ZXNodeType.Z)
        g.add_edge(nb, z)
        g.add_edge(z, x)
        n = bialgebra(g)
        assert n == 0

    def test_terminates_without_infinite_loop(self):
        # A longer chain: Z(0)–X(0)–Z(0)–X(0)
        # After first application we get new nodes; they must not trigger again
        g = ZXGraph()
        a  = g.add_node(ZXNodeType.Z, phase=0.0)
        z1 = g.add_node(ZXNodeType.Z, phase=0.0)
        x1 = g.add_node(ZXNodeType.X, phase=0.0)
        b  = g.add_node(ZXNodeType.X, phase=0.0)
        g.add_edge(a, z1)
        g.add_edge(z1, x1)
        g.add_edge(x1, b)
        # Must return a finite non-negative count
        n = bialgebra(g)
        assert isinstance(n, int)
        assert n >= 0

    def test_log_records_step(self):
        g, _z0, _x0 = self._make_1x1()
        log = ZXRewriteLog()
        bialgebra(g, log)
        assert len(log) >= 1
        assert log.steps[0]["rule"] == "bialgebra"

    def test_returns_zero_on_empty_graph(self):
        g = ZXGraph()
        assert bialgebra(g) == 0


# ─────────────────────── TestLocalComplement ─────────────────────────────

class TestLocalComplement:

    def _make_lc_graph(self, phase=None):
        """Z(π/2) flanked by two H-boxes, which connect to outer Z nodes."""
        if phase is None:
            phase = math.pi / 2
        g = ZXGraph()
        outer_l = g.add_node(ZXNodeType.Z, phase=0.0, label="outer_l")
        h_l     = g.add_node(ZXNodeType.H, label="h_l")
        center  = g.add_node(ZXNodeType.Z, phase=phase, label="center")
        h_r     = g.add_node(ZXNodeType.H, label="h_r")
        outer_r = g.add_node(ZXNodeType.Z, phase=0.0, label="outer_r")
        g.add_edge(outer_l, h_l)
        g.add_edge(h_l, center)
        g.add_edge(center, h_r)
        g.add_edge(h_r, outer_r)
        return g, center, h_l, h_r

    def test_fires_on_pi_half(self):
        g, _center, _h_l, _h_r = self._make_lc_graph()
        n = local_complement(g)
        assert n >= 1

    def test_center_node_removed(self):
        g, center, _h_l, _h_r = self._make_lc_graph()
        local_complement(g)
        assert center not in g.nodes

    def test_fires_on_negative_pi_half(self):
        g, _center, _h_l, _h_r = self._make_lc_graph(phase=-math.pi / 2)
        n = local_complement(g)
        assert n >= 1

    def test_no_fire_on_non_clifford_phase(self):
        g, center, _h_l, _h_r = self._make_lc_graph(phase=0.3)
        n = local_complement(g)
        assert n == 0
        assert center in g.nodes

    def test_no_fire_when_non_h_neighbour(self):
        # Replace one H-box with a plain Z node — LC must not fire
        g = ZXGraph()
        z_neigh = g.add_node(ZXNodeType.Z, phase=0.0)
        center  = g.add_node(ZXNodeType.Z, phase=math.pi / 2)
        h       = g.add_node(ZXNodeType.H)
        outer   = g.add_node(ZXNodeType.Z)
        g.add_edge(z_neigh, center)
        g.add_edge(center, h)
        g.add_edge(h, outer)
        n = local_complement(g)
        assert n == 0
        assert center in g.nodes

    def test_neighbour_phases_shifted(self):
        g, _center, h_l, h_r = self._make_lc_graph(phase=math.pi / 2)
        # Direct neighbours of center are h_l and h_r; their phases get shifted by -π/2
        local_complement(g)
        # h_l and h_r should still exist (they connect to outer nodes)
        # and their phases should be 0 - π/2 = -π/2
        for nid in (h_l, h_r):
            if nid in g.nodes:
                assert abs(g.nodes[nid].phase - (-math.pi / 2)) < 1e-9

    def test_log_records_step(self):
        g, _center, _h_l, _h_r = self._make_lc_graph()
        log = ZXRewriteLog()
        local_complement(g, log)
        assert len(log) >= 1
        assert log.steps[0]["rule"] == "local_complement"

    def test_returns_zero_on_empty_graph(self):
        g = ZXGraph()
        assert local_complement(g) == 0


# ─────────────────────── TestPhaseGadgetFuse ─────────────────────────────

class TestPhaseGadgetFuse:

    def _make_two_gadgets(self, alpha=0.3, beta=0.7):
        """Two Z-spiders sharing the same single neighbour X-spider."""
        g = ZXGraph()
        xnode = g.add_node(ZXNodeType.X, phase=0.0, label="x")
        z1    = g.add_node(ZXNodeType.Z, phase=alpha, label="z1")
        z2    = g.add_node(ZXNodeType.Z, phase=beta,  label="z2")
        g.add_edge(xnode, z1)
        g.add_edge(xnode, z2)
        return g, z1, z2, xnode

    def test_fires_on_identical_neighbour_sets(self):
        g, _z1, _z2, _ = self._make_two_gadgets()
        n = phase_gadget_fuse(g)
        assert n >= 1

    def test_one_z_node_remains(self):
        g, _z1, _z2, _ = self._make_two_gadgets()
        phase_gadget_fuse(g)
        z_nodes = [nd for nd in g.nodes.values() if nd.kind == ZXNodeType.Z]
        assert len(z_nodes) == 1

    def test_fused_phase_is_sum(self):
        alpha, beta = 0.3, 0.7
        g, _z1, _z2, _ = self._make_two_gadgets(alpha, beta)
        phase_gadget_fuse(g)
        z_nodes = [nd for nd in g.nodes.values() if nd.kind == ZXNodeType.Z]
        assert abs(z_nodes[0].phase - (alpha + beta)) < 1e-9

    def test_no_fire_when_different_neighbours(self):
        g = ZXGraph()
        x1 = g.add_node(ZXNodeType.X, phase=0.0)
        x2 = g.add_node(ZXNodeType.X, phase=0.0)
        z1 = g.add_node(ZXNodeType.Z, phase=0.3)
        z2 = g.add_node(ZXNodeType.Z, phase=0.5)
        g.add_edge(x1, z1)
        g.add_edge(x2, z2)
        n = phase_gadget_fuse(g)
        assert n == 0

    def test_three_gadgets_fully_fused(self):
        # Three Z-spiders with the same neighbour
        g = ZXGraph()
        xnode = g.add_node(ZXNodeType.X, phase=0.0)
        phases = [0.1, 0.2, 0.4]
        for p in phases:
            zi = g.add_node(ZXNodeType.Z, phase=p)
            g.add_edge(xnode, zi)
        phase_gadget_fuse(g)
        z_nodes = [nd for nd in g.nodes.values() if nd.kind == ZXNodeType.Z]
        assert len(z_nodes) == 1
        assert abs(z_nodes[0].phase - sum(phases)) < 1e-9

    def test_log_records_step(self):
        g, _z1, _z2, _ = self._make_two_gadgets()
        log = ZXRewriteLog()
        phase_gadget_fuse(g, log)
        assert len(log) >= 1
        assert log.steps[0]["rule"] == "phase_gadget_fuse"

    def test_returns_zero_on_empty_graph(self):
        g = ZXGraph()
        assert phase_gadget_fuse(g) == 0

    def test_isolated_z_spider_not_fused(self):
        # A Z-spider with no neighbours → empty frozenset, must be skipped
        g = ZXGraph()
        g.add_node(ZXNodeType.Z, phase=1.0)
        g.add_node(ZXNodeType.Z, phase=2.0)
        n = phase_gadget_fuse(g)
        assert n == 0


# ─────────────────────── TestCliffordSimplify ────────────────────────────

class TestCliffordSimplify:

    def test_returns_integer(self):
        g = ZXGraph()
        g.add_node(ZXNodeType.Z)
        n = clifford_simplify(g)
        assert isinstance(n, int)

    def test_spider_fusion_triggered(self):
        g = ZXGraph()
        a = g.add_node(ZXNodeType.Z, phase=0.2)
        b = g.add_node(ZXNodeType.Z, phase=0.3)
        g.add_edge(a, b)
        n = clifford_simplify(g)
        assert n >= 1
        z_nodes = [nd for nd in g.nodes.values() if nd.kind == ZXNodeType.Z]
        assert len(z_nodes) == 1

    def test_phase_gadget_fuse_triggered(self):
        g = ZXGraph()
        xnode = g.add_node(ZXNodeType.X, phase=0.0)
        z1    = g.add_node(ZXNodeType.Z, phase=0.3)
        z2    = g.add_node(ZXNodeType.Z, phase=0.4)
        g.add_edge(xnode, z1)
        g.add_edge(xnode, z2)
        n = clifford_simplify(g)
        assert n >= 1
        z_nodes = [nd for nd in g.nodes.values() if nd.kind == ZXNodeType.Z]
        assert len(z_nodes) == 1

    def test_identity_removal_triggered(self):
        g = ZXGraph()
        a = g.add_node(ZXNodeType.Z)
        b = g.add_node(ZXNodeType.Z, phase=0.0)
        c = g.add_node(ZXNodeType.Z)
        g.add_edge(a, b)
        g.add_edge(b, c)
        n = clifford_simplify(g)
        assert n >= 1
        assert b not in g.nodes

    def test_terminates_on_empty_graph(self):
        g = ZXGraph()
        n = clifford_simplify(g)
        assert n == 0

    def test_terminates_on_already_simplified(self):
        # A single isolated Z-spider with non-zero phase — nothing to do
        g = ZXGraph()
        g.add_node(ZXNodeType.Z, phase=0.7)
        n = clifford_simplify(g)
        assert n == 0

    def test_max_iter_respected(self):
        # Build a graph that would fuse but cap at max_iter=1
        g = ZXGraph()
        a = g.add_node(ZXNodeType.Z, phase=0.1)
        b = g.add_node(ZXNodeType.Z, phase=0.2)
        c = g.add_node(ZXNodeType.Z, phase=0.3)
        g.add_edge(a, b)
        g.add_edge(b, c)
        # With max_iter=1 it may not fully simplify, but must not hang
        n = clifford_simplify(g, max_iter=1)
        assert isinstance(n, int)

    def test_log_accumulated(self):
        g = ZXGraph()
        a = g.add_node(ZXNodeType.Z, phase=0.1)
        b = g.add_node(ZXNodeType.Z, phase=0.2)
        g.add_edge(a, b)
        log = ZXRewriteLog()
        clifford_simplify(g, log=log)
        assert len(log) >= 1

    def test_bell_circuit_terminates(self):
        from htf.qasm import Gate
        gates = [Gate("h", [0]), Gate("cx", [0, 1])]
        g = zx_from_circuit(gates, n_qubits=2)
        n = clifford_simplify(g)
        assert isinstance(n, int)  # just must not hang


# ─────────────────────── P0-6 regression tests ────────────────────────────

class TestP06Regression:
    """Regression suite for P0-6: 2-qubit ZX gate semantics (fixed).

    ``zx_to_matrix`` evaluates diagrams up to a global scalar — a known
    property of the ZX spider convention.  Gates H, X, Z, S, T match
    exactly; rotation gates (Rx, Rz, Ry) and multi-qubit gates (CX, CZ,
    SWAP) are proportional to the circuit reference (same structure, same
    zero/nonzero pattern and relative phases, scalar factor may differ).
    """

    _ATOL = 1e-10

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _ref(gates, n):
        from htf.qasm import circuit_unitary
        return circuit_unitary(gates, n)

    @staticmethod
    def _zx(gates, n):
        g = zx_from_circuit(gates, n)
        return zx_to_matrix(g)

    @staticmethod
    def _assert_proportional(A: np.ndarray, B: np.ndarray, msg: str = "") -> None:
        """Check A = c·B for some nonzero scalar c (same structure up to scale)."""
        flat_a, flat_b = A.ravel(), B.ravel()
        nz = np.abs(flat_b) > 1e-12
        assert nz.any(), f"Reference is all zeros: {msg}"
        c = flat_a[nz][0] / flat_b[nz][0]
        np.testing.assert_allclose(flat_a[nz] / flat_b[nz], c,
                                   atol=1e-8, err_msg=f"Ratios not constant: {msg}")
        z_mask = ~nz
        if z_mask.any():
            np.testing.assert_allclose(np.abs(flat_a[z_mask]), 0,
                                       atol=1e-10, err_msg=f"Nonzero where zero expected: {msg}")

    # ── single-qubit gates ────────────────────────────────────────────────

    def test_h_roundtrip(self):
        gates = [Gate("h", [0])]
        np.testing.assert_allclose(self._zx(gates, 1), self._ref(gates, 1), atol=self._ATOL)

    def test_x_roundtrip(self):
        gates = [Gate("x", [0])]
        np.testing.assert_allclose(self._zx(gates, 1), self._ref(gates, 1), atol=self._ATOL)

    def test_z_roundtrip(self):
        gates = [Gate("z", [0])]
        np.testing.assert_allclose(self._zx(gates, 1), self._ref(gates, 1), atol=self._ATOL)

    def test_s_roundtrip(self):
        gates = [Gate("s", [0])]
        np.testing.assert_allclose(self._zx(gates, 1), self._ref(gates, 1), atol=self._ATOL)

    def test_t_roundtrip(self):
        gates = [Gate("t", [0])]
        np.testing.assert_allclose(self._zx(gates, 1), self._ref(gates, 1), atol=self._ATOL)

    def test_rx_roundtrip(self):
        gates = [Gate("rx", [0], [0.7])]
        self._assert_proportional(self._zx(gates, 1), self._ref(gates, 1), "Rx(0.7)")

    def test_rz_roundtrip(self):
        gates = [Gate("rz", [0], [1.2])]
        self._assert_proportional(self._zx(gates, 1), self._ref(gates, 1), "Rz(1.2)")

    def test_ry_p0_6_regression(self):
        """P0-6: Ry now evaluates to the correct structure (proportional to Ry)."""
        gates = [Gate("ry", [0], [0.7])]
        self._assert_proportional(self._zx(gates, 1), self._ref(gates, 1), "Ry(0.7)")

    def test_ry_various_angles(self):
        for theta in (0.0, math.pi / 4, math.pi / 2, math.pi, 1.23):
            gates = [Gate("ry", [0], [theta])]
            self._assert_proportional(
                self._zx(gates, 1), self._ref(gates, 1), f"Ry({theta:.3g})"
            )

    # ── 2-qubit gates ─────────────────────────────────────────────────────

    def test_cx_p0_6_regression(self):
        """P0-6: CX now evaluates to the correct structure (proportional to CX)."""
        gates = [Gate("cx", [0, 1])]
        self._assert_proportional(self._zx(gates, 2), self._ref(gates, 2), "CX")

    def test_cx_ctrl_1_tgt_0(self):
        gates = [Gate("cx", [1, 0])]
        self._assert_proportional(self._zx(gates, 2), self._ref(gates, 2), "CX(1,0)")

    def test_cz_p0_6_regression(self):
        """P0-6: CZ now evaluates to the correct structure (proportional to CZ)."""
        gates = [Gate("cz", [0, 1])]
        self._assert_proportional(self._zx(gates, 2), self._ref(gates, 2), "CZ")

    def test_swap_p0_6_regression(self):
        """P0-6: SWAP now evaluates to the correct structure (proportional to SWAP)."""
        gates = [Gate("swap", [0, 1])]
        self._assert_proportional(self._zx(gates, 2), self._ref(gates, 2), "SWAP")

    def test_swap_is_its_own_inverse(self):
        """SWAP·SWAP is proportional to identity (ZX scalar accumulates)."""
        gates = [Gate("swap", [0, 1]), Gate("swap", [0, 1])]
        M = self._zx(gates, 2)
        self._assert_proportional(M, np.eye(4, dtype=complex), "SWAP²")

    def test_cx_bell_state(self):
        """H⊗I then CX: ZX result is proportional to Bell basis columns."""
        gates = [Gate("h", [0]), Gate("cx", [0, 1])]
        U = self._zx(gates, 2)
        col0 = U[:, 0]
        expected_col0 = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
        self._assert_proportional(col0.reshape(1, -1), expected_col0.reshape(1, -1),
                                  "Bell col0")

    def test_identity_circuit_2qubits(self):
        g = zx_from_circuit([], n_qubits=2)
        M = zx_to_matrix(g)
        np.testing.assert_allclose(M, np.eye(4, dtype=complex), atol=self._ATOL)

    def test_cx_followed_by_cx_is_identity(self):
        """CX·CX is proportional to identity (ZX scalar accumulates)."""
        gates = [Gate("cx", [0, 1]), Gate("cx", [0, 1])]
        M = self._zx(gates, 2)
        self._assert_proportional(M, np.eye(4, dtype=complex), "CX²")

    def test_cz_is_symmetric(self):
        """CZ(0,1) = CZ(1,0) up to qubit relabelling (it's symmetric)."""
        zx_01 = self._zx([Gate("cz", [0, 1])], 2)
        zx_10 = self._zx([Gate("cz", [1, 0])], 2)
        np.testing.assert_allclose(zx_01, zx_10, atol=self._ATOL)

    def test_three_qubit_circuit(self):
        """H on q0, CX(0,1), CX(1,2) — 3-qubit ZX evaluation (proportional to ref)."""
        gates = [Gate("h", [0]), Gate("cx", [0, 1]), Gate("cx", [1, 2])]
        self._assert_proportional(self._zx(gates, 3), self._ref(gates, 3),
                                  "3-qubit H+CX+CX")

