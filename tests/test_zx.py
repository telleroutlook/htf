"""Tests for htf/zx.py — ZX-calculus diagram rewriting."""
import math

import numpy as np
import pytest

from htf.qasm import Gate
from htf.zx import (
    ZXGraph,
    ZXNodeType,
    ZXRewriteLog,
    hadamard_cancel,
    identity_removal,
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
