"""Tests for htf/qasm.py — OpenQASM 2.0 quantum circuit interoperability."""
import numpy as np
import pytest

from htf.qasm import (
    Gate,
    circuit_to_diagram,
    circuit_to_qasm,
    circuit_unitary,
    get_gate_matrix,
    qasm_to_circuit,
)

# ─────────────────────── TestGetGateMatrix ────────────────────────────────

class TestGetGateMatrix:

    def test_h_is_unitary(self):
        H = get_gate_matrix("h")
        assert np.allclose(H @ H.conj().T, np.eye(2), atol=1e-12)

    def test_h_is_hadamard(self):
        H = get_gate_matrix("h")
        expected = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        assert np.allclose(H, expected, atol=1e-12)

    def test_x_is_pauli_x(self):
        X = get_gate_matrix("x")
        assert np.allclose(X, np.array([[0, 1], [1, 0]]), atol=1e-12)

    def test_y_is_pauli_y(self):
        Y = get_gate_matrix("y")
        assert np.allclose(Y, np.array([[0, -1j], [1j, 0]]), atol=1e-12)

    def test_z_is_pauli_z(self):
        Z = get_gate_matrix("z")
        assert np.allclose(Z, np.diag([1, -1]), atol=1e-12)

    def test_s_squared_is_z(self):
        S = get_gate_matrix("s")
        assert np.allclose(S @ S, get_gate_matrix("z"), atol=1e-12)

    def test_sdg_is_s_dagger(self):
        S = get_gate_matrix("s")
        Sdg = get_gate_matrix("sdg")
        assert np.allclose(Sdg, S.conj().T, atol=1e-12)

    def test_t_fourth_power_is_z(self):
        T = get_gate_matrix("t")
        Z = get_gate_matrix("z")
        assert np.allclose(np.linalg.matrix_power(T, 4), Z, atol=1e-12)

    def test_t_eighth_power_is_identity(self):
        T = get_gate_matrix("t")
        assert np.allclose(np.linalg.matrix_power(T, 8), np.eye(2), atol=1e-12)

    def test_cx_is_controlled_not(self):
        CX = get_gate_matrix("cx")
        expected = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        assert np.allclose(CX, expected, atol=1e-12)

    def test_cx_unitary(self):
        CX = get_gate_matrix("cx")
        assert np.allclose(CX @ CX.conj().T, np.eye(4), atol=1e-12)

    def test_cz_unitary(self):
        CZ = get_gate_matrix("cz")
        assert np.allclose(CZ @ CZ.conj().T, np.eye(4), atol=1e-12)

    def test_swap_swaps_states(self):
        SWAP = get_gate_matrix("swap")
        # |01⟩ → |10⟩
        v = np.array([0, 1, 0, 0], dtype=complex)  # |01⟩
        assert np.allclose(SWAP @ v, np.array([0, 0, 1, 0]), atol=1e-12)

    def test_rx_at_pi_is_ix(self):
        Rx_pi = get_gate_matrix("rx", [np.pi])
        X = get_gate_matrix("x")
        assert np.allclose(Rx_pi, -1j * X, atol=1e-10)

    def test_ry_at_pi_is_iy(self):
        Ry_pi = get_gate_matrix("ry", [np.pi])
        Y = get_gate_matrix("y")
        assert np.allclose(Ry_pi, -1j * Y, atol=1e-10)

    def test_rz_at_zero_is_identity(self):
        Rz_0 = get_gate_matrix("rz", [0.0])
        assert np.allclose(Rz_0, np.eye(2), atol=1e-12)

    def test_rx_ry_rz_are_unitary(self):
        for name, theta in [("rx", 1.2), ("ry", 0.7), ("rz", 2.5)]:
            U = get_gate_matrix(name, [theta])
            assert np.allclose(U @ U.conj().T, np.eye(2), atol=1e-12)

    def test_unknown_gate_raises(self):
        with pytest.raises(ValueError, match="Unknown gate"):
            get_gate_matrix("foo")

    def test_case_insensitive(self):
        H_lower = get_gate_matrix("h")
        H_upper = get_gate_matrix("H")
        assert np.allclose(H_lower, H_upper, atol=1e-12)

    def test_rx_missing_params_raises(self):
        with pytest.raises(ValueError, match="angle parameter"):
            get_gate_matrix("rx", [])


# ─────────────────────── TestCircuitToQasm ───────────────────────────────

class TestCircuitToQasm:

    def test_header_present(self):
        qasm = circuit_to_qasm([], n_qubits=2)
        assert "OPENQASM 2.0" in qasm
        assert 'include "qelib1.inc"' in qasm

    def test_qreg_line(self):
        qasm = circuit_to_qasm([], n_qubits=3)
        assert "qreg q[3]" in qasm

    def test_single_h_gate(self):
        gates = [Gate("h", [0])]
        qasm  = circuit_to_qasm(gates, n_qubits=1)
        assert "h q[0]" in qasm

    def test_cx_gate(self):
        gates = [Gate("cx", [0, 1])]
        qasm  = circuit_to_qasm(gates, n_qubits=2)
        assert "cx q[0], q[1]" in qasm

    def test_rx_gate_with_params(self):
        gates = [Gate("rx", [0], [np.pi / 2])]
        qasm  = circuit_to_qasm(gates, n_qubits=1)
        assert "rx(" in qasm
        assert "q[0]" in qasm

    def test_empty_circuit(self):
        qasm = circuit_to_qasm([], n_qubits=2)
        assert "qreg q[2]" in qasm

    def test_roundtrip_simple(self):
        gates_in = [Gate("h", [0]), Gate("cx", [0, 1])]
        qasm     = circuit_to_qasm(gates_in, n_qubits=2)
        gates_out = qasm_to_circuit(qasm)
        assert gates_out[0].name == "h"
        assert gates_out[0].qubits == [0]
        assert gates_out[1].name == "cx"
        assert gates_out[1].qubits == [0, 1]

    def test_roundtrip_preserves_params(self):
        theta = np.pi / 3
        gates_in = [Gate("rx", [0], [theta])]
        qasm     = circuit_to_qasm(gates_in, n_qubits=1)
        gates_out = qasm_to_circuit(qasm)
        assert gates_out[0].name == "rx"
        assert abs(gates_out[0].params[0] - theta) < 1e-6


# ─────────────────────── TestQasmToCircuit ───────────────────────────────

class TestQasmToCircuit:

    def test_parse_h_gate(self):
        src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\nh q[0];\n"
        gates = qasm_to_circuit(src)
        assert len(gates) == 1
        assert gates[0].name == "h"
        assert gates[0].qubits == [0]

    def test_parse_cx_gate(self):
        src = "OPENQASM 2.0;\nqreg q[2];\ncx q[0], q[1];\n"
        gates = qasm_to_circuit(src)
        assert len(gates) == 1
        assert gates[0].name == "cx"
        assert gates[0].qubits == [0, 1]

    def test_parse_rx_with_pi(self):
        src = "OPENQASM 2.0;\nqreg q[1];\nrx(pi/2) q[0];\n"
        gates = qasm_to_circuit(src)
        assert gates[0].name == "rx"
        assert abs(gates[0].params[0] - np.pi / 2) < 1e-10

    def test_skip_comments(self):
        src = "OPENQASM 2.0;\n// this is a comment\nqreg q[1];\nh q[0];\n"
        gates = qasm_to_circuit(src)
        assert len(gates) == 1

    def test_skip_measure(self):
        src = "OPENQASM 2.0;\nqreg q[1];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n"
        gates = qasm_to_circuit(src)
        assert len(gates) == 1

    def test_skip_barrier(self):
        src = "OPENQASM 2.0;\nqreg q[2];\nh q[0];\nbarrier q[0], q[1];\ncx q[0], q[1];\n"
        gates = qasm_to_circuit(src)
        assert len(gates) == 2

    def test_empty_source_returns_empty_list(self):
        gates = qasm_to_circuit("OPENQASM 2.0;\nqreg q[1];\n")
        assert gates == []

    def test_invalid_gate_line_raises(self):
        # Line that starts with a digit — cannot match any gate pattern
        with pytest.raises(ValueError, match="Cannot parse"):
            qasm_to_circuit("OPENQASM 2.0;\nqreg q[1];\n123bad q[0];\n")

    def test_multi_gate_order_preserved(self):
        src = (
            "OPENQASM 2.0;\nqreg q[2];\n"
            "h q[0];\nx q[1];\ncx q[0], q[1];\n"
        )
        gates = qasm_to_circuit(src)
        assert [g.name for g in gates] == ["h", "x", "cx"]


# ─────────────────────── TestCircuitUnitary ──────────────────────────────

class TestCircuitUnitary:

    def test_identity_circuit(self):
        U = circuit_unitary([], n_qubits=2)
        assert np.allclose(U, np.eye(4), atol=1e-12)

    def test_h_on_qubit_0_correct(self):
        gates = [Gate("h", [0])]
        U     = circuit_unitary(gates, n_qubits=1)
        H_mat = get_gate_matrix("h")
        assert np.allclose(U, H_mat, atol=1e-12)

    def test_h_h_is_identity(self):
        gates = [Gate("h", [0]), Gate("h", [0])]
        U     = circuit_unitary(gates, n_qubits=1)
        assert np.allclose(U, np.eye(2), atol=1e-12)

    def test_bell_state_preparation(self):
        # H on q0, then CNOT q0→q1
        gates = [Gate("h", [0]), Gate("cx", [0, 1])]
        U     = circuit_unitary(gates, n_qubits=2)
        # |00⟩ → Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        v0    = np.array([1, 0, 0, 0], dtype=complex)
        bell  = U @ v0
        expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        assert np.allclose(bell, expected, atol=1e-12)

    def test_result_is_unitary(self):
        gates = [Gate("h", [0]), Gate("cx", [0, 1]), Gate("z", [1])]
        U     = circuit_unitary(gates, n_qubits=2)
        assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-10)

    def test_x_on_qubit_1_in_2_qubit_system(self):
        gates = [Gate("x", [1])]
        U     = circuit_unitary(gates, n_qubits=2)
        I2    = np.eye(2, dtype=complex)
        X     = get_gate_matrix("x")
        expected = np.kron(I2, X)
        assert np.allclose(U, expected, atol=1e-12)

    def test_rx_on_qubit_0(self):
        theta = 0.8
        gates = [Gate("rx", [0], [theta])]
        U     = circuit_unitary(gates, n_qubits=1)
        expected = get_gate_matrix("rx", [theta])
        assert np.allclose(U, expected, atol=1e-12)

    def test_swap_swaps_state(self):
        gates = [Gate("swap", [0, 1])]
        U     = circuit_unitary(gates, n_qubits=2)
        v_01  = np.array([0, 1, 0, 0], dtype=complex)  # |01⟩
        out   = U @ v_01
        assert np.allclose(out, np.array([0, 0, 1, 0]), atol=1e-12)  # |10⟩


# ─────────────────────── TestCircuitToDiagram ────────────────────────────

class TestCircuitToDiagram:

    def test_empty_circuit_returns_id(self):
        from htf.topology import Id
        d = circuit_to_diagram([], n_qubits=2)
        assert isinstance(d, Id)

    def test_single_gate_is_diagram(self):
        from htf.topology import Diagram
        d = circuit_to_diagram([Gate("h", [0])], n_qubits=2)
        assert isinstance(d, Diagram)

    def test_dom_cod_widths_match_n_qubits(self):
        from htf.topology import dims
        d = circuit_to_diagram([Gate("h", [0]), Gate("cx", [0, 1])], n_qubits=2)
        assert len(d.dom) == 2
        assert len(d.cod) == 2

    def test_all_wire_dims_are_2(self):
        from htf.topology import dims
        d = circuit_to_diagram([Gate("x", [1])], n_qubits=3)
        assert all(di == 2 for di in dims(d.dom))
        assert all(di == 2 for di in dims(d.cod))

    def test_sequential_composition_type_checks(self):
        # If type checking fails, the constructor raises TypeError
        gates = [Gate("h", [0]), Gate("cx", [0, 1]), Gate("z", [1])]
        d = circuit_to_diagram(gates, n_qubits=2)  # must not raise
        assert d is not None

    def test_three_qubit_circuit(self):
        from htf.topology import Diagram
        gates = [Gate("h", [0]), Gate("cx", [0, 1]), Gate("cx", [1, 2])]
        d = circuit_to_diagram(gates, n_qubits=3)
        assert isinstance(d, Diagram)
        assert len(d.dom) == 3
