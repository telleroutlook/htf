"""Bell circuit: QASM 2.0 round-trip and HTF diagram bridge.

Run:  PYTHONPATH=. python examples/bell_circuit_qasm.py

Demonstrates:
* Construct a Bell-state preparation circuit (H + CNOT).
* Export to QASM 2.0 string; parse it back to a gate list.
* Simulate the unitary matrix directly.
* Verify the Bell state is produced from |00⟩.
* Bridge to an HTF Diagram (type-checked string-diagram form).
* Show SWAP decomposition for a non-adjacent two-qubit gate.

Honest scope
------------
* Simulation is via dense matrix products — exponential in qubit count.
* No noise model; open-system circuits are [研究].
* QASM 3.0 and OpenQASM extensions are not supported.
"""
import numpy as np

from htf.qasm import (
    Gate,
    circuit_to_diagram,
    circuit_to_qasm,
    circuit_unitary,
    qasm_to_circuit,
)
from htf.topology import dims

# ── Bell circuit ───────────────────────────────────────────────────────────
bell_gates = [Gate("h", [0]), Gate("cx", [0, 1])]

print("Bell-state preparation circuit:")
for g in bell_gates:
    print(f"  {g.name} q{g.qubits}")

# ── Export to QASM 2.0 ─────────────────────────────────────────────────────
qasm_str = circuit_to_qasm(bell_gates, n_qubits=2)
print("\nQASM 2.0 export:")
print(qasm_str)

# ── Round-trip parse ───────────────────────────────────────────────────────
parsed_gates = qasm_to_circuit(qasm_str)
assert [g.name for g in parsed_gates] == ["h", "cx"], "Round-trip gate names mismatch"
print("Round-trip parse: OK\n")

# ── Simulate unitary ───────────────────────────────────────────────────────
U = circuit_unitary(bell_gates, n_qubits=2)
print("Unitary matrix (real part):")
print(np.round(U.real, 4))

# Apply to |00⟩
psi0 = np.array([1, 0, 0, 0], dtype=complex)
bell_state = U @ psi0
expected   = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
print(f"\n|00⟩ → Bell state |Φ+⟩ = {np.round(bell_state, 4)}")
assert np.allclose(bell_state, expected, atol=1e-12), "Bell state mismatch"
print("Bell state check: PASS")

# ── HTF Diagram bridge ─────────────────────────────────────────────────────
diag = circuit_to_diagram(bell_gates, n_qubits=2)
print(f"\nHTF Diagram: dom={dims(diag.dom)}, cod={dims(diag.cod)}")
assert dims(diag.dom) == (2, 2)
assert dims(diag.cod) == (2, 2)
print("Diagram type check: PASS")

# ── Non-adjacent gate with SWAP decomposition ─────────────────────────────
print("\nNon-adjacent CNOT q[0], q[2] (3-qubit system):")
gates_na = [Gate("h", [0]), Gate("cx", [0, 2])]
diag_na  = circuit_to_diagram(gates_na, n_qubits=3, adjacent_only=False)
print(f"  Diagram with SWAP: dom={dims(diag_na.dom)}, cod={dims(diag_na.cod)}")

diag_fb  = circuit_to_diagram(gates_na, n_qubits=3, adjacent_only=True)
print(f"  Diagram fallback:  dom={dims(diag_fb.dom)}, cod={dims(diag_fb.cod)}")

U_na = circuit_unitary(gates_na, n_qubits=3)
assert np.allclose(U_na @ U_na.conj().T, np.eye(8), atol=1e-10)
print("  Unitary is unitary: PASS")

print("\nAll assertions passed.")
