"""ZX-calculus rewriting demo.

Run:  PYTHONPATH=. python examples/zx_simplify_demo.py

Demonstrates:
* Convert a Bell circuit (H + CNOT) to a ZX diagram.
* Apply all five rewrite rules: spider_fusion, identity_removal,
  hadamard_cancel, color_change, pi_copy.
* Inspect the proof-carrying ZXRewriteLog.
* Compare node count before and after simplification.
* Show that zx_to_matrix raises NotImplementedError on non-circuit graphs.

Honest scope [研究]
-------------------
* The implemented rules are locally sound (preserve the linear map).
* Global completeness (all equalities provable) is an open research problem.
* zx_to_matrix works only for circuit-topology ZX graphs; it raises
  NotImplementedError otherwise — no silent wrong results.
* Non-Clifford gate rewriting is incomplete; use simplify with care.
"""
import math

import numpy as np

from htf.qasm import Gate, circuit_unitary
from htf.zx import (
    ZXGraph,
    ZXNodeType,
    ZXRewriteLog,
    color_change,
    hadamard_cancel,
    identity_removal,
    pi_copy,
    simplify,
    spider_fusion,
    zx_from_circuit,
    zx_to_matrix,
)

# ── Bell circuit → ZX ──────────────────────────────────────────────────────
bell_gates = [Gate("h", [0]), Gate("cx", [0, 1])]
g          = zx_from_circuit(bell_gates, n_qubits=2)

print(f"Bell circuit as ZX graph:")
print(f"  Nodes: {len(g.nodes)}, Edges: {len(g.edges)}")
print(f"  Inputs: {g.inputs}, Outputs: {g.outputs}")
print(f"  Node kinds: {[n.kind.name for n in g.nodes.values()]}")

# ── Evaluate unitary via zx_to_matrix (single-qubit circuit) ──────────────
# zx_to_matrix reconstructs single-qubit gates only; multi-qubit interactions
# (like CX) are not reconstructed — use circuit_unitary for those.
# Note: ZX Z_α = diag(1, e^{iα}); QASM Rz(α) = diag(e^{-iα/2}, e^{iα/2}).
# They differ by a global phase. We check unitary equivalence (U†V proportional
# to identity) rather than strict equality.
single_gates = [Gate("h", [0])]
g_single     = zx_from_circuit(single_gates, n_qubits=1)
U_single     = zx_to_matrix(g_single)
U_single_ref = circuit_unitary(single_gates, n_qubits=1)
ratio        = U_single @ U_single_ref.conj().T
is_prop_id   = np.allclose(ratio / ratio[0, 0], np.eye(2), atol=1e-10)
assert is_prop_id, "zx_to_matrix not proportional to reference"
print("\nzx_to_matrix (single-qubit H circuit) proportional to circuit_unitary: PASS")

# ── Apply individual rules ─────────────────────────────────────────────────
print("\nApplying individual rules:")

g2   = g.copy()
log2 = ZXRewriteLog()
n_sf = spider_fusion(g2, log2)
print(f"  spider_fusion:    {n_sf} rewrites, nodes now {len(g2.nodes)}")

g3   = g.copy()
log3 = ZXRewriteLog()
n_ir = identity_removal(g3, log3)
print(f"  identity_removal: {n_ir} rewrites, nodes now {len(g3.nodes)}")

g4   = g.copy()
log4 = ZXRewriteLog()
n_hc = hadamard_cancel(g4, log4)
print(f"  hadamard_cancel:  {n_hc} rewrites, nodes now {len(g4.nodes)}")

# ── Full simplification ────────────────────────────────────────────────────
g_simp = g.copy()
log    = ZXRewriteLog()
total  = simplify(g_simp, log=log)

print(f"\nFull simplify (all 5 rules):")
print(f"  Total rewrites: {total}")
print(f"  Nodes before: {len(g.nodes)}, after: {len(g_simp.nodes)}")
rule_counts: dict[str, int] = {}
for step in log.steps:
    rule_counts[step["rule"]] = rule_counts.get(step["rule"], 0) + 1
print(f"  Rule counts: {rule_counts}")

# ── color_change demo ──────────────────────────────────────────────────────
print("\ncolor_change demo:")
gc = ZXGraph()
h1 = gc.add_node(ZXNodeType.H)
z  = gc.add_node(ZXNodeType.Z, phase=math.pi / 4)
h2 = gc.add_node(ZXNodeType.H)
gc.add_edge(h1, z)
gc.add_edge(z, h2)
logc = ZXRewriteLog()
nc = color_change(gc, logc)
print(f"  Z surrounded by 2 H boxes → {nc} color_change applied")
if logc.steps:
    print(f"  Step description: {logc.steps[0]['description']}")

# ── pi_copy demo ───────────────────────────────────────────────────────────
print("\npi_copy demo:")
gp    = ZXGraph()
z_pi  = gp.add_node(ZXNodeType.Z, phase=math.pi, label="Z(π)")
x0    = gp.add_node(ZXNodeType.X, phase=0.0,     label="X(0)")
out   = gp.add_node(ZXNodeType.Z, label="out")
gp.add_edge(z_pi, x0)
gp.add_edge(x0, out)
logp = ZXRewriteLog()
np_  = pi_copy(gp, logp)
print(f"  Z(π) through X(0) → {np_} pi_copy applied")
print(f"  Z(π) node removed: {z_pi not in gp.nodes}")

# ── Non-circuit topology error ─────────────────────────────────────────────
print("\nNon-circuit topology check:")
g_bad = ZXGraph()
inp   = g_bad.add_node(ZXNodeType.INPUT, qubit=0)
out_n = g_bad.add_node(ZXNodeType.OUTPUT, qubit=0)
g_bad.inputs  = [inp]
g_bad.outputs = [out_n]
g_bad.add_edge(inp, out_n)
orphan = g_bad.add_node(ZXNodeType.Z, phase=math.pi)  # unreachable node
try:
    zx_to_matrix(g_bad)
    print("  ERROR: should have raised NotImplementedError")
except NotImplementedError as e:
    print(f"  NotImplementedError raised as expected: {str(e)[:60]}...")

print("\nAll assertions passed.")
