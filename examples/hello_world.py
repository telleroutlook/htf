"""Phase-1 "hello world": build a tiny string diagram and contract it.

Run:  python examples/hello_world.py
"""
import numpy as np

from htf import Box, TensorFunctor, Wire, contract

# Layer 1 — draw the diagram (type-safe: illegal wiring would raise here).
spin = Wire("spin", 2)
psi = Box("psi", (), (spin,))      # state:  () -> spin
U = Box("U", (spin,), (spin,))      # gate:   spin -> spin
phi = Box("phi", (spin,), ())       # effect: spin -> ()
diagram = psi >> U >> phi            # () -> ()  (a scalar amplitude)

# Layer 2 — assign concrete tensors.
F = TensorFunctor(
    {
        "psi": np.array([1.0, 0.0]),                 # |0>
        "U": np.array([[0.0, 1.0], [1.0, 0.0]]),      # swap / NOT
        "phi": np.array([0.0, 1.0]),                  # <1|
    }
)

# Layer 3 — contract (float mode: discovery-tier, no error bound).
amplitude = contract(diagram, F, mode="float")
print("diagram:", diagram, "->", diagram.cod)
print("<phi| U |psi> =", float(amplitude))  # expected: 1.0
