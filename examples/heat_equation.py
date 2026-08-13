"""Phase 2 example: 1D heat equation solved via HTF.

Run:  python examples/heat_equation.py

Demonstrates:
* Building a 1-D heat-equation diagram by composing lattice Boxes.
* Contracting in float mode (discovery-tier, no error bound).
* Contracting in certified mode (rigorous floating-point rounding bound
  via python-flint Arb).

What "certified" means here
---------------------------
The error_bound covers floating-point rounding only — it bounds
|HTF_result - exact_contraction_of_these_float64_tensors|.
It does *not* bound the discretisation error of the PDE, nor the modelling
error between the discrete Laplacian and the continuum operator.  See
PLAN.md §6 for the full honest-scope statement.
"""
import numpy as np

from htf import TensorFunctor, contract
from htf.lattice import heat_step_box, laplacian_box, state_box

# ── problem parameters ─────────────────────────────────────────────
N = 16       # number of lattice sites
D = 0.1      # thermal diffusivity
dt = 0.04    # time step  (stability: D·dt/dx² = 0.04 ≤ 0.5 ✓)
dx = 1.0
STEPS = 20

# ── initial condition: Gaussian pulse centred at mid-lattice ───────
xs = np.arange(N, dtype=float)
u0 = np.exp(-0.5 * ((xs - N / 2) / 2.0) ** 2)
u0 /= u0.sum()  # normalise mass

# ── reference solution: direct matrix iteration ───────────────────
_, L = laplacian_box(N, dx)
M_ref = np.eye(N) + dt * D * L
u_ref = u0.copy()
for _ in range(STEPS):
    u_ref = M_ref @ u_ref

# ── HTF diagram: psi >> heat_step (composed STEPS times) ──────────
psi_box, psi_arr = state_box("psi", u0)
step_box, step_arr = heat_step_box(N, D, dt, dx)

diagram = psi_box
for _ in range(STEPS):
    diagram = diagram >> step_box

F = TensorFunctor({"psi": psi_arr, "heat_step": step_arr})

# ── float mode ────────────────────────────────────────────────────
result_float = contract(diagram, F, mode="float")

print("=== Float mode (discovery-tier, no error bound) ===")
print(f"  Result   (first 4): {result_float[:4]}")
print(f"  Reference(first 4): {u_ref[:4]}")
print(f"  Max |HTF - ref|:    {np.abs(result_float - u_ref).max():.2e}")
print()

# ── certified mode ────────────────────────────────────────────────
cert = contract(diagram, F, mode="certified")

print("=== Certified mode (rigorous floating-point rounding bound) ===")
print(f"  Midpoint (first 4): {np.asarray(cert.result)[:4]}")
print(f"  Error bound (max radius): {cert.error_bound:.2e}")
print(f"  Mode:    {cert.mode}")
print(f"  Backend: {cert.backend}")
print()
print("  Interpretation: the true contraction result lies within")
print(f"  result ± {cert.error_bound:.2e} (floating-point rounding only).")
print("  Discretisation and modelling errors are outside this bound.")

# ── sanity check: certified midpoint ≈ float result ──────────────
max_diff = np.abs(np.asarray(cert.result) - result_float).max()
assert max_diff <= cert.error_bound + 1e-14, (
    f"certified midpoint drifted from float by {max_diff}, "
    f"exceeding error_bound {cert.error_bound}"
)
print(f"\n  Sanity check passed: |certified - float| = {max_diff:.2e} "
      f"≤ error_bound + 1e-14")
