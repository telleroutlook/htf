"""Lanczos two-sided spectral bounds demo.

Run:  PYTHONPATH=. python examples/lanczos_bounds.py

Demonstrates:
* Build a 4-site transverse-field Ising Hamiltonian.
* Run k-step Lanczos with full re-orthogonalization (ghost-free).
* Compute Ritz values (variational upper bounds on eigenvalues).
* Apply Temple's inequality for a rigorous finite-lattice lower bound on E_0.
* Compare the Lanczos interval [E0_lower, E0_upper] with the exact ground state.

Honest scope
------------
* Ritz values are variational upper bounds on eigenvalues; convergence rate
  depends on the spectral gap and k.
* Temple's lower bound is a rigorous finite-lattice bound when E_var < E_1_exact.
  It does NOT certify the continuum gap (continuum limit is [OUT]).
* All computations are float mode; combining with certified mode is [研究].
"""
import numpy as np

from htf.lanczos import lanczos_eigs, temple_lanczos, two_sided_bounds
from htf.variational import transverse_ising_ham

# ── Hamiltonian ────────────────────────────────────────────────────────────
n    = 4
H    = transverse_ising_ham(n=n, J=1.0, h=0.5)
evals_exact = np.linalg.eigvalsh(H)
E0_exact    = evals_exact[0]
E1_exact    = evals_exact[1]
gap_exact   = E1_exact - E0_exact

print(f"System: {n}-site TFIM (J=1, h=0.5)")
print(f"  E0 exact = {E0_exact:.10f}")
print(f"  E1 exact = {E1_exact:.10f}")
print(f"  Gap exact = {gap_exact:.10f}")

# ── Lanczos Ritz values ────────────────────────────────────────────────────
k = 12
evals_ritz, _ = lanczos_eigs(H, k=k, seed=0)
print(f"\nLanczos Ritz values (k={k}):")
for i, ev in enumerate(evals_ritz[:4]):
    print(f"  θ_{i} = {ev:.10f}  (exact = {evals_exact[i]:.10f},"
          f" err = {ev - evals_exact[i]:+.2e})")

# ── Two-sided bounds on E_0 ────────────────────────────────────────────────
bounds = temple_lanczos(H, k=k, seed=0)
print(f"\nTwo-sided bounds on E_0 (k={k}):")
print(f"  E0 upper (Ritz)  = {bounds.E0_upper:.10f}")
print(f"  E0 lower (Temple)= {bounds.E0_lower:.10f}")
print(f"  Interval width   = {bounds.width:.2e}")
print(f"  Temple condition = {bounds.temple_condition_met}")
print(f"  E0 exact in interval: {bounds.E0_lower <= E0_exact <= bounds.E0_upper}")
print(f"\n  Notes: {bounds.notes}")

# ── Convergence with k ─────────────────────────────────────────────────────
print("\nConvergence of interval width with k:")
for k_val in [4, 6, 8, 10, 12, 16]:
    b = two_sided_bounds(H, k=k_val, seed=0)
    width = b.width if b.temple_condition_met else float("inf")
    print(f"  k={k_val:2d}: upper={b.E0_upper:.8f}, lower={b.E0_lower:.8f},"
          f" width={width:.2e}, temple={b.temple_condition_met}")
