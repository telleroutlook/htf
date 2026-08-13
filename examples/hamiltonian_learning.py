"""Hamiltonian learning via inverse design.

Run:  PYTHONPATH=. python examples/hamiltonian_learning.py

Demonstrates:
* Use inverse_design to find coupling/field parameters such that the
  ground-state energy of a transverse-field Ising model matches a target.
* Use hamiltonian_learning to recover parameters from observed energy levels.
* Report residuals, convergence, and honest scope.

Honest scope [工程]/[研究]
--------------------------
* Minimisation is L-BFGS-B with random restarts; local minima are possible.
* Uniqueness of solution is model-dependent and not guaranteed [研究].
* Finite-lattice only; the result does not certify continuum parameters.
* For certified bounds on E0, use temple_lanczos or variational_bound.
"""
import numpy as np

from htf.inverse import hamiltonian_learning, inverse_design
from htf.variational import transverse_ising_ham

# ── Problem setup ──────────────────────────────────────────────────────────
n_sites = 4
print(f"Hamiltonian learning demo ({n_sites}-site TFIM)")

# Ground truth: TFIM with J=1.0, h=0.8
H_true     = transverse_ising_ham(n=n_sites, J=1.0, h=0.8)
e_true     = np.linalg.eigvalsh(H_true)
E0_true    = float(e_true[0])
E1_true    = float(e_true[1])
print(f"\nGround-truth TFIM (J=1, h=0.8):")
print(f"  E0 = {E0_true:.8f}")
print(f"  E1 = {E1_true:.8f}")
print(f"  Gap = {E1_true - E0_true:.8f}")

# ── Inverse design: find params with E0 ≈ target ─────────────────────────
target_e0 = E0_true
print(f"\nInverse design: find TFIM params with E0 ≈ {target_e0:.6f}")

result = inverse_design(
    target_e0=target_e0,
    model="ising",
    n_sites=n_sites,
    n_restarts=8,
    seed=42,
)
print(f"  E0 achieved:   {result.E0_achieved:.8f}")
print(f"  Target:        {target_e0:.8f}")
print(f"  Residual:      {result.residual:.2e}")
print(f"  Converged:     {result.converged}")
print(f"  Restarts used: {result.n_restarts}")
print(f"  Params ({result.param_names}): {np.round(result.params_opt, 4)}")
print(f"  Notes: {result.notes[:80]}...")

# ── Hamiltonian learning from two energy levels ────────────────────────────
print(f"\nHamiltonian learning from observed energy levels [E0, E1]:")
target_levels = np.array([E0_true, E1_true])

lresult = hamiltonian_learning(
    target_energies=target_levels,
    model="ising",
    n_sites=n_sites,
    n_restarts=8,
    seed=0,
)
print(f"  Target levels:   {np.round(target_levels, 6)}")
print(f"  Achieved levels: {np.round(lresult.achieved_energies[:2], 6)}")
print(f"  Final loss:      {lresult.loss_final:.2e}")
print(f"  Converged:       {lresult.converged}")
print(f"  Recovered params ({lresult.param_names}): {np.round(lresult.params_opt, 4)}")

# ── Interpretation ─────────────────────────────────────────────────────────
print("\nHonest scope:")
print("  [工程] inverse_design minimises (E0(params) - target)² via L-BFGS-B.")
print("  [研究] Uniqueness of solution depends on the model; not guaranteed.")
print("  [OUT]  Does not certify continuum Hamiltonian parameters.")
