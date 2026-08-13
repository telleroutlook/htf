"""Phase 3 example: MERA variational ground state + certified energy bound.

Run:  PYTHONPATH=. python examples/mera_variational.py

Demonstrates:
* Build a random 4-site MERA (chi=2) on the transverse-field Ising model.
* Structure verification: check isometry of MERA layers (proof-carrying).
* Variational optimisation: minimise ⟨ψ|H|ψ⟩ using L-BFGS-B.
* Certified upper bound on ground-state energy via flint Arb.
* Compare with exact ground state from numpy eigensolver.

Honest scope
------------
The certified error_bound covers floating-point rounding in the energy
computation only.  Bond-dimension truncation bias (chi=2 vs chi→∞) is not
certified here — that is Phase 4 territory.  For chi=2 on 4 sites the
variational ansatz is expressive enough to get close to exact, but in
general the gap between E_var(chi) and E_0 is a separate, uncertified bias.
"""
import numpy as np

from htf import (
    check_isometry,
    enforce_isometry,
    random_mera,
    transverse_ising_ham,
)
from htf.variational import energy_expectation, optimize_mera, variational_bound

# ── Hamiltonian ────────────────────────────────────────────────────
N, J, h = 4, 1.0, 0.5
H = transverse_ising_ham(N, J=J, h=h)

exact_energies = np.linalg.eigvalsh(H)
E0_exact = exact_energies[0]
print(f"Exact ground-state energy (N={N}, J={J}, h={h}): {E0_exact:.6f}")
print()

# ── Initial MERA ───────────────────────────────────────────────────
mera0 = random_mera(N, chi=2, seed=42)
E_init = energy_expectation(H, mera0.state_vector())
print(f"Initial MERA energy: {E_init:.6f}")

# ── Structure check (Track B: proof-carrying) ──────────────────────
print("\n--- Structure checks on initial MERA ---")
for li, layer in enumerate(mera0.layers):
    for ki, w in enumerate(layer.isometries):
        report = check_isometry(w)
        print(f"  Layer {li}, isometry {ki}: {report}")

# ── Variational optimisation ───────────────────────────────────────
print("\n--- Optimising MERA (L-BFGS-B, 80 iterations) ---")
mera_opt, history = optimize_mera(H, mera0, n_iter=80, tol=1e-6)
E_opt = energy_expectation(H, mera_opt.state_vector())
print(f"  Energy: {E_init:.6f} → {E_opt:.6f}  (exact: {E0_exact:.6f})")
print(f"  Converged in {len(history)} history steps")
print(f"  Variational gap (chi=2 bias): {E_opt - E0_exact:.4e}  [OUT of certified scope]")

# ── Structure check after optimisation ────────────────────────────
print("\n--- Structure checks after optimisation (enforce_constraints applied) ---")
for li, layer in enumerate(mera_opt.layers):
    for ki, w in enumerate(layer.isometries):
        report = check_isometry(w)
        print(f"  Layer {li}, isometry {ki}: {report}")

# ── Certified upper bound (Track A: first certified bound) ─────────
print("\n--- Certified upper bound on ground-state energy ---")
cert = variational_bound(H, mera_opt)
print(f"  result     = {cert.result:.8f}")
print(f"  error_bound= {cert.error_bound:.2e}  (floating-point rounding only)")
print(f"  mode       = {cert.mode},  backend = {cert.backend}")
print()
print("  Certified: E_0 ≤ result + error_bound  (variational upper bound)")
print(f"           = {cert.result + cert.error_bound:.8f}")
print(f"  Exact E_0 = {E0_exact:.8f}  ← should be ≤ certified bound")
assert E0_exact <= cert.result + cert.error_bound + 1e-10, (
    "certified bound violated!"
)
print("\n  ✓ Certified bound is valid (E_0 ≤ E_var + ε).")
print()
print(cert.to_json(indent=2))
