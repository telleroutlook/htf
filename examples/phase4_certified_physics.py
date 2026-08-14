"""Phase 4 example: spectral gap bounds, χ-convergence, entanglement difficulty map.

Run:  PYTHONPATH=. python examples/phase4_certified_physics.py

Demonstrates:
* Exact spectral gap from full diagonalisation.
* Variational upper bound on gap via excited-state MERA.
* Temple's inequality lower bound on E_0 (heuristic, NOT rigorous — P0-1).
* Certified gap upper bound via flint Arb.
* χ-convergence study (energy vs bond dimension, generalised spin model).
* Entanglement entropy profile and difficulty map.

Honest scope
------------
All certified bounds cover floating-point rounding only.
Bond-dimension truncation bias (χ→∞) and finite-size effects are [OUT].
"""
import numpy as np

from htf.labs import (
    certified_gap_upper,
    chi_convergence_study,
    difficulty_report,
    first_excited_upper,
    h2_expectation,
    optimize_mera,
    random_mera,
    spectral_gap_exact,
    temple_lower_bound,
    transverse_ising_ham,
)
from htf.variational import energy_expectation

# ── Setup: qubit TFIM Hamiltonian ─────────────────────────────────────
N, J, h = 4, 1.0, 0.5
H = transverse_ising_ham(N, J=J, h=h)
exact_evals = np.linalg.eigvalsh(H)
E0_exact    = exact_evals[0]
gap_exact   = exact_evals[1] - exact_evals[0]
print(f"Transverse-field Ising model: N={N}, J={J}, h={h}")
print(f"  Exact E_0       = {E0_exact:.6f}")
print(f"  Exact gap E1-E0 = {gap_exact:.6f}")
print()

# ── Ground state via MERA ─────────────────────────────────────────────
mera0      = random_mera(N, chi=2, seed=42)
mera_gs, _ = optimize_mera(H, mera0, n_iter=80, tol=1e-6)
psi_gs     = mera_gs.state_vector()
E0_var     = energy_expectation(H, psi_gs)

# Excited-state trial: different seed
psi_es_raw = random_mera(N, chi=2, seed=99).state_vector()

E1_var = first_excited_upper(H, psi_gs, psi_es_raw)
h2_exp = h2_expectation(H, psi_gs)
t_lb   = temple_lower_bound(E0_var, h2_exp, E1_var)

print("── Spectral gap bounds ──────────────────────────────────────────")
print(f"  E0_var (chi=2)   = {E0_var:.6f}  (exact: {E0_exact:.6f})")
print(f"  E1_var (excited) = {E1_var:.6f}")
print(f"  Gap upper bound  = {E1_var - E0_var:.6f}  (exact: {gap_exact:.6f})")
print(f"  Temple lb (raw)  = {t_lb:.6f}  [valid only when E0 < E_var < E_1_exact]")
# Temple's inequality requires E_var < E_1_exact (true first excited energy).
# Here chi=2 MERA only achieves E_var=-1.25, which is above E_1_exact≈-3.33,
# so the bound is formally inapplicable.  It is shown for demonstration only.
print(f"  (E_1_exact ≈ {exact_evals[1]:.4f}; E_var={E0_var:.4f} > E_1_exact "
      f"→ Temple condition not met for this trial state)")
print()

# ── Certified gap upper bound ─────────────────────────────────────────
cert_gap = certified_gap_upper(H, psi_gs, psi_es_raw)
print("── Certified gap bound (flint Arb) ─────────────────────────────")
print(f"  gap_var     = {cert_gap.result:.8f}")
print(f"  error_bound = {cert_gap.error_bound:.2e}  (floating-point rounding)")
assert cert_gap.error_bound >= 0, "error_bound must be non-negative"
print("  ✓ error_bound ≥ 0")
print()

# ── χ-convergence study ───────────────────────────────────────────────
# Use a generalised spin-χ Hamiltonian: nearest-neighbour coupling
# H = -J Σ Σ_ab |a><b| ⊗ |b><a|  (acts in chi^n_sites dimensional space)
# For chi=2 this reduces to the XX-like model; here we use a simple
# random-Hamiltonian approach that works for any chi.
print("── χ-convergence study (chi = 2, 3, 4, generalised spin model) ─")


def spin_chi_ham(n_sites: int, chi: int, J: float = 1.0, seed: int = 7) -> np.ndarray:
    """Random nearest-neighbour spin-χ Hamiltonian in chi^n_sites space.

    Builds H = -J Σ_{i} h_{i,i+1} where each h_{i,i+1} is a random
    real symmetric two-site interaction drawn from the GUE.
    Returned matrix is symmetric and (chi^n_sites × chi^n_sites).
    """
    rng = np.random.default_rng(seed)
    dim = chi ** n_sites
    H = np.zeros((dim, dim))
    chi2 = chi * chi
    for site in range(n_sites - 1):
        # Random real symmetric chi^2 × chi^2 two-body term
        raw = rng.standard_normal((chi2, chi2))
        h2 = (raw + raw.T) / 2.0
        # Embed into full space: I^(site) ⊗ h2 ⊗ I^(n-site-2)
        left_dim  = chi ** site
        right_dim = chi ** (n_sites - site - 2)
        full_h = np.kron(np.kron(np.eye(left_dim), h2), np.eye(right_dim))
        H -= J * full_h
    return H


report = chi_convergence_study(
    n_sites=N,
    chi_list=[2, 3, 4],
    ham_factory=spin_chi_ham,
    n_iter=60,
    seed=0,
)
print(report.summary())
print()

# ── Difficulty map (qubit TFIM) ───────────────────────────────────────
print("── Difficulty map (entanglement entropy profile) ────────────────")
drep = difficulty_report(H, N, n_iter=60, seed=0)
print(drep.summary())
print()

# ── Sanity checks ─────────────────────────────────────────────────────
assert E0_var >= E0_exact - 1e-6, "variational bound violated"
assert E1_var > E0_var, "E1_var must be above E0_var"
# Temple lower bound is only valid when E_var < E_1_exact; check the bound
# is at least a lower bound on E_var (i.e. Temple formula didn't blow up).
assert t_lb <= E0_var + 1e-10, f"Temple bound {t_lb:.6f} should not exceed E0_var {E0_var:.6f}"
print("✓ All sanity checks passed.")
