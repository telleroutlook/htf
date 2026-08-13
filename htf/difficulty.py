"""HTF Phase 4 — Entanglement entropy and computational difficulty map.

Provides tools to assess how hard a tensor-network ground-state calculation
is.  The key diagnostic is bipartite entanglement entropy: states with
area-law entanglement (S ≈ const) are accessible to MPS/MERA with moderate χ,
while volume-law or logarithmically growing entanglement signals a hard or
critical regime.

Honest scope
------------
* All computations here are float mode (discovery-tier).
* Area-law / volume-law classification is a **heuristic diagnostic**, not a
  certified result.  The difficulty labels are descriptive.
* The continuum entanglement scaling (L→∞, χ→∞) is ``[OUT]``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


def entanglement_entropy(
    state_vec: np.ndarray,
    n_sites: int,
    cut: int,
) -> float:
    """Von Neumann entanglement entropy S(A) = −Tr(ρ_A log ρ_A).

    Splits the ``n_sites``-qudit state at bond ``cut`` (sites 0…cut−1 vs
    cut…n_sites−1), computes the reduced density matrix via SVD, and returns
    the von Neumann entropy in nats.

    Parameters
    ----------
    state_vec : 1-D array of length ``d^n_sites`` (d = local dimension).
    n_sites   : total number of sites.
    cut       : number of sites in subsystem A (``1 ≤ cut < n_sites``).

    Returns
    -------
    float
        Von Neumann entropy in nats.  Returns 0.0 for product states.
    """
    psi = np.asarray(state_vec, dtype=float).reshape(-1)
    total_dim = len(psi)
    d = round(total_dim ** (1.0 / n_sites))
    if d ** n_sites != total_dim:
        raise ValueError(
            f"state_vec length {total_dim} is not d^n_sites for any integer d"
        )
    if not (1 <= cut < n_sites):
        raise ValueError(f"cut must satisfy 1 ≤ cut < n_sites, got {cut}")
    norm = np.linalg.norm(psi)
    if norm < 1e-15:
        raise ValueError("state vector has zero norm")
    psi = psi / norm

    dim_A = d ** cut
    dim_B = d ** (n_sites - cut)
    sv = np.linalg.svd(psi.reshape(dim_A, dim_B), compute_uv=False)
    probs = sv ** 2
    probs = probs[probs > 1e-15]
    return float(-np.sum(probs * np.log(probs)))


def entanglement_spectrum(
    state_vec: np.ndarray,
    n_sites: int,
    cut: int,
) -> np.ndarray:
    """Schmidt values (singular values) across the bipartition at *cut*.

    Returns a 1-D array sorted in descending order.
    """
    psi = np.asarray(state_vec, dtype=float).reshape(-1)
    total_dim = len(psi)
    d = round(total_dim ** (1.0 / n_sites))
    if d ** n_sites != total_dim:
        raise ValueError(
            f"state_vec length {total_dim} is not d^n_sites for any integer d"
        )
    if not (1 <= cut < n_sites):
        raise ValueError(f"cut must satisfy 1 ≤ cut < n_sites, got {cut}")
    norm = np.linalg.norm(psi)
    if norm < 1e-15:
        raise ValueError("state vector has zero norm")
    psi = psi / norm
    dim_A = d ** cut
    dim_B = d ** (n_sites - cut)
    return np.linalg.svd(psi.reshape(dim_A, dim_B), compute_uv=False)


def bipartite_entanglement_profile(
    state_vec: np.ndarray,
    n_sites: int,
) -> np.ndarray:
    """Entanglement entropy S(cut) for all cuts ``1 ≤ cut < n_sites``.

    Returns an array of length ``n_sites − 1``.
    """
    return np.array([
        entanglement_entropy(state_vec, n_sites, cut)
        for cut in range(1, n_sites)
    ])


@dataclass
class DifficultyReport:
    """Computational difficulty assessment for a tensor-network calculation."""
    n_sites: int
    chi_used: int
    energy: float
    entanglement_profile: np.ndarray  # S(cut) for cut = 1 … n_sites−1
    max_entropy: float
    area_law_limit: float             # log(chi): rough MERA capacity threshold
    likely_area_law: bool             # S_max < area_law_limit (heuristic)
    notes: str = ""

    def summary(self) -> str:
        profile_str = "  ".join(
            f"S({c+1})={s:.3f}" for c, s in enumerate(self.entanglement_profile)
        )
        regime = (
            "area-law (possibly gapped)"
            if self.likely_area_law
            else "possibly critical / volume-law"
        )
        return (
            f"Difficulty report: n_sites={self.n_sites}, chi={self.chi_used}\n"
            f"  Energy = {self.energy:.6f}\n"
            f"  Entanglement profile: {profile_str}\n"
            f"  Max S = {self.max_entropy:.4f}  "
            f"(area-law limit ≈ log(chi) = {self.area_law_limit:.4f})\n"
            f"  Regime: {regime}  [heuristic, discovery-tier]\n"
            f"  {self.notes}"
        )


def difficulty_report(
    ham: np.ndarray,
    n_sites: int,
    n_iter: int = 50,
    seed: int = 0,
) -> DifficultyReport:
    """Compute a full difficulty report for a finite-lattice Hamiltonian.

    The local (physical) dimension ``d`` is derived from the Hamiltonian:
    ``d = ham.shape[0] ** (1 / n_sites)``.  The MERA uses ``chi = d``.
    For qubits (d=2) this means chi=2; for qutrits (d=3) chi=3, etc.

    Parameters
    ----------
    ham     : dense ``(d^n_sites × d^n_sites)`` Hamiltonian.
    n_sites : number of physical sites (power of 2).
    n_iter  : L-BFGS-B iterations.
    seed    : RNG seed.
    """
    from .mera import random_mera
    from .variational import energy_expectation, optimize_mera

    dim = ham.shape[0]
    chi = round(dim ** (1.0 / n_sites))
    if chi ** n_sites != dim:
        raise ValueError(
            f"Cannot derive integer local dimension from ham.shape[0]={dim} "
            f"and n_sites={n_sites}.  "
            f"Ensure ham.shape[0] == d^n_sites for some integer d."
        )

    mera0       = random_mera(n_sites, chi=chi, seed=seed)
    mera_opt, _ = optimize_mera(ham, mera0, n_iter=n_iter, tol=1e-6)
    psi         = mera_opt.state_vector()
    energy      = energy_expectation(ham, psi)
    profile     = bipartite_entanglement_profile(psi, n_sites)
    max_s       = float(np.max(profile))
    area_lim    = float(np.log(chi)) if chi > 1 else 0.0

    return DifficultyReport(
        n_sites=n_sites,
        chi_used=chi,
        energy=energy,
        entanglement_profile=profile,
        max_entropy=max_s,
        area_law_limit=area_lim,
        likely_area_law=max_s < area_lim,
        notes=(
            "area-law classification is heuristic (discovery-tier); "
            "continuum limit (χ→∞, L→∞) is [OUT]"
        ),
    )
