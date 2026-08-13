"""HTF §9-D / §9-J — Finite-temperature thermal states via MPS purification.

Algorithm: imaginary-time TEBD on the purified (physical ⊗ ancilla) MPS.

Starting from the maximally entangled state |Φ⟩ = ⊗_i |bell_i⟩ (β=0,
infinite temperature), apply e^{-βH/2} on physical indices only.  The
resulting state represents ρ(β) = e^{-βH}/Z with
    Z(β) = Tr(e^{-βH})  and  ⟨O⟩_β = Tr(O e^{-βH}) / Z(β).

Super-site convention: each site has dimension D = d × d (physical ⊗ ancilla).
The physical Hamiltonian is embedded as H_ext = H_phys ⊗ I_anc.

Honest scope
------------
* Truncation error from χ-limited TEBD is not certified.  ``[工程]``
* Trotter error is O(dt²) per step (1st-order TEBD); reduce dt to improve.
* The partition function Z is estimated by tracking accumulated MPS norms
  during normalised imaginary-time evolution.
* Continuum limit is ``[OUT]``.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .mps import MPS, mps_expectation, mps_inner, mps_norm, mps_normalise
from .tebd import _nn_energy, tebd_step


@dataclass
class ThermalResult:
    """Result of a finite-temperature thermal-state calculation.

    Attributes
    ----------
    mps_purified:       Normalized purified MPS at inverse temperature ``beta``.
                        Thermal expectations are computed from this state.
    beta:               Requested inverse temperature β = 1/(kT).
    beta_achieved:      Actual β after step discretisation.
    partition_function: Z(β) = Tr(e^{-βH}), reconstructed from accumulated
                        MPS norms.
    free_energy_upper:  F = -ln(Z)/β.  Upper bound on true free energy
                        because MPS truncation overestimates Z.  ``[工程]``
    energies:           Thermal energy E(β') at each measurement checkpoint
                        (recorded every ``measure_every`` steps plus final).
    """
    mps_purified:       MPS
    beta:               float
    beta_achieved:      float
    partition_function: float
    free_energy_upper:  float
    energies:           list[float] = field(default_factory=list)


# ── building blocks ───────────────────────────────────────────────────────


def purified_initial_mps(n: int, d: int = 2) -> MPS:
    """Create the β=0 (infinite-T) maximally entangled purified MPS.

    Each super-site has dimension D = d² (physical ⊗ ancilla).  The
    tensor is the normalised Bell pair I_d / √d.
    """
    D    = d * d
    bell = np.eye(d, dtype=float).ravel() / np.sqrt(d)   # norm = 1
    return MPS([bell.reshape(1, D, 1).copy() for _ in range(n)])


def purification_bonds(
    h_terms: list[np.ndarray],
    d: int = 2,
) -> list[np.ndarray]:
    """Extend physical bond Hamiltonians to the purified D = d²-dim super-sites.

    For each bond h (shape d²×d²), the extended bond acts on physical
    indices only:
        h_ext[(s1,a1)(s2,a2), (s1',a1')(s2',a2')] = h[s1,s2;s1',s2'] δ(a1,a1') δ(a2,a2')
    """
    I_d = np.eye(d, dtype=float)
    D   = d * d
    result: list[np.ndarray] = []
    for h in h_terms:
        h4    = h.reshape(d, d, d, d)
        # output indices: (s1,a1,s2,a2,s1',a1',s2',a2')
        h_ext = np.einsum("ijkl,mn,op->imjoknlp", h4, I_d, I_d, optimize=True)
        result.append(h_ext.reshape(D ** 2, D ** 2))
    return result


# ── main API ──────────────────────────────────────────────────────────────


def thermal_state(
    h_terms: list[np.ndarray],
    n: int,
    beta: float,
    chi: int = 16,
    d: int = 2,
    dt: float = 0.05,
    measure_every: int = 10,
) -> ThermalResult:
    """Compute the thermal state ρ(β) = e^{-βH}/Z via imaginary-time TEBD.

    Parameters
    ----------
    h_terms:      nearest-neighbour bond Hamiltonians (shape d²×d² each).
    n:            number of lattice sites.
    beta:         target inverse temperature (β = 1/kT; β→∞ → ground state).
    chi:          MPS bond-dimension cap for TEBD truncation.
    d:            physical site dimension.
    dt:           imaginary-time step; Trotter error ∝ dt².
    measure_every: record thermal energy every this many TEBD steps plus final.

    Returns
    -------
    :class:`ThermalResult` with normalised purified MPS and Z estimate.
    """
    mps       = purified_initial_mps(n, d)
    bonds_ext = purification_bonds(h_terms, d)
    n_steps   = round(beta / (2.0 * dt))

    # β = 0 edge case: return the infinite-T state immediately
    if n_steps == 0:
        Z = float(d ** n)
        return ThermalResult(
            mps_purified=mps,
            beta=beta,
            beta_achieved=0.0,
            partition_function=Z,
            free_energy_upper=float("nan"),
            energies=[float(_nn_energy(mps, bonds_ext))],
        )

    dt_actual      = beta / (2.0 * n_steps)
    log_norm_accum = 0.0
    energies: list[float] = []

    for step in range(n_steps):
        if step % measure_every == 0:
            energies.append(float(_nn_energy(mps, bonds_ext)))

        mps_new, _ = tebd_step(mps, bonds_ext, dt=dt_actual, chi=chi, imaginary=True)
        nrm = float(mps_norm(mps_new))
        if nrm > 0.0:
            log_norm_accum += np.log(nrm)
        mps = mps_normalise(mps_new)

    energies.append(float(_nn_energy(mps, bonds_ext)))
    beta_achieved = 2.0 * n_steps * dt_actual

    # Z(β) = ||Φ||² · exp(2 · Σ log nrm_k)
    # ||Φ||² = d^n  (unnormalized Bell product has norm² = d per site)
    Z = float(d ** n) * float(np.exp(2.0 * log_norm_accum))
    F = (
        -np.log(max(Z, 1e-300)) / beta_achieved
        if beta_achieved > 0 and Z > 0
        else float("nan")
    )

    return ThermalResult(
        mps_purified=mps,
        beta=beta,
        beta_achieved=beta_achieved,
        partition_function=Z,
        free_energy_upper=F,
        energies=energies,
    )


def thermal_expectation(
    mps_purified: MPS,
    operator: np.ndarray,
    site: int,
    d: int = 2,
) -> float:
    """Compute ⟨O⟩_β = Tr(O e^{-βH}) / Z for a single-site physical operator.

    Parameters
    ----------
    mps_purified: normalised purified MPS from :func:`thermal_state`.
    operator:     single-site physical operator of shape (d, d).
    site:         lattice site index (0-based).
    d:            physical dimension.
    """
    O_ext = np.kron(operator, np.eye(d))          # (d², d²): identity on ancilla
    num   = float(mps_expectation(mps_purified, [(site, O_ext)]).real)
    den   = float(mps_inner(mps_purified, mps_purified).real)
    return num / den


# ── §9-J: parallel β-scan ────────────────────────────────────────────────────


@dataclass
class ThermalScanPoint:
    """One temperature point in a parallel β-scan.

    Attributes
    ----------
    beta:              Inverse temperature β = 1/(kT) actually achieved.
    energy:            Thermal energy E(β) = ⟨H⟩_β.
    free_energy:       F(β) = -ln Z / β (upper bound; heuristic ``[工程]``).
    partition_function: Z(β) = Tr(e^{-βH}).
    """
    beta:               float
    energy:             float
    free_energy:        float
    partition_function: float


@dataclass
class ThermalScanResult:
    """Result of a parallel β-scan.

    Attributes
    ----------
    points:    :class:`ThermalScanPoint` list, sorted by β (ascending).
    n_workers: Number of parallel worker processes used.
    """
    points:    list = field(default_factory=list)
    n_workers: int  = 1

    def summary(self) -> str:
        lines = [
            "Thermal β-scan:",
            f"{'beta':>8}  {'E(beta)':>14}  {'F(beta)':>14}  {'Z':>12}",
        ]
        for p in self.points:
            lines.append(
                f"{p.beta:>8.3f}  {p.energy:>14.8f}"
                f"  {p.free_energy:>14.8f}  {p.partition_function:>12.4e}"
            )
        return "\n".join(lines)


def _thermal_scan_worker(packed):
    """Top-level picklable worker for thermal_scan."""
    h_terms, n, beta, chi, d, dt = packed
    result = thermal_state(h_terms, n, beta, chi=chi, d=d, dt=dt)
    return ThermalScanPoint(
        beta=result.beta_achieved,
        energy=result.energies[-1] if result.energies else float("nan"),
        free_energy=result.free_energy_upper,
        partition_function=result.partition_function,
    )


def thermal_scan(
    h_terms: list,
    n: int,
    beta_list: list,
    chi: int = 16,
    d: int = 2,
    dt: float = 0.05,
    n_workers: int | None = None,
) -> ThermalScanResult:
    """Parallel β-scan via imaginary-time TEBD.

    Runs :func:`thermal_state` independently at every β in *beta_list*
    using ``ProcessPoolExecutor``.  Each run starts from the same β=0
    infinite-temperature state and cools independently, so all β values
    are fully parallel — the total wall-clock time is the cost of the
    *single longest run* (largest β) rather than their sum.

    Parameters
    ----------
    h_terms   : nearest-neighbour bond Hamiltonians (shape d²×d² each).
    n         : number of lattice sites.
    beta_list : inverse temperatures to compute (any order; output is
                sorted by β ascending).
    chi       : MPS bond-dimension cap for TEBD truncation.
    d         : physical site dimension.
    dt        : imaginary-time step per TEBD step.
    n_workers : Worker processes.  ``None`` uses ``os.cpu_count()``.
                Pass ``1`` to run sequentially.

    Returns
    -------
    :class:`ThermalScanResult` with ``points`` sorted by β.
    """
    packed_list = [
        (h_terms, n, beta, chi, d, dt)
        for beta in beta_list
    ]

    effective_workers = n_workers if n_workers is not None else (os.cpu_count() or 1)

    if effective_workers == 1 or len(packed_list) == 1:
        points = [_thermal_scan_worker(p) for p in packed_list]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            points = list(pool.map(_thermal_scan_worker, packed_list))

    points.sort(key=lambda p: p.beta)
    return ThermalScanResult(points=points, n_workers=effective_workers)
