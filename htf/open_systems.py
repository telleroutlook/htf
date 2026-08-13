"""HTF §4-H — Open quantum systems and CPTP maps.

Provides density matrix operations, Lindblad master equation integration,
and structure verification for completely positive trace-preserving (CPTP)
channels.

Key functions
-------------
density_matrix_from_pure  — |ψ⟩⟨ψ| / norm².
partial_trace             — Tr_B(ρ) keeping arbitrary sites.
check_density_matrix      — verify ρ is Hermitian, PSD, unit-trace.
choi_matrix               — Choi–Jamiołkowski matrix of a Kraus channel.
check_kraus_completeness  — verify Σ_k K_k† K_k = I.
lindblad_superoperator    — d²×d² Lindblad superoperator matrix.
lindblad_step             — exact integration ρ(t+dt) via matrix expm.
steady_state              — find ρ_ss with L(ρ_ss) = 0, Tr(ρ_ss) = 1.

Honest scope
------------
* All computations are float mode (discovery-tier).
* Certified error bounds are not yet propagated through the Lindblad
  evolution (that would require interval-arithmetic Padé expm).
* Only Markovian (memoryless) Lindblad dynamics is handled.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .structure import StructureReport

# ─────────────────────── density matrix ──────────────────────────────────

def density_matrix_from_pure(state_vec: np.ndarray) -> np.ndarray:
    """Pure-state density matrix ρ = |ψ⟩⟨ψ| / ⟨ψ|ψ⟩.

    Parameters
    ----------
    state_vec : 1-D array of length d (normalised internally).

    Returns
    -------
    ρ : (d × d) real or complex Hermitian PSD matrix with Tr = 1.
    """
    psi = np.asarray(state_vec, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(psi))
    if norm < 1e-15:
        raise ValueError("state vector has zero norm")
    psi = psi / norm
    return np.outer(psi, psi.conj())


def partial_trace(
    rho: np.ndarray,
    n_sites: int,
    keep_sites: list[int],
    d: int = 2,
) -> np.ndarray:
    """Reduce ρ by tracing out all sites not in *keep_sites*.

    Parameters
    ----------
    rho        : (d^n × d^n) density matrix.
    n_sites    : total number of sites n.
    keep_sites : sites to retain (0-indexed, big-endian ordering).
    d          : local Hilbert-space dimension (default 2).

    Returns
    -------
    ρ_A : (d^|keep| × d^|keep|) reduced density matrix.
    """
    keep = sorted(keep_sites)
    trace_out = sorted(set(range(n_sites)) - set(keep))
    n_keep  = len(keep)
    n_trace = len(trace_out)
    dim_keep  = d ** n_keep
    dim_trace = d ** n_trace

    rho_t = np.asarray(rho, dtype=complex).reshape([d] * (2 * n_sites))

    # Permute: keep_row, trace_row, keep_col, trace_col
    row_perm = keep + trace_out
    col_perm = [k + n_sites for k in keep] + [k + n_sites for k in trace_out]
    rho_perm = np.transpose(rho_t, row_perm + col_perm)
    rho_perm = rho_perm.reshape(dim_keep, dim_trace, dim_keep, dim_trace)

    # Trace over trace_out: result[i, k] = Σ_j rho_perm[i, j, k, j]
    return np.einsum("abcb->ac", rho_perm)


# ─────────────────────── density matrix check ────────────────────────────

def check_density_matrix(
    rho: np.ndarray,
    tol: float = 1e-10,
) -> dict:
    """Verify that *rho* is a valid density matrix.

    Checks three independent properties:

    hermitian   — ``||ρ − ρ†||_max ≤ tol``
    psd         — min eigenvalue ≥ −tol
    unit_trace  — ``|Tr(ρ) − 1| ≤ tol``

    Returns
    -------
    dict with keys ``hermitian``, ``psd``, ``unit_trace``
    (:class:`~htf.structure.StructureReport` each) and ``all_passed`` (bool).
    """
    rho = np.asarray(rho, dtype=complex)

    herm_defect = float(np.abs(rho - rho.conj().T).max())
    r_herm = StructureReport(
        property_name="density_matrix_hermitian",
        passed=herm_defect <= tol,
        defect=herm_defect,
        tolerance=tol,
        notes=f"||ρ−ρ†||_max={herm_defect:.3e}",
    )

    min_ev = float(np.linalg.eigvalsh(rho).min())
    r_psd = StructureReport(
        property_name="density_matrix_psd",
        passed=min_ev >= -tol,
        defect=max(0.0, -min_ev),
        tolerance=tol,
        notes=f"min_eig={min_ev:.3e}",
    )

    trace_defect = float(abs(np.trace(rho) - 1.0))
    r_trace = StructureReport(
        property_name="density_matrix_unit_trace",
        passed=trace_defect <= tol,
        defect=trace_defect,
        tolerance=tol,
        notes=f"|Tr(ρ)−1|={trace_defect:.3e}",
    )

    return {
        "hermitian":  r_herm,
        "psd":        r_psd,
        "unit_trace": r_trace,
        "all_passed": r_herm.passed and r_psd.passed and r_trace.passed,
    }


# ─────────────────────── Kraus / CPTP ────────────────────────────────────

def choi_matrix(kraus_ops: list[np.ndarray]) -> np.ndarray:
    """Choi–Jamiołkowski matrix of the channel Φ(ρ) = Σ_k K_k ρ K_k†.

    Convention: J[i*d+j, k*d+l] = Φ(|i⟩⟨j|)[k, l]

    Properties:
    * Φ is CP  ⟺  J ≥ 0  (always true for Kraus-given maps).
    * Φ is TP  ⟺  Σ_i Φ(|i⟩⟨i|) = I, i.e. the block-diagonal of J sums to I.

    Parameters
    ----------
    kraus_ops : list of (d_out × d_in) Kraus operators.

    Returns
    -------
    J : (d_in*d_out × d_in*d_out) complex matrix.
    """
    kraus_ops = [np.asarray(K, dtype=complex) for K in kraus_ops]
    d_out, d_in = kraus_ops[0].shape
    J = np.zeros((d_in * d_out, d_in * d_out), dtype=complex)
    E = np.zeros((d_in, d_in), dtype=complex)
    for i in range(d_in):
        for j in range(d_in):
            E[i, j] = 1.0
            phi_E = sum(K @ E @ K.conj().T for K in kraus_ops)
            J[i * d_out:(i + 1) * d_out, j * d_out:(j + 1) * d_out] = phi_E
            E[i, j] = 0.0
    return J


def check_kraus_completeness(
    kraus_ops: list[np.ndarray],
    tol: float = 1e-10,
) -> StructureReport:
    """Check the trace-preserving completeness relation Σ_k K_k† K_k = I.

    Returns a :class:`~htf.structure.StructureReport` with
    defect = ``||Σ K†K − I||_max``.
    """
    kraus_ops = [np.asarray(K, dtype=complex) for K in kraus_ops]
    d_in = kraus_ops[0].shape[1]
    total = sum(K.conj().T @ K for K in kraus_ops)
    defect = float(np.abs(total - np.eye(d_in)).max())
    return StructureReport(
        property_name="kraus_completeness",
        passed=defect <= tol,
        defect=defect,
        tolerance=tol,
        notes=f"||Σ K†K − I||_max={defect:.3e}  (n_kraus={len(kraus_ops)})",
    )


# ─────────────────────── Lindblad ────────────────────────────────────────

def lindblad_superoperator(
    ham: np.ndarray,
    lindblad_ops: list[np.ndarray],
) -> np.ndarray:
    """Full Lindblad superoperator as a (d²×d²) matrix.

    In column-vectorized form  vec(ρ) = ρ.flatten('F') (Fortran order),
    the evolution dρ/dt = L(ρ) becomes d/dt vec(ρ) = L_super @ vec(ρ)
    where::

        L_super = −i(I⊗H − H.T⊗I)
                + Σ_k [ L_k*⊗L_k − ½(I⊗L_k†L_k) − ½(L_k†L_k).T⊗I ]

    Parameters
    ----------
    ham          : (d×d) Hamiltonian (Hermitian).
    lindblad_ops : list of (d×d) jump operators L_k.

    Returns
    -------
    L_super : (d²×d²) complex matrix.
    """
    H = np.asarray(ham, dtype=complex)
    d = H.shape[0]
    I = np.eye(d, dtype=complex)

    L_super = -1j * (np.kron(I, H) - np.kron(H.T, I))

    for Lk in lindblad_ops:
        Lk = np.asarray(Lk, dtype=complex)
        LkdLk = Lk.conj().T @ Lk
        L_super += (
            np.kron(Lk.conj(), Lk)
            - 0.5 * np.kron(I, LkdLk)
            - 0.5 * np.kron(LkdLk.T, I)
        )
    return L_super


def lindblad_step(
    rho: np.ndarray,
    ham: np.ndarray,
    lindblad_ops: list[np.ndarray],
    dt: float,
) -> np.ndarray:
    """Exact Lindblad step ρ(t+dt) = exp(dt L)[ρ] via matrix exponentiation.

    This is exact for time-independent H and {L_k}.  For large d or many
    steps, use the steady_state solver for the asymptotic limit instead.

    Returns
    -------
    ρ(t+dt) : (d×d) density matrix.
    """
    d = ham.shape[0]
    L_super = lindblad_superoperator(ham, lindblad_ops)
    rho_vec = np.asarray(rho, dtype=complex).flatten(order='F')
    rho_new_vec = expm(dt * L_super) @ rho_vec
    return rho_new_vec.reshape(d, d, order='F')


def steady_state(
    ham: np.ndarray,
    lindblad_ops: list[np.ndarray],
    tol: float = 1e-12,
) -> np.ndarray:
    """Steady-state density matrix ρ_ss satisfying L(ρ_ss) = 0, Tr(ρ_ss) = 1.

    Solves the constrained linear system by replacing the first row of the
    superoperator with the trace-normalisation constraint.

    Returns
    -------
    ρ_ss : (d×d) density matrix.  Uniqueness is guaranteed for ergodic
           (full-rank Lindblad) dynamics; degenerate cases return one solution.
    """
    d = ham.shape[0]
    L_super = lindblad_superoperator(ham, lindblad_ops)

    # Replace first equation with Tr(ρ) = 1
    # In column-vectorized form (Fortran order), ρ[i,i] → index i + i*d = i*(d+1)
    A = L_super.copy()
    b = np.zeros(d * d, dtype=complex)
    trace_row = np.zeros(d * d, dtype=complex)
    for i in range(d):
        trace_row[i + i * d] = 1.0
    A[0] = trace_row
    b[0] = 1.0

    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x.reshape(d, d, order='F')
