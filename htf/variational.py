"""HTF Phase 3 — Variational ground-state search.

Provides:

* Hamiltonian builders for small 1-D models (dense ``2^n × 2^n`` matrices).
* Energy expectation value (float and **certified** via flint Arb).
* Simple L-BFGS-B optimiser for MERA parameters.

Honest scope
------------
``variational_bound`` certifies **floating-point rounding** in the energy
computation (Phase 3 scope).  Bond-dimension truncation bias (``chi → ∞``)
is *not* certified here — that is Phase 4 territory.  The variational
principle ensures that ``E_var ≥ E_0`` (true ground-state energy) only when
the state is normalised and the Hamiltonian is exact; the certified
``error_bound`` covers the rounding on top of that variational estimate.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .certificate import Certificate

if TYPE_CHECKING:
    from .mera import MERA

# ─────────────────────── Hamiltonian builders ─────────────────────

def _kron_op(ops: list[np.ndarray]) -> np.ndarray:
    """Kronecker product of a list of 2×2 matrices → (2^n × 2^n)."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def transverse_ising_ham(n: int, J: float = 1.0, h: float = 0.5) -> np.ndarray:
    """Dense 1-D TFIM Hamiltonian: ``H = -J Σ Z_i Z_{i+1} - h Σ X_i``.

    Returns a real symmetric ``(2^n × 2^n)`` matrix.
    Practical limit: ``n ≤ 16`` (RAM scales as ``4^n`` bytes).
    """
    I = np.eye(2, dtype=float)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])
    X = np.array([[0.0, 1.0], [1.0, 0.0]])

    dim = 2 ** n
    H = np.zeros((dim, dim))

    # ZZ nearest-neighbour terms
    for i in range(n - 1):
        ops = [Z if j == i else (Z if j == i + 1 else I) for j in range(n)]
        H -= J * _kron_op(ops)

    # Transverse-field X terms
    for i in range(n):
        ops = [X if j == i else I for j in range(n)]
        H -= h * _kron_op(ops)

    return H


def xx_model_ham(n: int, J: float = 1.0) -> np.ndarray:
    """Dense 1-D XX Hamiltonian: ``H = -J Σ (X_i X_{i+1} + Y_i Y_{i+1}) / 2``.

    Returns a real symmetric ``(2^n × 2^n)`` matrix.
    """
    I = np.eye(2, dtype=float)
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    # Y_real = [[0,-1],[1,0]] = -i * Y_Pauli, so Y_real⊗Y_real = -(Y_Pauli⊗Y_Pauli).
    # To get X⊗X + Y_Pauli⊗Y_Pauli we must SUBTRACT Y_real⊗Y_real.
    Y = np.array([[0.0, -1.0], [1.0, 0.0]])

    dim = 2 ** n
    H = np.zeros((dim, dim))

    for i in range(n - 1):
        opsX = [X if j == i else (X if j == i + 1 else I) for j in range(n)]
        opsY = [Y if j == i else (Y if j == i + 1 else I) for j in range(n)]
        H -= J * 0.5 * (_kron_op(opsX) - _kron_op(opsY))

    return H


# ──────────────────── energy computation ──────────────────────────

def energy_expectation(ham: np.ndarray, state_vec: np.ndarray) -> float:
    """Variational energy ``⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩`` (float, discovery-tier)."""
    psi = np.asarray(state_vec, dtype=float).reshape(-1)
    norm_sq = float(psi @ psi)
    if norm_sq < 1e-15:
        raise ValueError("state vector has zero norm")
    return float(psi @ ham @ psi) / norm_sq


def variational_bound(
    ham: np.ndarray,
    mera: MERA,
) -> Certificate:
    """Certified upper bound on ground-state energy via variational principle.

    Uses flint Arb to bound the floating-point rounding error in
    ``⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩`` where ``|ψ⟩`` is the MERA state vector.

    Returns a :class:`~htf.certificate.Certificate` with:

    * ``result``      — variational energy midpoint.
    * ``error_bound`` — rigorous rounding bound (``≥ 0``).

    The variational principle guarantees ``E_var ≥ E_0``, so
    ``E_0 ≤ result + error_bound`` is a certified upper bound on the
    ground-state energy — **provided the state is properly normalised and
    the Hamiltonian is exact**.  Bond-dimension bias is ``[OUT]``.
    """
    try:
        from flint import arb, arb_mat
    except ImportError as exc:
        raise ImportError("variational_bound requires python-flint") from exc

    from ._rayleigh_primitives import _check_preconditions

    psi = mera.state_vector()
    n = len(psi)

    # Precondition checks: ham must be self-adjoint, finite, same dimension as psi.
    # Without these, the Rayleigh-Ritz theorem does not apply and a
    # Certificate(mode="certified") would carry an unsound claim.
    if ham.shape != (n, n):
        raise ValueError(
            f"ham shape {ham.shape} does not match MERA state dimension ({n},)"
        )
    _check_preconditions(np.asarray(ham, dtype=float), psi)

    psi_row = arb_mat([[arb(float(psi[i])) for i in range(n)]])
    psi_col = arb_mat([[arb(float(psi[i]))] for i in range(n)])
    H_mat = arb_mat([[arb(float(ham[i, j])) for j in range(n)] for i in range(n)])

    # ⟨ψ|H|ψ⟩ and ⟨ψ|ψ⟩ via matrix products
    Hpsi = H_mat * psi_col            # (n, 1)
    numerator_mat = psi_row * Hpsi    # (1, 1)
    denominator_mat = psi_row * psi_col  # (1, 1)

    energy_arb = numerator_mat[0, 0] / denominator_mat[0, 0]

    return Certificate(
        result=float(energy_arb.mid()),
        mode="certified",
        error_bound=float(energy_arb.rad()),
        backend="flint-arb",
        notes=(
            "variational upper bound on ground-state energy; "
            "certifies floating-point rounding only; "
            "bond-dimension truncation bias is [OUT] (Phase 4 scope)"
        ),
    )


# ──────────────────── optimiser ───────────────────────────────────

def optimize_mera(
    ham: np.ndarray,
    mera: MERA,
    n_iter: int = 50,
    tol: float = 1e-5,
) -> tuple[MERA, list[float]]:
    """Minimise ``⟨ψ(θ)|H|ψ(θ)⟩ / ⟨ψ|ψ⟩`` using L-BFGS-B + retraction.

    After optimisation the returned MERA has its constraints re-enforced
    (isometries and disentanglers projected back via SVD polar factor).

    Parameters
    ----------
    ham    : dense Hamiltonian ``(2^n × 2^n)`` matrix.
    mera   : initial MERA (not modified in-place).
    n_iter : maximum number of L-BFGS-B iterations.
    tol    : convergence tolerance (``ftol`` and ``gtol``).

    Returns
    -------
    (optimised_mera, energy_history)
        *energy_history* contains the energy at each callback step (at
        least the initial energy plus the final energy).
    """
    from scipy.optimize import minimize

    history: list[float] = [energy_expectation(ham, mera.state_vector())]

    def objective(params: np.ndarray) -> float:
        m = mera.from_flat_params(params)
        return energy_expectation(ham, m.state_vector())

    def callback(xk: np.ndarray) -> None:
        history.append(objective(xk))

    p0 = mera.to_flat_params()
    result = minimize(
        objective,
        p0,
        method="L-BFGS-B",
        callback=callback,
        options={"maxiter": n_iter, "ftol": tol, "gtol": tol},
    )

    optimised = mera.from_flat_params(result.x)
    optimised.enforce_constraints()

    # Ensure final energy is in history
    e_final = energy_expectation(ham, optimised.state_vector())
    if not history or abs(history[-1] - e_final) > 1e-10:
        history.append(e_final)

    return optimised, history
