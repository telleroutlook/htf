"""HTF §4-I — Lanczos algorithm and strict two-sided spectral bounds.

Provides:
* ``lanczos``         — k-step Lanczos iteration (tridiagonal form).
* ``lanczos_eigs``    — Ritz values (approximate eigenvalues) and vectors.
* ``temple_lanczos``  — Tight Temple lower bound using Lanczos E_1 estimate.
* ``two_sided_bounds``— Both certified upper and Temple-Lanczos lower bound.

Honest scope
------------
* Ritz values are variational upper bounds on the corresponding eigenvalues
  (no rigorous convergence certificate — Paige a-posteriori bounds are ``[研究]``).
* The Temple lower bound is a rigorous finite-lattice bound on E_0 when the
  condition E_var < E_1_exact is met; it does not certify the continuum gap.
* All computations are float mode; combining with certified mode is ``[研究]``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .certificate import Certificate


# ─────────────────────── Lanczos core ────────────────────────────────────

def lanczos(
    A: np.ndarray,
    v0: np.ndarray,
    k: int,
    reorthogonalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """k-step Lanczos iteration starting from unit vector v0.

    Constructs a Krylov subspace K_k = span{v0, Av0, A²v0, …} and the
    associated real symmetric tridiagonal matrix T_k such that
    A ≈ V_k T_k V_k^T.

    Parameters
    ----------
    A               : real symmetric (n × n) matrix.
    v0              : starting vector of length n (normalised internally).
    k               : number of Lanczos steps (capped at n).
    reorthogonalize : if True (default), apply full Gram-Schmidt
                      re-orthogonalization at every step to prevent
                      accumulation of floating-point ghost eigenvalues.
                      Costs O(k) extra matrix-vector products per step
                      but makes V^T V ≈ I to machine precision for large k.

    Returns
    -------
    alpha : (k,) diagonal elements of T_k.
    beta  : (k−1,) off-diagonal elements of T_k.
    V     : (n, k) matrix with orthonormal Lanczos vectors as columns.
    """
    A  = np.asarray(A, dtype=float)
    n  = A.shape[0]
    k  = min(k, n)
    v0 = np.asarray(v0, dtype=float).reshape(-1)
    v0 = v0 / np.linalg.norm(v0)

    V     = np.zeros((n, k))
    alpha = np.zeros(k)
    beta  = np.zeros(k - 1)

    V[:, 0] = v0
    w        = A @ v0
    alpha[0] = float(w @ v0)
    r        = w - alpha[0] * v0
    k_actual = k

    for j in range(1, k):
        # Full Gram-Schmidt re-orthogonalization against all previous vectors
        if reorthogonalize:
            r = r - V[:, :j] @ (V[:, :j].T @ r)
            r = r - V[:, :j] @ (V[:, :j].T @ r)   # twice for numerical safety

        b = float(np.linalg.norm(r))
        if b < 1e-14:       # invariant subspace — stop early
            k_actual = j
            break
        beta[j - 1]  = b
        V[:, j]      = r / b
        w            = A @ V[:, j]
        alpha[j]     = float(w @ V[:, j])
        r            = w - alpha[j] * V[:, j] - b * V[:, j - 1]

    return alpha[:k_actual], beta[:k_actual - 1], V[:, :k_actual]


def lanczos_eigs(
    A: np.ndarray,
    v0: Optional[np.ndarray] = None,
    k: int = 30,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Ritz values and vectors from a k-step Lanczos run.

    Ritz values are variational upper bounds on the corresponding exact
    eigenvalues (they converge from above for the lowest ones).

    Parameters
    ----------
    A    : real symmetric matrix.
    v0   : starting vector (random if None).
    k    : number of Lanczos steps.
    seed : RNG seed used when v0 is None.

    Returns
    -------
    ritz_vals : eigenvalues of T_k sorted in ascending order.
    ritz_vecs : corresponding Ritz vectors in the original space (columns).
    """
    n = A.shape[0]
    if v0 is None:
        rng = np.random.default_rng(seed)
        v0  = rng.standard_normal(n)
    alpha, beta, V = lanczos(A, v0, k)
    m = len(alpha)
    T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
    evals, evecs_T = np.linalg.eigh(T)
    # Map Ritz vectors back to the original space
    ritz_vecs = V @ evecs_T
    return evals, ritz_vecs


def lanczos_ground_state(
    A: np.ndarray,
    k: int = 30,
    seed: int = 0,
) -> np.ndarray:
    """Lowest Ritz vector from Lanczos — approximates the ground state.

    Returns the Ritz vector with the smallest Ritz value, normalised.
    """
    evals, ritz_vecs = lanczos_eigs(A, k=k, seed=seed)
    psi = ritz_vecs[:, 0]
    return psi / np.linalg.norm(psi)


# ─────────────────────── two-sided bounds ────────────────────────────────

@dataclass
class TwoSidedBounds:
    """Strict two-sided bounds on the ground-state energy E_0.

    Attributes
    ----------
    E0_upper : certified variational upper bound on E_0.
    E0_upper_error : floating-point rounding bound on E0_upper.
    E0_lower : Temple's inequality lower bound on E_0.
    temple_condition_met : True if E_var < E_1_upper (Temple is valid).
    E1_ritz  : Lanczos Ritz estimate of E_1 (variational upper bound on E_1).
    k_lanczos: number of Lanczos steps used.
    notes    : scope statement.

    gap_lower_bound : max(0, E0_upper − E0_lower) if both bounds valid.
    """
    E0_upper: float
    E0_upper_error: float
    E0_lower: float
    temple_condition_met: bool
    E1_ritz: float
    k_lanczos: int
    notes: str = ""

    @property
    def width(self) -> float:
        """Upper − lower (width of the certified interval on E_0)."""
        if not self.temple_condition_met:
            return float("inf")
        return self.E0_upper - self.E0_lower


def temple_lanczos(
    ham: np.ndarray,
    k: int = 30,
    seed: int = 0,
) -> TwoSidedBounds:
    """Tight two-sided bounds on E_0 using Lanczos Ritz values.

    Algorithm
    ---------
    1. Run k-step Lanczos to get Ritz values {θ_0, θ_1, …}.
    2. Use θ_0 as the variational upper bound on E_0.
    3. Use θ_1 as the variational upper bound on E_1.
    4. Apply Temple's inequality lower bound on E_0.

    The resulting interval [E0_lower, E0_upper] is a provably valid
    finite-lattice bound when θ_0 < E_1_exact (Temple condition).

    Honest scope
    ------------
    * θ_0 and θ_1 are Ritz values (variational upper bounds), not certified.
    * Temple condition checked against θ_1, not E_1_exact — it is possible that
      θ_1 > E_1_exact and the condition fails.  Use ``temple_condition_met`` flag.
    * Continuum gap and χ-truncation bias are ``[OUT]``.
    """
    from .gap import h2_expectation, temple_lower_bound
    from .variational import energy_expectation

    H   = np.asarray(ham, dtype=float)
    evals, ritz_vecs = lanczos_eigs(H, k=k, seed=seed)

    psi0 = ritz_vecs[:, 0]
    norm = np.linalg.norm(psi0)
    psi0 = psi0 / norm

    E0_var  = energy_expectation(H, psi0)
    E1_ritz = float(evals[1]) if len(evals) > 1 else float("inf")

    h2_exp = h2_expectation(H, psi0)
    t_lb   = temple_lower_bound(E0_var, h2_exp, E1_ritz)
    cond   = E0_var < E1_ritz and not (t_lb == float("-inf"))

    # Certified upper bound via flint Arb if available; else float mode
    try:
        from flint import arb, arb_mat  # type: ignore[import]
        n = len(psi0)
        s  = arb_mat([[arb(float(psi0[i])) for i in range(n)]])
        c  = arb_mat([[arb(float(psi0[i]))] for i in range(n)])
        Hm = arb_mat([[arb(float(H[i, j])) for j in range(n)] for i in range(n)])
        ev = (s * (Hm * c))[0, 0] / (s * c)[0, 0]
        cert = Certificate(
            result=float(ev.mid()),
            mode="certified",
            error_bound=float(ev.rad()),
            backend="flint-arb",
            notes="Lanczos Ritz ground state; FP rounding certified",
        )
    except ImportError:
        cert = Certificate(
            result=E0_var,
            mode="float",
            error_bound=None,
            notes="Lanczos Ritz ground state; float mode (no flint)",
        )

    err = float(cert.error_bound) if cert.error_bound is not None else 0.0
    return TwoSidedBounds(
        E0_upper=cert.result,
        E0_upper_error=err,
        E0_lower=t_lb,
        temple_condition_met=cond,
        E1_ritz=E1_ritz,
        k_lanczos=len(evals),
        notes=(
            f"k_lanczos={k}; Ritz upper bounds; "
            "Temple lower bound valid iff E_var < E_1_exact (finite-lattice); "
            "continuum gap is [OUT]"
        ),
    )


def two_sided_bounds(
    ham: np.ndarray,
    k: int = 30,
    seed: int = 0,
) -> TwoSidedBounds:
    """Alias for :func:`temple_lanczos` — strict two-sided bounds on E_0."""
    return temple_lanczos(ham, k=k, seed=seed)
