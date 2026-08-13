"""HTF §4-J — Differentiable inverse design and Hamiltonian learning.

Provides two complementary workflows:

1. **Inverse design** — given a target ground-state energy ``E_target``,
   find Hamiltonian parameters ``(J, h)`` (TFIM) or ``(J,)`` (XX) that
   minimise ``|E_0(params) - E_target|``.

2. **Hamiltonian learning** — given a set of observed energies (e.g. from
   an experiment or a reference solver), recover the parameters of a
   known model family that best reproduces them.

Honest scope
------------
* Optimisation is by L-BFGS-B.  When JAX is installed, ``energy_gradient``
  uses exact autodiff (``jax.grad`` through ``jnp.linalg.eigvalsh``) for the
  built-in ``ising`` and ``xx`` models; otherwise it falls back to centred
  finite differences.  ``[工程]``
* Identifiability (uniqueness of the recovered parameters) depends on the
  observable set and is not guaranteed — use ``n_restarts`` to mitigate
  local minima.
* All energies are float-mode; certifying the inversion result is ``[研究]``.
* Continuum-limit guarantees are ``[OUT]``.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from .variational import transverse_ising_ham, xx_model_ham

# ────────────────────── parametric Hamiltonian ────────────────────────────

@dataclass
class ParametricHam:
    """A parametric Hamiltonian family with named scalar parameters.

    Attributes
    ----------
    model    : ``"ising"`` or ``"xx"``.
    n_sites  : lattice size.
    param_names : ordered list of optimisable parameter names.

    Call ``ham(params)`` to build the matrix for a given parameter vector.
    """
    model:       str
    n_sites:     int
    param_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.model == "ising":
            self.param_names = self.param_names or ["J", "h"]
        elif self.model == "xx":
            self.param_names = self.param_names or ["J"]
        else:
            raise ValueError(f"Unknown model '{self.model}'. Choose 'ising' or 'xx'.")

    def ham(self, params: Sequence[float] | np.ndarray) -> np.ndarray:
        """Build the Hamiltonian matrix for the given parameter vector."""
        p = list(params)
        if self.model == "ising":
            J = p[0] if len(p) > 0 else 1.0
            h = p[1] if len(p) > 1 else 0.5
            return transverse_ising_ham(self.n_sites, J=J, h=h)
        if self.model == "xx":
            J = p[0] if len(p) > 0 else 1.0
            return xx_model_ham(self.n_sites, J=J)
        raise ValueError(f"Unknown model '{self.model}'.")

    def ground_energy(self, params: Sequence[float] | np.ndarray) -> float:
        """Exact ground-state energy (smallest eigenvalue)."""
        return float(np.linalg.eigvalsh(self.ham(params))[0])

    def spectrum(self, params: Sequence[float] | np.ndarray, k: int = 4) -> np.ndarray:
        """Lowest ``k`` eigenvalues."""
        evals = np.linalg.eigvalsh(self.ham(params))
        return evals[:k]

    def n_params(self) -> int:
        return len(self.param_names)


# ────────────────────── result dataclasses ───────────────────────────────

@dataclass
class InverseDesignResult:
    """Result of an inverse-design run.

    Attributes
    ----------
    params_opt   : recovered parameter vector (best restart).
    E0_achieved  : ground-state energy at ``params_opt``.
    E0_target    : the target energy passed in.
    residual     : ``|E0_achieved - E0_target|``.
    n_restarts   : number of independent L-BFGS-B restarts performed.
    converged    : True if at least one restart reported convergence.
    param_names  : name of each entry in ``params_opt``.
    notes        : honest-scope annotation.
    """
    params_opt:  np.ndarray
    E0_achieved: float
    E0_target:   float
    residual:    float
    n_restarts:  int
    converged:   bool
    param_names: list[str]
    notes:       str = ""


@dataclass
class LearningResult:
    """Result of a Hamiltonian-learning run.

    Attributes
    ----------
    params_opt   : recovered parameter vector (best restart).
    loss_final   : final loss value (sum of squared energy residuals).
    target_energies : the target energy levels passed in.
    achieved_energies : energy levels at ``params_opt``.
    n_restarts   : number of independent L-BFGS-B restarts performed.
    converged    : True if at least one restart reported convergence.
    param_names  : name of each entry in ``params_opt``.
    notes        : honest-scope annotation.
    """
    params_opt:        np.ndarray
    loss_final:        float
    target_energies:   np.ndarray
    achieved_energies: np.ndarray
    n_restarts:        int
    converged:         bool
    param_names:       list[str]
    notes:             str = ""


# ────────────────────── inverse design ───────────────────────────────────

def inverse_design(
    target_e0: float,
    model: str = "ising",
    n_sites: int = 4,
    param_bounds: list[tuple[float, float]] | None = None,
    x0: np.ndarray | None = None,
    n_restarts: int = 5,
    seed: int = 0,
    tol: float = 1e-10,
) -> InverseDesignResult:
    """Find Hamiltonian parameters whose ground-state energy equals ``target_e0``.

    Minimises ``(E_0(params) - target_e0)^2`` with L-BFGS-B.

    Parameters
    ----------
    target_e0    : desired ground-state energy.
    model        : ``"ising"`` or ``"xx"``.
    n_sites      : lattice size.
    param_bounds : box constraints for each parameter, e.g. ``[(0.1, 5.0), (0.0, 3.0)]``.
                   Defaults to ``[(0.01, 10.0)]`` per parameter.
    x0           : initial parameter vector (overrides random restarts if given).
    n_restarts   : number of independent random restarts.
    seed         : RNG seed for restarts.
    tol          : convergence tolerance.

    Returns
    -------
    :class:`InverseDesignResult`

    Honest scope
    ------------
    Local minima are possible; increase ``n_restarts`` to mitigate.
    Uniqueness of the solution is model-dependent and not guaranteed `[研究]`.
    """
    phys = ParametricHam(model=model, n_sites=n_sites)
    n_p  = phys.n_params()
    if param_bounds is None:
        param_bounds = [(0.01, 10.0)] * n_p

    def loss(p: np.ndarray) -> float:
        return (phys.ground_energy(p) - target_e0) ** 2

    rng = np.random.default_rng(seed)
    best: OptimizeResult | None = None

    starts: list[np.ndarray] = []
    if x0 is not None:
        starts.append(np.asarray(x0, dtype=float))
    for _ in range(n_restarts):
        p = np.array([rng.uniform(lo, hi) for lo, hi in param_bounds])
        starts.append(p)

    for start in starts:
        res = minimize(loss, start, method="L-BFGS-B", bounds=param_bounds,
                       options={"ftol": tol, "gtol": tol, "maxiter": 500})
        if best is None or res.fun < best.fun:
            best = res

    p_opt     = np.asarray(best.x)
    e0_achiev = phys.ground_energy(p_opt)
    return InverseDesignResult(
        params_opt=p_opt,
        E0_achieved=e0_achiev,
        E0_target=target_e0,
        residual=abs(e0_achiev - target_e0),
        n_restarts=len(starts),
        converged=bool(best.success),
        param_names=phys.param_names,
        notes=(
            f"L-BFGS-B inverse design; model={model}; n_sites={n_sites}; "
            "local minima possible — use n_restarts>1; "
            "uniqueness not guaranteed [研究]; continuum limit [OUT]"
        ),
    )


# ────────────────────── Hamiltonian learning ─────────────────────────────

def hamiltonian_learning(
    target_energies: Sequence[float],
    model: str = "ising",
    n_sites: int = 4,
    param_bounds: list[tuple[float, float]] | None = None,
    x0: np.ndarray | None = None,
    n_restarts: int = 5,
    seed: int = 0,
    tol: float = 1e-10,
) -> LearningResult:
    """Recover Hamiltonian parameters from a set of observed energy levels.

    Minimises the sum of squared residuals
    ``Σ_k (E_k(params) - target_k)^2``
    where the sum runs over the ``len(target_energies)`` lowest eigenvalues.

    Parameters
    ----------
    target_energies : observed energy levels to fit (ascending order expected).
    model           : ``"ising"`` or ``"xx"``.
    n_sites         : lattice size.
    param_bounds    : box constraints.  Defaults to ``[(0.01, 10.0)]`` per param.
    x0              : initial parameter vector (overrides restarts if given).
    n_restarts      : number of independent random restarts.
    seed            : RNG seed.
    tol             : convergence tolerance.

    Returns
    -------
    :class:`LearningResult`

    Honest scope
    ------------
    Recovery is only guaranteed when ``target_energies`` is consistent with
    the model family and the observable set is sufficient `[研究]`.
    """
    tgt  = np.asarray(target_energies, dtype=float)
    k    = len(tgt)
    phys = ParametricHam(model=model, n_sites=n_sites)
    n_p  = phys.n_params()
    if param_bounds is None:
        param_bounds = [(0.01, 10.0)] * n_p

    def loss(p: np.ndarray) -> float:
        achieved = phys.spectrum(p, k=k)
        return float(np.sum((achieved - tgt) ** 2))

    rng    = np.random.default_rng(seed)
    best: OptimizeResult | None = None
    starts: list[np.ndarray] = []
    if x0 is not None:
        starts.append(np.asarray(x0, dtype=float))
    for _ in range(n_restarts):
        p = np.array([rng.uniform(lo, hi) for lo, hi in param_bounds])
        starts.append(p)

    for start in starts:
        res = minimize(loss, start, method="L-BFGS-B", bounds=param_bounds,
                       options={"ftol": tol, "gtol": tol, "maxiter": 500})
        if best is None or res.fun < best.fun:
            best = res

    p_opt    = np.asarray(best.x)
    achieved = phys.spectrum(p_opt, k=k)
    return LearningResult(
        params_opt=p_opt,
        loss_final=float(best.fun),
        target_energies=tgt,
        achieved_energies=achieved,
        n_restarts=len(starts),
        converged=bool(best.success),
        param_names=phys.param_names,
        notes=(
            f"L-BFGS-B Hamiltonian learning; model={model}; n_sites={n_sites}; "
            f"fitting {k} energy levels; "
            "identifiability depends on observable set [研究]; continuum [OUT]"
        ),
    )


# ────────────────────── energy gradient ──────────────────────────────────

def _ham_component_matrices(phys: ParametricHam) -> list[np.ndarray] | None:
    """Return the fixed component matrices for a linear parametric Hamiltonian.

    H(params) = Σ_i params[i] * components[i]

    Returns None when the model is unknown (caller falls back to FD).
    """
    n = phys.n_sites
    I2 = np.eye(2, dtype=float)
    Z  = np.array([[1.0, 0.0], [0.0, -1.0]])
    X  = np.array([[0.0, 1.0], [1.0, 0.0]])

    dim = 2 ** n

    def _kron_op(ops: list[np.ndarray]) -> np.ndarray:
        r = ops[0]
        for op in ops[1:]:
            r = np.kron(r, op)
        return r

    if phys.model == "ising":
        H_ZZ = np.zeros((dim, dim))
        for i in range(n - 1):
            ops = [Z if j == i or j == i + 1 else I2 for j in range(n)]
            H_ZZ -= _kron_op(ops)
        H_X = np.zeros((dim, dim))
        for i in range(n):
            ops = [X if j == i else I2 for j in range(n)]
            H_X -= _kron_op(ops)
        return [H_ZZ, H_X]   # H = J*H_ZZ + h*H_X

    if phys.model == "xx":
        Y_real = np.array([[0.0, -1.0], [1.0, 0.0]])  # same as xx_model_ham
        H_XX = np.zeros((dim, dim))
        for i in range(n - 1):
            ops_x = [X if j == i or j == i + 1 else I2 for j in range(n)]
            ops_y = [Y_real if j == i or j == i + 1 else I2 for j in range(n)]
            H_XX -= 0.5 * (_kron_op(ops_x) + _kron_op(ops_y))
        return [H_XX]   # H = J*H_XX

    return None


def energy_gradient(
    params: Sequence[float],
    phys: ParametricHam,
    eps: float = 1e-5,
) -> np.ndarray:
    """Gradient of the ground-state energy w.r.t. ``params``.

    When JAX is installed and the model is ``"ising"`` or ``"xx"``,
    uses exact autodiff (``jax.grad`` through ``jnp.linalg.eigvalsh``).
    Otherwise falls back to centred finite differences with step ``eps``.

    Parameters
    ----------
    params : current parameter vector.
    phys   : :class:`ParametricHam` instance.
    eps    : finite-difference step size (ignored when JAX path is used).

    Returns
    -------
    Gradient vector of the same length as ``params``.
    """
    p = np.asarray(params, dtype=float)

    # JAX exact-autodiff path
    components = _ham_component_matrices(phys)
    if components is not None:
        try:
            import jax
            import jax.numpy as jnp

            jax.config.update("jax_enable_x64", True)
            comps_jax = [jnp.asarray(c.real, dtype=jnp.float64) for c in components]

            def _energy(p_jax: jax.Array) -> jax.Array:
                H = sum(p_jax[i] * comps_jax[i] for i in range(len(comps_jax)))
                return jnp.linalg.eigvalsh(H)[0]

            grad_fn = jax.grad(_energy)
            return np.asarray(grad_fn(jnp.asarray(p, dtype=jnp.float64)), dtype=float)
        except ImportError:
            pass   # JAX not installed; fall through to FD

    # Centred finite-difference fallback
    grad = np.zeros_like(p)
    for i in range(len(p)):
        pp = p.copy(); pp[i] += eps
        pm = p.copy(); pm[i] -= eps
        grad[i] = (phys.ground_energy(pp) - phys.ground_energy(pm)) / (2 * eps)
    return grad
