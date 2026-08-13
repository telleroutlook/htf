"""HTF Layer 3 — Tensor Engine.

Executes a diagram by recursive tensor contraction.  Two modes:

* ``"float"``  — fast, discovery-tier (numpy float64); **carries no error
  bound**.
* ``"certified"`` — rigorous interval arithmetic via python-flint's Arb
  library.  Every arithmetic operation is tracked with outward-rounded ball
  arithmetic; the returned :class:`~htf.certificate.Certificate` carries a
  rigorous ``error_bound`` (maximum ball-radius over all result entries),
  bounding the floating-point rounding error accumulated during contraction.

  Requires ``pip install python-flint``.

What "certified" certifies
--------------------------
The error bound covers **floating-point rounding only** (Phase 2 scope).
It does *not* bound bond-dimension truncation error (Phase 3/4 scope), nor
modeling error — those are separate, explicitly out-of-scope walls described
in ``PLAN.md``.
"""
from __future__ import annotations

import numpy as np

from .functor import TensorFunctor
from .topology import Box, Id, Tensor, Then, dims

# opt_einsum provides optimised contraction-path selection.
# If available it is used automatically in float mode; otherwise numpy fallback.
try:
    import opt_einsum as _opt_einsum
    _HAS_OPT_EINSUM = True
except ImportError:
    _opt_einsum = None  # type: ignore[assignment]
    _HAS_OPT_EINSUM = False


# ─────────────────────────────── float mode ────────────────────────────────

def _eval(d, F: TensorFunctor) -> np.ndarray:
    """Return the dense tensor of diagram *d* with layout dims(cod)+dims(dom)."""
    if isinstance(d, Box):
        return F.tensor(d)

    if isinstance(d, Id):
        ds = dims(d.ty)
        if not ds:
            return np.array(1.0)
        n = int(np.prod(ds))
        return np.eye(n, dtype=float).reshape(ds + ds)

    if isinstance(d, Then):
        Ff = _eval(d.f, F)  # shape: dims(f.cod) + dims(f.dom)
        Gg = _eval(d.g, F)  # shape: dims(g.cod) + dims(g.dom)
        nb = len(d.f.cod)   # shared wires (f.cod == g.dom)
        nc = len(d.g.cod)
        nd = len(d.f.dom)
        if _HAS_OPT_EINSUM and nb > 0:
            # opt_einsum selects an optimised contraction path.
            # Gg axes: [0..nc-1, nc..nc+nb-1]; Ff axes: [nc..nc+nb-1, nc+nb..nc+nb+nd-1]
            idx_gg  = list(range(nc + nb))
            idx_ff  = list(range(nc, nc + nb + nd))
            idx_out = list(range(nc)) + list(range(nc + nb, nc + nb + nd))
            return _opt_einsum.contract(Gg, idx_gg, Ff, idx_ff, idx_out)
        g_in = list(range(nc, nc + nb))   # g's input axes
        f_out = list(range(nb))        # f's output axes
        return np.tensordot(Gg, Ff, axes=(g_in, f_out))

    if isinstance(d, Tensor):
        Ff = _eval(d.f, F)
        Gg = _eval(d.g, F)
        nfo, nfi = len(d.f.cod), len(d.f.dom)
        ngo, ngi = len(d.g.cod), len(d.g.dom)
        outer = np.tensordot(Ff, Gg, axes=0)  # fcod+fdom+gcod+gdom
        perm = (
            list(range(nfo))
            + list(range(nfo + nfi, nfo + nfi + ngo))
            + list(range(nfo, nfo + nfi))
            + list(range(nfo + nfi + ngo, nfo + nfi + ngo + ngi))
        )
        return np.transpose(outer, perm) if perm else outer

    raise TypeError(f"unknown diagram node: {type(d)!r}")


# ────────────────────────────── certified mode ─────────────────────────────

def _prod(t: tuple) -> int:
    return int(np.prod(t)) if t else 1


def _numpy_to_arb_mat(arr: np.ndarray, nrows: int, ncols: int):
    """Reshape *arr* to (nrows, ncols) and convert to a flint ``arb_mat``."""
    from flint import arb, arb_mat
    flat = arr.reshape(nrows, ncols)
    return arb_mat(
        [[arb(float(flat[i, j])) for j in range(ncols)] for i in range(nrows)]
    )


def _eval_certified(d, F: TensorFunctor):
    """Evaluate *d* using flint Arb, returning an ``arb_mat`` of shape
    ``(_prod(dims(cod)), _prod(dims(dom)))``.

    Sequential composition (Then) maps to ``arb_mat`` matrix multiplication;
    parallel composition (Tensor) maps to the Kronecker-product layout that
    matches the axis permutation in ``_eval``.
    """
    from flint import arb, arb_mat

    cs = _prod(dims(d.cod))   # product of all cod dimensions
    ds_ = _prod(dims(d.dom))  # product of all dom dimensions

    if isinstance(d, Box):
        arr = F.tensor(d)  # shape dims(cod) + dims(dom)
        return _numpy_to_arb_mat(arr, cs, ds_)

    if isinstance(d, Id):
        n = cs  # cs == ds_ for identity
        return arb_mat(
            [[arb(1.0) if i == j else arb(0.0) for j in range(n)] for i in range(n)]
        )

    if isinstance(d, Then):
        # f: dom_f → cod_f,  g: dom_g=cod_f → cod_g
        # arb_mat(g) · arb_mat(f) : (cod_g_size × cod_f_size) · (cod_f_size × dom_f_size)
        Ff = _eval_certified(d.f, F)
        Gg = _eval_certified(d.g, F)
        return Gg * Ff  # arb_mat multiplication tracks rounding rigorously

    if isinstance(d, Tensor):
        # Parallel composition: result has Kronecker-product layout.
        # result[i·g_cs + k, j·g_ds + l] = Ff[i,j] · Gg[k,l]
        # This matches the axis permutation in _eval (fcod, gcod, fdom, gdom).
        f_cs = _prod(dims(d.f.cod))
        f_ds = _prod(dims(d.f.dom))
        g_cs = _prod(dims(d.g.cod))
        g_ds = _prod(dims(d.g.dom))
        Ff = _eval_certified(d.f, F)
        Gg = _eval_certified(d.g, F)
        rows = []
        for i in range(f_cs):
            for k in range(g_cs):
                row = [Ff[i, j] * Gg[k, l] for j in range(f_ds) for l in range(g_ds)]
                rows.append(row)
        return arb_mat(rows)

    raise TypeError(f"unknown diagram node: {type(d)!r}")


def _extract_arb_mat(
    mat, result_shape: tuple
) -> tuple[np.ndarray | float, float]:
    """Extract (midpoint_array, max_radius) from a flint ``arb_mat``.

    *result_shape* is the target numpy shape (dims(cod) + dims(dom)).
    Returns a scalar float when *result_shape* is ``()``.
    """
    m, n = mat.nrows(), mat.ncols()
    mid = np.zeros((m, n))
    max_rad = 0.0
    for i in range(m):
        for j in range(n):
            e = mat[i, j]
            mid[i, j] = float(e.mid())
            r = float(e.rad())
            max_rad = max(max_rad, r)
    if result_shape:
        return mid.reshape(result_shape), max_rad
    return float(mid[0, 0]), max_rad


# ──────────────────────────────── public API ───────────────────────────────

def contract(diagram, functor: TensorFunctor, mode: str = "float"):
    """Contract *diagram* under *functor*.

    Parameters
    ----------
    mode : ``"float"`` | ``"certified"``
        ``"float"``
            numpy float64 dense result (discovery-tier, no error bound).
        ``"certified"``
            Rigorous Arb interval arithmetic.  Returns a
            :class:`~htf.certificate.Certificate` with ``error_bound`` set
            to the maximum ball-radius over all result entries.  Requires
            ``python-flint`` (``pip install python-flint``).

    Returns
    -------
    numpy.ndarray
        Float-mode result array.
    Certificate
        Certified-mode result with ``result`` (midpoint array) and
        ``error_bound`` (rigorous floating-point rounding bound).
    """
    if mode == "float":
        return _eval(diagram, functor)

    if mode == "certified":
        try:
            import flint  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "certified mode requires python-flint "
                "(pip install python-flint)"
            ) from exc

        from .certificate import Certificate

        arb_result = _eval_certified(diagram, functor)
        result_shape = dims(diagram.cod) + dims(diagram.dom)
        result_arr, error_bound = _extract_arb_mat(arb_result, result_shape)

        return Certificate(
            result=result_arr,
            mode="certified",
            error_bound=error_bound,
            backend="flint-arb",
        )

    raise ValueError(f"unknown mode {mode!r} (expected 'float' or 'certified')")
