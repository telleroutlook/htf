"""HTF adapter for TeNPy MatrixProductState.

Extracts the full state vector from a TeNPy finite MPS and produces a
:class:`~htf.rayleigh_cert.RayleighCertificate` for a given Hamiltonian H.

HTF does not call the TeNPy solver — it only reads the already-computed MPS
and certifies the Rayleigh quotient.

Supported MPS types
-------------------
Only ``bc == "finite"`` single-physical-leg MPS are supported.  Infinite
(iMPS) and segment MPS are explicitly rejected to prevent silently certifying
a per-unit-cell state vector as a full-system state.

Extraction semantics
--------------------
When TeNPy is installed the adapter delegates to
``tenpy.algorithms.exact_diag.get_full_wavefunction(mps, undo_sort_charge=True)``
— the only path that guarantees correct finite-MPS extraction (bc check,
virtual-leg squeeze, charge-sort undo).

When TeNPy is *not* installed (duck-type / test mode) the adapter uses:

    ``mps_like.bc == "finite"`` (required attribute)
    ``mps_like.L``              — number of sites (int)
    ``mps_like.get_theta(0, L)`` — full-chain theta; may be a TeNPy ``Array``
                                    (has ``.to_ndarray()``) or plain numpy array.

The duck-type contract requires the caller to supply an object whose theta is
already in C-order with axes ``(vL, p0, …, p_{L-1}, vR)`` and boundary legs
of size 1 — these semantics cannot be verified from the array alone.

Dependency
----------
tenpy is an *optional* dependency.

Usage::

    import tenpy
    from htf.adapters.tenpy_adapter import rayleigh_from_tenpy_mps

    mps  = tenpy.MPS.from_lat_product_state(...)  # finite TeNPy MPS
    H    = build_hamiltonian(...)                  # numpy 2-D array
    cert = rayleigh_from_tenpy_mps(mps, H)
    print(cert.to_json(indent=2))
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from htf.rayleigh_cert import RayleighCertificate


def _preserve_state_dtype(raw: np.ndarray) -> np.ndarray:
    """Ravel, validate, and canonicalise a raw state array.

    Real input → float64.  Complex input → complex128.  The complex amplitude
    is preserved in full; no imaginary-part projection is performed.

    Raises
    ------
    ValueError
        If the resulting vector is empty or contains non-finite values.
    """
    raw = np.asarray(raw).reshape(-1, order="C")
    if raw.size == 0:
        raise ValueError("empty state vector")
    dtype = np.complex128 if np.iscomplexobj(raw) else np.float64
    raw = raw.astype(dtype, copy=False)
    if not np.all(np.isfinite(raw)):
        raise ValueError("state vector contains non-finite values")
    return raw


def _extract_tenpy_state_vector(mps_like) -> np.ndarray:
    """Pull a 1-D state vector from a TeNPy finite MPS.

    Extraction priority:

    1. **Real TeNPy** (when importable): delegates to
       ``get_full_wavefunction(mps_like, undo_sort_charge=True)`` which checks
       ``bc == "finite"``, squeezes boundary virtual legs, and undoes any
       charge-sort permutation applied by ``Site.sort_charge()``.
    2. **Duck-type** (TeNPy not installed): requires ``mps_like.bc ==
       "finite"`` and uses ``mps_like.get_theta(0, mps_like.L)``.  The caller
       is responsible for ensuring the theta axes are in the order
       ``(vL, p0, …, p_{L-1}, vR)`` with boundary legs of size 1, and that
       the local basis matches ``H``.

    Raises
    ------
    TypeError
        If neither the TeNPy interface nor ``to_dense`` is available.
    ValueError
        If ``bc != "finite"``; or the extracted vector is empty / non-finite.
    """
    # ── path 1: real TeNPy ────────────────────────────────────────────────────
    try:
        from tenpy.algorithms.exact_diag import get_full_wavefunction as _gwf  # type: ignore[import]
        bc = getattr(mps_like, "bc", None)
        if bc != "finite":
            raise ValueError(
                f"TeNPy adapter accepts only bc='finite'; got bc={bc!r}. "
                "Infinite (iMPS) and segment MPS are not supported."
            )
        raw = np.asarray(_gwf(mps_like, undo_sort_charge=True))
        return _preserve_state_dtype(raw)
    except ImportError:
        pass  # TeNPy not installed; fall through to duck-type

    # ── path 2: duck-type ─────────────────────────────────────────────────────
    bc = getattr(mps_like, "bc", None)
    if bc is not None and bc != "finite":
        raise ValueError(
            f"TeNPy adapter accepts only bc='finite'; got bc={bc!r}. "
            "Infinite (iMPS) and segment MPS are not supported."
        )

    if hasattr(mps_like, "get_theta") and hasattr(mps_like, "L"):
        n_sites = int(mps_like.L)
        theta = mps_like.get_theta(0, n_sites)
        if hasattr(theta, "to_ndarray"):
            raw = theta.to_ndarray()
        else:
            raw = np.asarray(theta)
        return _preserve_state_dtype(raw)

    if hasattr(mps_like, "to_dense"):
        raw = np.asarray(mps_like.to_dense())
        return _preserve_state_dtype(raw)

    raise TypeError(
        f"{type(mps_like).__name__!r} has neither 'get_theta'+'L' (TeNPy "
        "interface) nor 'to_dense' (fallback interface). "
        "Pass a tenpy MatrixProductState or any object implementing "
        "get_theta(i, n) -> array and .L -> int."
    )


def rayleigh_from_tenpy_mps(
    mps_like,
    H: np.ndarray,
    *,
    notes: str = "",
) -> RayleighCertificate:
    """Compute a Rayleigh Certificate from a TeNPy finite MPS and a Hamiltonian.

    HTF acts purely as a verifier: it reads the state vector from *mps_like*
    and certifies that ``E0 ≤ Re(⟨ψ|H|ψ⟩/⟨ψ|ψ⟩)`` via the Rayleigh-Ritz
    theorem.  No optimisation is performed.

    **Supported MPS:** ``bc == "finite"``, standard single-physical-leg MPS
    only.  Infinite (iMPS) and segment MPS raise ``ValueError``.

    **Basis contract (caller's responsibility):**
    ``H`` MUST be expressed in exactly the tensor-product basis used by the
    extracted state vector.  When TeNPy is installed the adapter uses
    ``get_full_wavefunction(..., undo_sort_charge=True)``, so the local basis
    is the pre-charge-sort basis of each ``Site``.  In duck-type mode the
    adapter uses sites ``0..L-1`` with C-order flattening of the theta axes
    ``(vL, p0, …, p_{L-1}, vR)``; the caller must ensure this matches ``H``.

    **Charge-sorting warning (duck-type mode only):** when TeNPy is *not*
    installed and you pass a raw MPS object whose ``get_theta(0, L)`` was
    produced by a Site with non-trivial charge sorting (``Site.sort_charge``),
    the extracted state vector may be in a permuted basis that does NOT match
    a conventionally built ``H``.  Always use the real TeNPy path (which calls
    ``get_full_wavefunction(..., undo_sort_charge=True)``) for U(1)- or
    SU(2)-symmetric MPS.  Duck-type mode is intended for testing with plain
    numpy arrays only.

    Matching total dimension ``H.shape[0] == len(psi)`` is a necessary but not
    sufficient condition; site permutations and local-basis permutations
    (``Site.perm``) are invisible to the adapter.

    Parameters
    ----------
    mps_like : tenpy ``MatrixProductState`` (bc='finite') or duck-type equivalent
        The MPS whose state vector is extracted.
    H : np.ndarray, shape (n, n)
        Hermitian Hamiltonian expressed in the same tensor-product basis as the
        extracted state vector.
    notes : str
        Free-text notes stored in the certificate.

    Returns
    -------
    cert : :class:`~htf.rayleigh_cert.RayleighCertificate`
        With ``verified=False``.  Call
        :func:`~htf.rayleigh_cert.verify_rayleigh_certificate` to confirm.

    Raises
    ------
    TypeError
        ``mps_like`` has neither ``get_theta``+``L`` nor ``to_dense``.
    ValueError
        ``bc != "finite"``; state vector is empty or non-finite; H/ψ dimension
        mismatch; or H is not symmetric/Hermitian.

    Notes
    -----
    Full Hilbert-space certification scales as O(d^L) in memory.  For d=2
    (qubits) L ≤ ~20 is practical; for larger L use TeNPy's native energy
    expectation instead (no certification).

    Examples
    --------
    Without TeNPy (mock interface for testing)::

        class MockTeNPyMPS:
            bc = "finite"
            L = 2
            def get_theta(self, i, n):
                # shape (vL, p0, p1, vR) = (1, 2, 2, 1) for d=2, L=2
                arr = np.zeros((1, 2, 2, 1)); arr[0, 0, 0, 0] = 1.0
                return arr

        H = np.diag([0.0, 1.0, 1.0, 2.0])
        cert = rayleigh_from_tenpy_mps(MockTeNPyMPS(), H)
        assert cert.upper >= 0.0

    With TeNPy::

        import tenpy
        mps = tenpy.MPS.from_lat_product_state(...)
        H_mat = build_full_hamiltonian(n=4)
        cert = rayleigh_from_tenpy_mps(mps, H_mat, notes="tenpy finite MPS L=4")
    """
    from ..rayleigh_cert import rayleigh_certificate

    psi = _extract_tenpy_state_vector(mps_like)
    H_arr = np.asarray(H)

    bc_val = getattr(mps_like, "bc", "duck-type")
    provenance = (
        f"tenpy-adapter: backend={type(mps_like).__name__!r}; "
        f"bc={bc_val!r}; "
        f"basis=caller-asserted tensor-product; "
        f"undo_sort_charge=True (when real TeNPy); H_source=caller"
    )
    combined_notes = f"{provenance}; {notes}" if notes else provenance

    return rayleigh_certificate(H_arr, psi, notes=combined_notes)
