"""HTF adapter for quimb MatrixProductState.

Extracts the full state vector from a quimb MPS (or any object that exposes
a ``to_dense()`` method returning a numpy array) and produces a
:class:`~htf.rayleigh_cert.RayleighCertificate` for a given Hamiltonian H.

HTF does not call the quimb solver — it only reads the already-computed MPS
and certifies the Rayleigh quotient.

Dependency
----------
quimb is an *optional* dependency.  The adapter works with any object whose
``to_dense()`` returns a numpy array (including mock objects in tests).

Usage::

    import quimb.tensor as qtn
    from htf.adapters.quimb_adapter import rayleigh_from_quimb_mps

    mps = qtn.MPS_rand_state(n=4, bond_dim=4)   # or your optimised MPS
    H   = build_hamiltonian(...)                  # numpy 2-D array
    cert = rayleigh_from_quimb_mps(mps, H)
    print(cert.to_json(indent=2))
"""
from __future__ import annotations

import numpy as np


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


def _extract_state_vector(mps_like) -> np.ndarray:
    """Pull a 1-D state vector from a quimb MPS-like object.

    Calls ``mps_like.to_dense()`` (quimb API) and ravels to 1-D.  Real arrays
    become float64; complex arrays become complex128 — no imaginary-part
    truncation is applied.

    Raises
    ------
    TypeError
        If the object has no ``to_dense`` method.
    ValueError
        If the resulting vector is empty or non-finite.
    """
    if not hasattr(mps_like, "to_dense"):
        raise TypeError(
            f"{type(mps_like).__name__!r} has no 'to_dense' method. "
            "Pass a quimb MatrixProductState or any object implementing "
            "to_dense() -> np.ndarray."
        )
    raw = np.asarray(mps_like.to_dense())
    return _preserve_state_dtype(raw)


def rayleigh_from_quimb_mps(
    mps_like,
    H: np.ndarray,
    *,
    notes: str = "",
) -> "htf.rayleigh_cert.RayleighCertificate":  # type: ignore[name-defined]
    """Compute a Rayleigh Certificate from a quimb MPS and a Hamiltonian.

    HTF acts purely as a verifier: it reads the state vector from *mps_like*
    and certifies that ``E0 ≤ Re(⟨ψ|H|ψ⟩/⟨ψ|ψ⟩)`` via the Rayleigh-Ritz
    theorem.  No optimisation is performed.

    **Basis contract (caller's responsibility):**
    ``H`` MUST be expressed in exactly the tensor-product basis used by the
    extracted state vector.  For quimb ``MatrixProductState`` this is the
    order of the present ``mps.sites`` and each physical-index order used by
    ``to_dense()``.  The current quimb default returns a ket of shape
    ``(D, 1)``; the adapter ravels this to ``(D,)``.  Any lattice/site/local-
    basis permutation must be applied consistently to both ``H`` and the MPS
    before calling this function.  Matching total dimension
    ``H.shape[0] == len(psi)`` is a necessary but not sufficient condition for
    semantic consistency; the adapter cannot verify basis agreement from a
    plain ndarray.

    Parameters
    ----------
    mps_like : quimb ``MatrixProductState`` (or any object with ``to_dense()``)
        The MPS whose state vector is extracted.  Must implement
        ``to_dense() -> array_like`` (quimb convention).
    H : np.ndarray, shape (n, n)
        Hermitian Hamiltonian expressed in the same tensor-product basis as the
        extracted state vector (dimension must match the flattened state vector).
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
        ``mps_like`` has no ``to_dense`` method.
    ValueError
        State vector is empty or non-finite; or H/ψ dimension mismatch;
        or H is not symmetric/Hermitian.

    Examples
    --------
    Without quimb (mock interface for testing)::

        class MockMPS:
            def to_dense(self): return np.array([1.0, 0.0])

        H = np.diag([0.0, 1.0])
        cert = rayleigh_from_quimb_mps(MockMPS(), H)
        assert cert.upper >= 0.0

    With quimb::

        import quimb.tensor as qtn
        mps = qtn.MPS_rand_state(n=4, bond_dim=8, seed=0)
        H   = build_full_hamiltonian(n=4)
        cert = rayleigh_from_quimb_mps(mps, H, notes="quimb MPS n=4 chi=8")
    """
    from ..rayleigh_cert import rayleigh_certificate

    psi = _extract_state_vector(mps_like)
    H_arr = np.asarray(H)

    provenance = (
        f"quimb-adapter: backend={type(mps_like).__name__!r}; "
        f"basis=caller-asserted tensor-product; H_source=caller"
    )
    combined_notes = f"{provenance}; {notes}" if notes else provenance

    return rayleigh_certificate(H_arr, psi, notes=combined_notes)
