"""Factorized Rayleigh certificate using MPS/MPO tensors (schema: rayleigh-cert-mps/v1).

[工程]  assurance="reproducible": stores MPS/MPO tensors, verifies via float64
         MPS contractions.  Memory O(n·χ²·d + n·W²·d²) vs O(d^{2n}) for dense.

[工程]  assurance="rigorous": Arb/Acb transfer-matrix contractions over the full
         MPS/MPO chain.  Requires python-flint.  Wall-clock cost scales as
         O(n · χ²·d · χ_r + n · χ²·W²·d²) per site — practical for χ ≤ 32.

Limitation
----------
Bounds E0 ≤ rayleigh_upper only for the *stored* MPS/MPO.
Bond-dimension truncation error (χ bias) and modeling error are [OUT].
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass, field

import numpy as np

from ._rayleigh_primitives import (
    EXPECTED_THEOREM,
    _decode_canonical,
    _encode_canonical,
)
from .mpo import MPO, mpo_expectation
from .mps import MPS, mps_inner

MPS_CERT_SCHEMA = "rayleigh-cert-mps/v1"


# ──────────────────────────────────────────────────────────────────────────────
# Tensor encode / decode (reuse canonical codec for arbitrary-rank arrays)
# ──────────────────────────────────────────────────────────────────────────────

def _encode_tensor(arr: np.ndarray) -> object:
    return _encode_canonical(arr)


def _decode_tensor(data: object) -> np.ndarray:
    return _decode_canonical(data)


# ──────────────────────────────────────────────────────────────────────────────
# Digest
# ──────────────────────────────────────────────────────────────────────────────

def _canonical_digest_mps(mps: MPS, mpo: MPO) -> str:
    """Domain-separated SHA-256 over all MPS and MPO tensor bytes (v1).

    Format::

        b"rayleigh-cert-mps-input/v1\\x00" + type_flag (C|R)
        uint32 n_sites_mps
        uint32 n_sites_mpo
        For each MPS tensor A[i]:
            field(f"mps[{i}].shape", big-endian ndim + dims as uint64)
            field(f"mps[{i}].real",  C-order big-endian float64 bytes)
            [if complex: field(f"mps[{i}].imag", ...)]
        For each MPO tensor W[i]: same pattern with tag "mpo[{i}]"
    """
    def _field(tag: bytes, payload: bytes) -> bytes:
        return struct.pack(">H", len(tag)) + tag + struct.pack(">Q", len(payload)) + payload

    def _shape_bytes(arr: np.ndarray) -> bytes:
        return struct.pack(">I", arr.ndim) + b"".join(
            struct.pack(">Q", int(s)) for s in arr.shape
        )

    is_complex = any(np.iscomplexobj(t) for t in mps.tensors) or \
                 any(np.iscomplexobj(t) for t in mpo.tensors)

    raw = b"rayleigh-cert-mps-input/v1\x00" + (b"C" if is_complex else b"R")
    raw += struct.pack(">I", mps.n_sites)
    raw += struct.pack(">I", mpo.n_sites)

    for tensors, prefix in ((mps.tensors, "mps"), (mpo.tensors, "mpo")):
        for i, arr in enumerate(tensors):
            tag = f"{prefix}[{i}]".encode()
            raw += _field(tag + b".shape", _shape_bytes(arr))
            if is_complex:
                raw += _field(tag + b".real",
                              np.asarray(arr.real, dtype=">f8").tobytes(order="C"))
                raw += _field(tag + b".imag",
                              np.asarray(arr.imag, dtype=">f8").tobytes(order="C"))
            else:
                raw += _field(tag + b".real",
                              np.asarray(arr, dtype=">f8").tobytes(order="C"))

    return hashlib.sha256(raw).hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# Dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RayleighCertificateMPS:
    """Factorized Rayleigh certificate: ``E0 ≤ rayleigh_upper``.

    Stores MPS/MPO tensors instead of the dense Hamiltonian and state vector.
    For n sites, physical dimension d, MPS bond χ, MPO bond W the storage is
    O(n·χ²·d + n·W²·d²) compared with O(d^{2n} + d^n) for dense certificates.

    Fields
    ------
    schema_version  Always ``"rayleigh-cert-mps/v1"``.
    claim           Human-readable statement of what is certified.
    theorem         Rayleigh-Ritz theorem string.
    assurance       ``"reproducible"`` — float64 contraction, no Arb interval.
    backend         ``"float64-mps"`` for this implementation.
    n_sites         Number of lattice sites.
    phys_dim        Physical dimension per site.
    mps_max_bond    Maximum MPS bond dimension.
    mpo_bond_dim    Maximum MPO bond dimension.
    rayleigh_upper  Stored upper bound (one ULP above computed quotient).
    rayleigh_lower  Stored lower bound (one ULP below computed quotient).
    input_digest    SHA-256 of all MPS/MPO tensor bytes.
    htf_version     Version of HTF that produced this certificate.
    notes           Optional annotation.
    verified        Set to ``True`` by :func:`verify_rayleigh_certificate_mps`.
    """
    schema_version: str
    claim:          str
    theorem:        str
    assurance:      str
    backend:        str
    n_sites:        int
    phys_dim:       int
    mps_max_bond:   int
    mpo_bond_dim:   int
    rayleigh_upper: float
    rayleigh_lower: float
    input_digest:   str
    htf_version:    str
    notes:          str  = ""
    verified:       bool = False

    # Populated by rayleigh_certificate_mps; required by verify_rayleigh_certificate_mps.
    _mps_tensors: list = field(default_factory=list, repr=False)
    _mpo_tensors: list = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return {
            "schema_version":  self.schema_version,
            "claim":           self.claim,
            "theorem":         self.theorem,
            "assurance":       self.assurance,
            "backend":         self.backend,
            "n_sites":         self.n_sites,
            "phys_dim":        self.phys_dim,
            "mps_max_bond":    self.mps_max_bond,
            "mpo_bond_dim":    self.mpo_bond_dim,
            "rayleigh_upper":  self.rayleigh_upper,
            "rayleigh_lower":  self.rayleigh_lower,
            "input_digest":    self.input_digest,
            "htf_version":     self.htf_version,
            "notes":           self.notes,
        }

    def to_full_dict(self) -> dict:
        """Serialisation including stored tensors for independent replay."""
        d = self.to_dict()
        d["mps_tensors"] = self._mps_tensors
        d["mpo_tensors"] = self._mpo_tensors
        return d

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)

    def to_full_json(self, **kwargs) -> str:
        return json.dumps(self.to_full_dict(), ensure_ascii=False, **kwargs)

    @classmethod
    def from_dict(cls, d: dict) -> RayleighCertificateMPS:
        """Reconstruct from :meth:`to_dict` or :meth:`to_full_dict` output."""
        cert = cls(
            schema_version = d["schema_version"],
            claim          = d["claim"],
            theorem        = d["theorem"],
            assurance      = d["assurance"],
            backend        = d["backend"],
            n_sites        = int(d["n_sites"]),
            phys_dim       = int(d["phys_dim"]),
            mps_max_bond   = int(d["mps_max_bond"]),
            mpo_bond_dim   = int(d["mpo_bond_dim"]),
            rayleigh_upper = float(d["rayleigh_upper"]),
            rayleigh_lower = float(d["rayleigh_lower"]),
            input_digest   = d["input_digest"],
            htf_version    = d.get("htf_version", "unknown"),
            notes          = d.get("notes", ""),
        )
        if "mps_tensors" in d:
            cert._mps_tensors = d["mps_tensors"]
        if "mpo_tensors" in d:
            cert._mpo_tensors = d["mpo_tensors"]
        return cert


# ──────────────────────────────────────────────────────────────────────────────
# Rigorous Arb/Acb transfer-matrix contraction
# ──────────────────────────────────────────────────────────────────────────────

def _arb_rayleigh_mps(mps: MPS, mpo: MPO) -> tuple[float, float, float, str]:
    """Compute ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ for MPS/MPO via Arb transfer-matrix contractions.

    Algorithm
    ---------
    For ⟨ψ|ψ⟩: propagate a χ×χ left environment L left-to-right,
      L_new = Σ_s  conj(A_s)^T @ L @ A_s
    using arb_mat (real) or acb_mat (complex) matrix multiplication at each site.

    For ⟨ψ|H|ψ⟩: propagate a (χ·W·χ)×1 column vector C left-to-right.
    At each site build the (χ_r·W_r·χ_r) × (χ_l·W_l·χ_l) transfer matrix
      T[(β̄,b,β),(ᾱ,a,α)] = Σ_{s̄,s} conj(A[ᾱ,s̄,β̄]) · W[a,s̄,s,b] · A[α,s,β]
    in arb_mat / acb_mat arithmetic, then apply C_new = T @ C.

    Returns (lower, upper, radius, backend_label).  Raises ``ImportError``
    if python-flint is absent.
    """
    try:
        from flint import arb, arb_mat, ctx
    except ImportError as exc:
        raise ImportError(
            "_arb_rayleigh_mps requires python-flint "
            "(pip install python-flint)."
        ) from exc

    is_complex = (
        any(np.iscomplexobj(t) for t in mps.tensors)
        or any(np.iscomplexobj(t) for t in mpo.tensors)
    )

    if is_complex:
        from flint import acb, acb_mat
        mat_cls = acb_mat
        def av(x):
            return acb(complex(x))
        def cv(x):
            return acb(complex(x).conjugate())
    else:
        mat_cls = arb_mat  # type: ignore[assignment]
        def av(x):
            return arb(float(x))
        cv = av  # real: conjugate is identity

    saved_prec = ctx.prec
    try:
        ctx.prec = 128

        # ── ⟨ψ|ψ⟩ ────────────────────────────────────────────────────────
        L = mat_cls([[av(1.0)]])  # 1×1, grows to right boundary (1×1 for OBC)

        for A_np in mps.tensors:
            chi_l, d, chi_r = A_np.shape
            L_acc = [[av(0.0)] * chi_r for _ in range(chi_r)]
            for s in range(d):
                As = A_np[:, s, :]  # (chi_l, chi_r)
                A_mat  = mat_cls([[av(As[al, be]) for be in range(chi_r)]
                                  for al in range(chi_l)])
                A_matH = mat_cls([[cv(As[al, be]) for al in range(chi_l)]
                                  for be in range(chi_r)])
                contrib = A_matH * L * A_mat
                for i in range(chi_r):
                    for j in range(chi_r):
                        L_acc[i][j] = L_acc[i][j] + contrib[i, j]
            L = mat_cls(L_acc)

        denom = L[0, 0]
        if is_complex:
            if denom.real.contains(0):
                raise ValueError("Arb ⟨ψ|ψ⟩ ball contains zero; cannot certify.")
        else:
            if denom.contains(0):
                raise ValueError("Arb ⟨ψ|ψ⟩ ball contains zero; cannot certify.")

        # ── ⟨ψ|H|ψ⟩ ──────────────────────────────────────────────────────
        C = mat_cls([[av(1.0)]])  # starts 1×1 for OBC left boundary

        for A_np, W_np in zip(mps.tensors, mpo.tensors):
            chi_l, d, chi_r = A_np.shape
            W_l, d_bra, d_ket, W_r = W_np.shape
            dim_in  = chi_l * W_l * chi_l
            dim_out = chi_r * W_r * chi_r

            # Build T: (dim_out) × (dim_in) arb_mat.
            # Combined index order (row-major): (bond1, mpo_bond, bond2).
            T_rows = []
            for idx_out in range(dim_out):
                bb   = idx_out  // (W_r * chi_r)
                rem  = idx_out   % (W_r * chi_r)
                b    = rem      //  chi_r
                be   = rem       %  chi_r
                row = []
                for idx_in in range(dim_in):
                    ab   = idx_in  // (W_l * chi_l)
                    rem2 = idx_in   % (W_l * chi_l)
                    a    = rem2    //  chi_l
                    al   = rem2     %  chi_l
                    elem = av(0.0)
                    for s_bra in range(d_bra):
                        bra_v = cv(A_np[ab, s_bra, bb])
                        for s_ket in range(d_ket):
                            w_v   = av(W_np[a, s_bra, s_ket, b])
                            ket_v = av(A_np[al, s_ket, be])
                            elem  = elem + bra_v * w_v * ket_v
                    row.append(elem)
                T_rows.append(row)

            C = mat_cls(T_rows) * C

        numer = C[0, 0]

        if is_complex:
            q = numer / denom
            if not q.imag.contains(0):
                raise ArithmeticError(
                    "Acb Rayleigh quotient has non-zero imaginary part; "
                    "check Hermiticity of MPO."
                )
            lower = math.nextafter(float(q.real.lower()), -math.inf)
            upper = math.nextafter(float(q.real.upper()),  math.inf)
            backend = "flint-acb/mps-transfer/prec=128"
        else:
            q = numer / denom
            lower = math.nextafter(float(q.lower()), -math.inf)  # type: ignore[attr-defined]
            upper = math.nextafter(float(q.upper()),  math.inf)  # type: ignore[attr-defined]
            backend = "flint-arb/mps-transfer/prec=128"

        if not (math.isfinite(lower) and math.isfinite(upper)):
            raise ValueError(
                f"Arb MPS/MPO ball endpoints not finite: [{lower}, {upper}]"
            )
        radius = (upper - lower) / 2
        return lower, upper, radius, backend

    finally:
        ctx.prec = saved_prec


# ──────────────────────────────────────────────────────────────────────────────
# Certificate production
# ──────────────────────────────────────────────────────────────────────────────

def rayleigh_certificate_mps(
    mps: MPS,
    mpo: MPO,
    *,
    assurance: str = "reproducible",
    notes: str = "",
) -> RayleighCertificateMPS:
    """Produce a factorized Rayleigh certificate from an MPS trial state and MPO Hamiltonian.

    Parameters
    ----------
    mps:       Trial state as an :class:`~htf.mps.MPS`.
    mpo:       Hamiltonian as an :class:`~htf.mpo.MPO`.
    assurance: ``"reproducible"`` (default) — float64 MPS/MPO contractions;
               ``"rigorous"`` — Arb/Acb transfer-matrix contractions
               (requires python-flint).
    notes:     Optional annotation.

    Returns
    -------
    :class:`RayleighCertificateMPS`.

    Raises
    ------
    ValueError
        If site counts or physical dimensions differ, MPS norm is zero,
        or the computed Rayleigh quotient is non-finite.
    ImportError
        If assurance="rigorous" and python-flint is not installed.
    """
    if assurance not in ("reproducible", "rigorous"):
        raise ValueError(
            f"assurance must be 'reproducible' or 'rigorous'; got {assurance!r}"
        )

    if mps.n_sites != mpo.n_sites:
        raise ValueError(
            f"MPS n_sites={mps.n_sites} != MPO n_sites={mpo.n_sites}"
        )
    if mps.phys_dim != mpo.phys_dim:
        raise ValueError(
            f"MPS phys_dim={mps.phys_dim} != MPO phys_dim={mpo.phys_dim}"
        )

    if assurance == "rigorous":
        lower, upper, _, backend = _arb_rayleigh_mps(mps, mpo)
    else:
        # assurance == "reproducible": float64 contractions
        norm2 = float(mps_inner(mps, mps).real)
        if norm2 == 0.0:
            raise ValueError("MPS has zero norm; cannot compute Rayleigh quotient.")
        if not math.isfinite(norm2):
            raise ValueError(f"MPS ⟨ψ|ψ⟩ is non-finite: {norm2}")

        expectation = mpo_expectation(mpo, mps)
        rq = float(expectation.real) / norm2
        if not math.isfinite(rq):
            raise ValueError(f"Rayleigh quotient is non-finite: {rq}")

        upper = math.nextafter(rq, +math.inf)
        lower = math.nextafter(rq, -math.inf)
        backend = "float64-mps"

    digest = _canonical_digest_mps(mps, mpo)
    mpo_max_bond = max(int(t.shape[3]) for t in mpo.tensors)

    from .rayleigh_cert import _htf_version
    ver = _htf_version()

    assurance_tag = (
        "assurance=rigorous; Arb/Acb transfer-matrix"
        if assurance == "rigorous"
        else "assurance=reproducible"
    )
    return RayleighCertificateMPS(
        schema_version = MPS_CERT_SCHEMA,
        claim          = (
            f"E0 ≤ {upper:.17g}  "
            f"[Rayleigh-Ritz upper bound; MPS/MPO factorized storage; {assurance_tag}]"
        ),
        theorem        = EXPECTED_THEOREM,
        assurance      = assurance,
        backend        = backend,
        n_sites        = mps.n_sites,
        phys_dim       = mps.phys_dim,
        mps_max_bond   = mps.max_bond,
        mpo_bond_dim   = mpo_max_bond,
        rayleigh_upper = upper,
        rayleigh_lower = lower,
        input_digest   = digest,
        htf_version    = ver,
        notes          = notes,
        _mps_tensors   = [_encode_tensor(t) for t in mps.tensors],
        _mpo_tensors   = [_encode_tensor(t) for t in mpo.tensors],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────────────────────────────────────

def _restore_mps(tensors_encoded: list) -> MPS:
    return MPS([_decode_tensor(t) for t in tensors_encoded])


def _restore_mpo(tensors_encoded: list, n_sites: int) -> MPO:
    tensors = [_decode_tensor(t) for t in tensors_encoded]
    if len(tensors) != n_sites:
        raise ValueError(
            f"Stored MPO has {len(tensors)} tensors but n_sites={n_sites}"
        )
    return MPO(tensors)


def verify_rayleigh_certificate_mps(
    cert: RayleighCertificateMPS,
) -> RayleighCertificateMPS:
    """Verify a factorized Rayleigh certificate by replaying from stored tensors.

    For assurance="reproducible": replays float64 contractions and checks
    recomputed upper ≤ stored upper.

    For assurance="rigorous": replays Arb/Acb transfer-matrix contractions
    (requires python-flint) and checks recomputed upper ≤ stored upper.

    Both paths check:
    1. theorem field is intact.
    2. Tensors are present.
    3. Digest matches stored tensors (tamper detection).
    4. Recomputed upper ≤ stored upper.

    Sets ``cert.verified = True`` on success; raises on any failure.
    """
    if cert.assurance not in ("reproducible", "rigorous"):
        raise ValueError(
            f"Unknown assurance value: {cert.assurance!r}; "
            "expected 'reproducible' or 'rigorous'."
        )

    if cert.theorem != EXPECTED_THEOREM:
        raise ValueError(
            f"Theorem has been tampered:\n"
            f"  expected: {EXPECTED_THEOREM!r}\n"
            f"  stored:   {cert.theorem!r}"
        )

    if not cert._mps_tensors or not cert._mpo_tensors:
        raise ValueError(
            "Certificate lacks stored tensors.  "
            "Call cert.to_full_dict() / to_full_json() to include them before verifying."
        )

    mps = _restore_mps(cert._mps_tensors)
    mpo = _restore_mpo(cert._mpo_tensors, cert.n_sites)

    recomputed_digest = _canonical_digest_mps(mps, mpo)
    if recomputed_digest != cert.input_digest:
        raise ValueError("Digest mismatch — MPS/MPO tensors have been modified.")

    if cert.assurance == "rigorous":
        _, recomputed_upper, _, _ = _arb_rayleigh_mps(mps, mpo)
    else:
        norm2 = float(mps_inner(mps, mps).real)
        if norm2 == 0.0:
            raise ValueError("Restored MPS has zero norm.")
        expectation = mpo_expectation(mpo, mps)
        rq = float(expectation.real) / norm2
        if not math.isfinite(rq):
            raise ValueError(f"Recomputed Rayleigh quotient is non-finite: {rq}")
        recomputed_upper = math.nextafter(rq, +math.inf)

    if not (recomputed_upper <= cert.rayleigh_upper):
        raise ValueError(
            f"Verification failed: recomputed_upper={recomputed_upper!r} "
            f"> stored upper={cert.rayleigh_upper!r}"
        )

    cert.verified = True
    return cert
