"""HTF Public Benchmark Corpus — htf/corpus.py

A curated set of reproducible test cases for Rayleigh-quotient certification,
covering five categories required for public beta:

* **exact**          — analytically solvable small systems
* **near-degenerate** — spectral gap ≈ 0; tests bound tightness
* **complex**        — complex Hermitian H / complex ψ
* **ill-conditioned** — extreme scale ratios or near-rank-deficiency
* **cross-platform** — analytically-specified inputs; SHA-256 digest is
                        bit-for-bit identical on all platforms (numpy-only,
                        no eigvsh dependency)

Usage::

    from htf.corpus import CORPUS, run_corpus

    for case in CORPUS:
        cert = case.certificate()
        assert cert.upper <= case.expected_upper

    report = run_corpus()
    print(report.summary())
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from htf.rayleigh_cert import RayleighCertificate

# ──────────────────────────────────────────────────────────────────────────────
# CorpusCase dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CorpusCase:
    """One benchmark case for Rayleigh-quotient certification.

    Fields
    ------
    name            : unique kebab-case identifier.
    description     : human-readable description.
    tags            : list of category tags (``"exact"``, ``"near-degenerate"``,
                      ``"complex"``, ``"ill-conditioned"``, ``"cross-platform"``).
    H               : Hamiltonian matrix (real symmetric or complex Hermitian).
    psi             : trial state vector (need not be normalised).
    expected_E0     : exact ground-state energy (for reference only; not machine-checked
                      by this module).
    expected_upper  : maximum acceptable value for ``cert.upper``.  The corpus
                      validator asserts ``cert.upper <= expected_upper``.
    """
    name: str
    description: str
    tags: list[str]
    H: np.ndarray
    psi: np.ndarray
    expected_E0: float
    expected_upper: float

    def certificate(self) -> RayleighCertificate:
        """Produce a :class:`~htf.rayleigh_cert.RayleighCertificate` for this case."""
        from htf.rayleigh_cert import rayleigh_certificate
        return rayleigh_certificate(self.H, self.psi, notes=f"corpus:{self.name}")

    def run(self) -> CorpusCaseResult:
        """Run this case: certify and verify, return a :class:`CorpusCaseResult`."""
        from htf.rayleigh_cert import verify_rayleigh_certificate
        t0 = time.perf_counter()
        try:
            cert = self.certificate()
            verify_rayleigh_certificate(cert)
            elapsed = time.perf_counter() - t0
            passed = cert.upper <= self.expected_upper + 1e-15
            return CorpusCaseResult(
                case=self,
                passed=passed,
                upper=cert.upper,
                radius=cert.radius,
                backend=cert.backend,
                input_digest=cert.input_digest,
                elapsed_s=elapsed,
                error=None,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            return CorpusCaseResult(
                case=self,
                passed=False,
                upper=float("nan"),
                radius=float("nan"),
                backend="error",
                input_digest="",
                elapsed_s=elapsed,
                error=str(exc),
            )


# ──────────────────────────────────────────────────────────────────────────────
# CorpusCaseResult & CorpusReport
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CorpusCaseResult:
    """Result of running a single :class:`CorpusCase`."""
    case: CorpusCase
    passed: bool
    upper: float
    radius: float
    backend: str
    input_digest: str
    elapsed_s: float
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.case.name,
            "tags": self.case.tags,
            "passed": self.passed,
            "expected_upper": self.case.expected_upper,
            "upper": self.upper,
            "radius": self.radius,
            "backend": self.backend,
            "input_digest": self.input_digest,
            "elapsed_s": self.elapsed_s,
            "error": self.error,
        }


@dataclass
class CorpusReport:
    """Result of running the full benchmark corpus."""
    results: list[CorpusCaseResult]

    @property
    def n_total(self) -> int:
        return len(self.results)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self) -> int:
        return self.n_total - self.n_passed

    def summary(self) -> str:
        lines = [
            f"HTF Corpus: {self.n_passed}/{self.n_total} passed",
            f"{'Name':<35} {'Tags':<35} {'Pass':<6} {'Upper':>18} {'Radius':>12} {'ms':>8}",
            "-" * 118,
        ]
        for r in self.results:
            tag_str = ",".join(r.case.tags)
            ok = "PASS" if r.passed else "FAIL"
            lines.append(
                f"{r.case.name:<35} {tag_str:<35} {ok:<6} "
                f"{r.upper:>18.10g} {r.radius:>12.3e} {r.elapsed_s * 1000:>8.1f}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_total": self.n_total,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "results": [r.to_dict() for r in self.results],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Corpus construction helpers
# ──────────────────────────────────────────────────────────────────────────────

def _exact_gs(H: np.ndarray) -> tuple[float, np.ndarray]:
    """Return (E0, psi0) from full diagonalisation."""
    evals, evecs = np.linalg.eigh(H)
    return float(evals[0]), evecs[:, 0]


def _tfim_ham(n: int, J: float = 1.0, h: float = 0.5) -> np.ndarray:
    from htf.variational import transverse_ising_ham
    return transverse_ising_ham(n, J=J, h=h)


def _xx_ham(n: int, J: float = 1.0) -> np.ndarray:
    from htf.variational import xx_model_ham
    return xx_model_ham(n, J=J)


def _random_sym(n: int, seed: int = 0) -> np.ndarray:
    """Fixed-seed random real symmetric matrix (bit-for-bit reproducible)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return (A + A.T) / 2


def _build_corpus() -> list[CorpusCase]:
    cases: list[CorpusCase] = []

    # ── 1. Trivial 2×2 diagonal ──────────────────────────────────────────────
    H1 = np.diag([0.0, 1.0])
    cases.append(CorpusCase(
        name="trivial_2x2",
        description=(
            "Diagonal H = diag(0, 1).  Ground state |0⟩ analytically specified.  "
            "Exact E0 = 0.  Simplest possible Rayleigh certification."
        ),
        tags=["exact", "real", "n=2"],
        H=H1,
        psi=np.array([1.0, 0.0]),
        expected_E0=0.0,
        expected_upper=1e-9,
    ))

    # ── 2. Near-degenerate 2×2 (gap = 1e-8) ─────────────────────────────────
    H2 = np.diag([0.0, 1e-8])
    cases.append(CorpusCase(
        name="near_degenerate_2x2",
        description=(
            "2×2 diagonal with spectral gap = 1e-8.  Tests that Arb interval "
            "arithmetic resolves the bound even when levels nearly coincide."
        ),
        tags=["near-degenerate", "real", "n=2"],
        H=H2,
        psi=np.array([1.0, 0.0]),
        expected_E0=0.0,
        expected_upper=1e-9,
    ))

    # ── 3. Near-degenerate n=4 (multiple near-zero levels) ───────────────────
    H3 = np.diag([0.0, 1e-7, 1e-7 + 1e-14, 1.0])
    cases.append(CorpusCase(
        name="near_degenerate_n4",
        description=(
            "4×4 diagonal with levels 0, 1e-7, 1e-7+1e-14, 1.  Near-degenerate "
            "excited subspace tests bound robustness."
        ),
        tags=["near-degenerate", "real", "n=4"],
        H=H3,
        psi=np.array([1.0, 0.0, 0.0, 0.0]),
        expected_E0=0.0,
        expected_upper=1e-9,
    ))

    # ── 4. Complex Hermitian 2×2 ─────────────────────────────────────────────
    # H = [[1, i], [-i, 1]]; eigenvalues 0, 2; GS = [1, i]/sqrt(2)
    H4 = np.array([[1.0 + 0j, 1j], [-1j, 1.0 + 0j]])
    psi4 = np.array([1.0, 1j]) / np.sqrt(2)
    cases.append(CorpusCase(
        name="complex_hermitian_2x2",
        description=(
            "Complex Hermitian 2×2: H = [[1, i], [-i, 1]].  Exact GS = [1, i]/√2 "
            "with E0 = 0.  Tests Acb (complex interval) arithmetic path."
        ),
        tags=["complex", "exact", "n=2"],
        H=H4,
        psi=psi4,
        expected_E0=0.0,
        expected_upper=1e-9,
    ))

    # ── 5. Complex Hermitian n=4 (block diagonal, analytically-specified GS) ──
    # H = diag_block([[1,i],[-i,1]], [[3,0],[0,4]]).
    # Top-left block: eigenvalues 0, 2; GS = [1, i]/sqrt(2).
    # Full GS: psi = [1, i, 0, 0]/sqrt(2), E0 = 0.  No eigvsh needed.
    H5 = np.array([
        [1.0+0j,  1j,    0.0, 0.0],
        [-1j,     1.0+0j, 0.0, 0.0],
        [0.0,     0.0,    3.0, 0.0],
        [0.0,     0.0,    0.0, 4.0],
    ])
    psi5 = np.array([1.0, 1j, 0.0, 0.0]) / np.sqrt(2)
    cases.append(CorpusCase(
        name="complex_hermitian_n4_block",
        description=(
            "Block-diagonal complex Hermitian n=4: top 2×2 block [[1,i],[-i,1]], "
            "bottom diag(3,4).  GS = [1,i,0,0]/√2, E0 = 0.  Analytically specified "
            "— no eigvsh dependency.  Tests Acb complex interval arithmetic."
        ),
        tags=["complex", "exact", "n=4"],
        H=H5,
        psi=psi5,
        expected_E0=0.0,
        expected_upper=1e-9,
    ))

    # ── 6. Ill-conditioned: large scale separation ───────────────────────────
    H6 = np.diag([0.0, 1e12])
    cases.append(CorpusCase(
        name="ill_conditioned_scale_1e12",
        description=(
            "Diagonal H = diag(0, 1e12).  Extreme scale ratio tests that "
            "Arb interval arithmetic maintains tight bounds under large entries."
        ),
        tags=["ill-conditioned", "real", "n=2"],
        H=H6,
        psi=np.array([1.0, 0.0]),
        expected_E0=0.0,
        expected_upper=1e-9,
    ))

    # ── 7. Ill-conditioned: near-zero GS with large excited states ───────────
    H7 = np.diag([1e-15, 1.0, 2.0, 3.0])
    cases.append(CorpusCase(
        name="ill_conditioned_near_zero_gs",
        description=(
            "GS energy ≈ 1e-15 (near machine epsilon).  Tests that the bound "
            "correctly resolves near-zero ground state against O(1) excited states."
        ),
        tags=["ill-conditioned", "real", "n=4"],
        H=H7,
        psi=np.array([1.0, 0.0, 0.0, 0.0]),
        expected_E0=1e-15,
        expected_upper=1e-14,
    ))

    # ── 8. TFIM n=4 exact GS ─────────────────────────────────────────────────
    H8 = _tfim_ham(4, J=1.0, h=0.5)
    e8, psi8 = _exact_gs(H8)
    cases.append(CorpusCase(
        name="tfim_n4_exact",
        description=(
            "Transverse-field Ising model n=4, J=1, h=0.5.  Exact GS from full "
            "diagonalisation.  Rayleigh quotient = E0 up to numerical precision."
        ),
        tags=["exact", "real", "physics", "n=4"],
        H=H8,
        psi=psi8,
        expected_E0=e8,
        expected_upper=e8 + 1e-9,
    ))

    # ── 9. XX model n=4 exact GS ─────────────────────────────────────────────
    H9 = _xx_ham(4, J=1.0)
    e9, psi9 = _exact_gs(H9)
    cases.append(CorpusCase(
        name="xx_n4_exact",
        description=(
            "XX model n=4, J=1.  Exact GS from full diagonalisation.  "
            "Tests the sign-corrected XX Hamiltonian (P0-4 fix)."
        ),
        tags=["exact", "real", "physics", "n=4"],
        H=H9,
        psi=psi9,
        expected_E0=e9,
        expected_upper=e9 + 1e-9,
    ))

    # ── 10. Cross-platform: random n=4, analytically-specified psi ───────────
    # H is deterministic (fixed numpy seed), psi is the uniform state (analytical).
    # SHA-256(H_bytes || psi_bytes) is bit-for-bit identical on all platforms.
    H10 = _random_sym(4, seed=0)
    psi10_uniform = np.ones(4) / 2.0  # uniform state, analytically specified
    e10 = float(np.linalg.eigvalsh(H10)[0])
    # Rayleigh quotient of uniform state = sum(H) / (psi^T psi) / 4 = ...
    rq10 = float(psi10_uniform @ H10 @ psi10_uniform) / float(psi10_uniform @ psi10_uniform)
    cases.append(CorpusCase(
        name="cross_platform_random_n4_uniform",
        description=(
            "Random symmetric n=4 (numpy seed=0), psi = uniform [1/2, 1/2, 1/2, 1/2]. "
            "Both H and psi are analytically specified so the SHA-256 input_digest "
            "is bit-for-bit identical on all platforms (no eigvsh dependency)."
        ),
        tags=["cross-platform", "real", "n=4"],
        H=H10,
        psi=psi10_uniform,
        expected_E0=e10,
        expected_upper=rq10 + 1e-9,
    ))

    # ── 11. Cross-platform: 2-qubit Bell-like state ───────────────────────────
    # H = Heisenberg ZZ + XX + YY coupling (real), psi = |00⟩+|11⟩/sqrt(2)
    _sz = np.array([[1.0, 0.0], [0.0, -1.0]])
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    # sy_real = [[0, -1], [1, 0]] (imaginary part set to 0 for real H)
    H11 = -(np.kron(sx, sx) + np.diag([1.0, -1.0, -1.0, 1.0]))
    H11 = (H11 + H11.T) / 2  # enforce symmetry
    psi11 = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2)  # Bell state |Φ+⟩
    rq11 = float(psi11 @ H11 @ psi11)
    e11 = float(np.linalg.eigvalsh(H11)[0])
    cases.append(CorpusCase(
        name="cross_platform_bell_2qubit",
        description=(
            "2-qubit ZZ+XX Hamiltonian, psi = Bell state |Φ+⟩ = (|00⟩+|11⟩)/√2. "
            "Analytically specified inputs; cross-platform deterministic digest."
        ),
        tags=["cross-platform", "real", "n=4"],
        H=H11,
        psi=psi11,
        expected_E0=e11,
        expected_upper=rq11 + 1e-9,
    ))

    return cases


CORPUS: list[CorpusCase] = _build_corpus()


# ──────────────────────────────────────────────────────────────────────────────
# Public runner
# ──────────────────────────────────────────────────────────────────────────────

def run_corpus(cases: list[CorpusCase] | None = None) -> CorpusReport:
    """Run all (or a subset of) corpus cases and return a :class:`CorpusReport`.

    Parameters
    ----------
    cases : list[CorpusCase] | None
        Cases to run.  If ``None``, runs :data:`CORPUS`.

    Returns
    -------
    :class:`CorpusReport`
    """
    if cases is None:
        cases = CORPUS
    results = [c.run() for c in cases]
    return CorpusReport(results=results)


def corpus_by_tag(*tags: str) -> list[CorpusCase]:
    """Return corpus cases that have *all* of the given tags."""
    return [c for c in CORPUS if all(t in c.tags for t in tags)]
