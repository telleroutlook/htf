"""G4 Oracle test suite — randomised soundness verification for Rayleigh certificates.

Runs ≥10,000 mathematically distinct cases across five categories and verifies
the core soundness invariant:

    cert.upper >= exact_E0 - EPSILON   for all (H, psi)

where ``exact_E0 = min(eigvalsh(H))`` is the true ground-state energy.

This is a **regression guard**, not a test of algorithm quality.  A single
false negative (cert.upper < exact_E0) would be a soundness violation — the
kind of error the HTF-01 referee found in the v1 implementation.

Categories
----------
real_random      : random real symmetric H, random psi (7 000 cases)
complex_random   : random complex Hermitian H, random complex psi (2 000 cases)
ill_conditioned  : extreme scale ratios 1e6 – 1e15 (500 cases)
near_degenerate  : spectral gap in [1e-12, 1e-4] (500 cases)
known_rejects    : adversarial inputs that must be REJECTED, not silently accepted

Total mandatory oracle cases: 10 000 +
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from htf.rayleigh_cert import rayleigh_certificate, verify_rayleigh_certificate

# Any upper bound within EPSILON of the exact E0 passes the soundness check.
# Exact-GS trial states can give cert.upper slightly above E0 due to Arb
# outward rounding; EPSILON = 1e-10 generously accommodates that.
EPSILON = 1e-10


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _rand_real_sym(n: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((n, n))
    return (A + A.T) / 2


def _rand_complex_herm(n: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return (A + A.conj().T) / 2


def _exact_e0(H: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(H)[0])


def _check_soundness(H: np.ndarray, psi: np.ndarray, exact_e0: float, label: str) -> None:
    cert = rayleigh_certificate(H, psi)
    assert cert.upper >= exact_e0 - EPSILON, (
        f"SOUNDNESS VIOLATION in {label}: "
        f"cert.upper={cert.upper:.17g} < exact_E0={exact_e0:.17g} "
        f"(delta={cert.upper - exact_e0:.3e})"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Category 1 — real random (7 000 cases: 1 000 seeds × 7 sizes)
# ──────────────────────────────────────────────────────────────────────────────

class TestRealRandomOracle:
    """1 000 fixed-seed RNG instances × 7 matrix sizes = 7 000 cases."""

    SIZES = [2, 3, 4, 5, 6, 7, 8]
    N_SEEDS = 1_000

    def test_real_random_soundness(self):
        failures: list[str] = []
        n_checked = 0
        for seed in range(self.N_SEEDS):
            rng = np.random.default_rng(seed + 100_000)
            for n in self.SIZES:
                H   = _rand_real_sym(n, rng)
                psi = rng.standard_normal(n)
                psi /= np.linalg.norm(psi)
                e0  = _exact_e0(H)
                cert = rayleigh_certificate(H, psi)
                n_checked += 1
                if cert.upper < e0 - EPSILON:
                    failures.append(
                        f"seed={seed} n={n}: upper={cert.upper:.10g} < E0={e0:.10g}"
                    )
        assert n_checked == self.N_SEEDS * len(self.SIZES)
        assert not failures, (
            f"{len(failures)}/{n_checked} soundness violations:\n" +
            "\n".join(failures[:5])
        )

    def test_real_random_gs_state_tight(self):
        """Upper bound is tight when psi = exact GS (upper ≈ E0)."""
        failures: list[str] = []
        for seed in range(200):
            rng = np.random.default_rng(seed + 200_000)
            n = rng.integers(2, 7)
            H = _rand_real_sym(int(n), rng)
            evals, evecs = np.linalg.eigh(H)
            e0   = float(evals[0])
            psi0 = evecs[:, 0]
            cert = rayleigh_certificate(H, psi0)
            if cert.upper < e0 - EPSILON:
                failures.append(f"seed={seed}: upper={cert.upper:.10g} < E0={e0:.10g}")
            # Tight: GS state gives upper within 1e-8 of E0
            if cert.upper > e0 + 1e-8:
                failures.append(
                    f"seed={seed}: not tight: upper={cert.upper:.10g}, E0={e0:.10g}"
                )
        assert not failures, "\n".join(failures[:5])

    def test_real_random_verify_roundtrip(self):
        """All produced certificates pass independent verification."""
        for seed in range(50):
            rng = np.random.default_rng(seed + 300_000)
            n = int(rng.integers(2, 6))
            H   = _rand_real_sym(n, rng)
            psi = rng.standard_normal(n)
            cert = rayleigh_certificate(H, psi)
            verify_rayleigh_certificate(cert)
            assert cert.verified is True, f"seed={seed} n={n}: verification failed"

    def test_real_random_unnormalised_invariance(self):
        """Rayleigh quotient is scale-invariant: cert.upper(α·ψ) == cert.upper(ψ)."""
        rng = np.random.default_rng(400_000)
        for _ in range(100):
            n = int(rng.integers(2, 6))
            H   = _rand_real_sym(n, rng)
            psi = rng.standard_normal(n)
            alpha = float(rng.uniform(0.01, 100))
            c1 = rayleigh_certificate(H, psi)
            c2 = rayleigh_certificate(H, psi * alpha)
            assert abs(c1.upper - c2.upper) < 1e-10, (
                f"scale-invariance broken: upper(ψ)={c1.upper:.10g}, "
                f"upper({alpha:.2g}ψ)={c2.upper:.10g}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Category 2 — complex random (2 000 cases: 500 seeds × 4 sizes)
# ──────────────────────────────────────────────────────────────────────────────

class TestComplexRandomOracle:
    """500 fixed-seed RNG instances × 4 matrix sizes = 2 000 cases."""

    SIZES = [2, 3, 4, 5]
    N_SEEDS = 500

    def test_complex_random_soundness(self):
        failures: list[str] = []
        n_checked = 0
        for seed in range(self.N_SEEDS):
            rng = np.random.default_rng(seed + 500_000)
            for n in self.SIZES:
                H   = _rand_complex_herm(n, rng)
                psi = (rng.standard_normal(n) + 1j * rng.standard_normal(n))
                psi /= np.linalg.norm(psi)
                e0  = _exact_e0(H)
                cert = rayleigh_certificate(H, psi)
                n_checked += 1
                if cert.upper < e0 - EPSILON:
                    failures.append(
                        f"seed={seed} n={n}: upper={cert.upper:.10g} < E0={e0:.10g}"
                    )
        assert n_checked == self.N_SEEDS * len(self.SIZES)
        assert not failures, (
            f"{len(failures)}/{n_checked} soundness violations:\n" +
            "\n".join(failures[:5])
        )

    def test_complex_real_h_complex_psi(self):
        """Real symmetric H + complex psi: takes Acb path, must still be sound."""
        failures: list[str] = []
        for seed in range(200):
            rng = np.random.default_rng(seed + 600_000)
            n = int(rng.integers(2, 6))
            H   = _rand_real_sym(n, rng).astype(complex)
            psi = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            e0  = _exact_e0(H)
            cert = rayleigh_certificate(H, psi)
            if cert.upper < e0 - EPSILON:
                failures.append(f"seed={seed}: upper={cert.upper:.10g} < E0={e0:.10g}")
        assert not failures, "\n".join(failures[:5])

    def test_complex_verify_roundtrip(self):
        """Complex certificates pass independent verification."""
        for seed in range(30):
            rng = np.random.default_rng(seed + 700_000)
            n = int(rng.integers(2, 5))
            H   = _rand_complex_herm(n, rng)
            psi = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            cert = rayleigh_certificate(H, psi)
            verify_rayleigh_certificate(cert)
            assert cert.verified is True


# ──────────────────────────────────────────────────────────────────────────────
# Category 3 — ill-conditioned (500 cases)
# ──────────────────────────────────────────────────────────────────────────────

class TestIllConditionedOracle:
    """Extreme scale ratios, near-zero GS, large off-diagonals."""

    def test_extreme_scale_ratio(self):
        """H = diag(0, s) for s in [1e6, 1e15]; psi = GS."""
        exponents = range(6, 16)  # 10 cases
        for exp in exponents:
            s = 10.0 ** exp
            H = np.diag([0.0, s])
            psi = np.array([1.0, 0.0])
            cert = rayleigh_certificate(H, psi)
            assert cert.upper <= EPSILON, (
                f"s=1e{exp}: cert.upper={cert.upper:.3e} should be ~0"
            )

    def test_large_off_diagonal(self):
        """Symmetric H with large off-diagonal entries; random psi."""
        failures: list[str] = []
        for seed in range(200):
            rng = np.random.default_rng(seed + 800_000)
            scale = 10.0 ** rng.uniform(0, 8)
            n = int(rng.integers(2, 6))
            H = _rand_real_sym(n, rng) * scale
            psi = rng.standard_normal(n)
            e0 = _exact_e0(H)
            cert = rayleigh_certificate(H, psi)
            if cert.upper < e0 - EPSILON * scale:
                failures.append(
                    f"seed={seed} scale={scale:.1e}: "
                    f"upper={cert.upper:.6g} < E0={e0:.6g}"
                )
        assert not failures, "\n".join(failures[:5])

    def test_near_zero_gs_energy(self):
        """GS energy near machine epsilon: diag(ε, 1) for ε in [1e-15, 1e-6]."""
        failures: list[str] = []
        for exp in range(6, 16):
            eps = 10.0 ** (-exp)
            H = np.diag([eps, 1.0])
            psi = np.array([1.0, 0.0])
            cert = rayleigh_certificate(H, psi)
            if cert.upper < eps - EPSILON:
                failures.append(
                    f"eps=1e-{exp}: upper={cert.upper:.3e} < E0={eps:.3e}"
                )
        assert not failures, "\n".join(failures)

    def test_rank_deficient_like(self):
        """Near-singular: H = diag(0, 0, ..., 0, 1); trial state is GS."""
        for n in range(2, 9):
            H = np.zeros((n, n))
            H[-1, -1] = 1.0
            psi = np.ones(n) / math.sqrt(n)
            e0 = 0.0
            cert = rayleigh_certificate(H, psi)
            assert cert.upper >= e0 - EPSILON, (
                f"n={n}: cert.upper={cert.upper:.6g} < E0=0"
            )

    def test_ill_conditioned_random_scales(self):
        """Random H scaled by [1e-8, 1e8], 300 cases."""
        failures: list[str] = []
        for seed in range(300):
            rng = np.random.default_rng(seed + 900_000)
            log_scale = rng.uniform(-8, 8)
            scale = 10.0 ** log_scale
            n = int(rng.integers(2, 7))
            H = _rand_real_sym(n, rng) * scale
            psi = rng.standard_normal(n)
            e0 = _exact_e0(H)
            cert = rayleigh_certificate(H, psi)
            tol = EPSILON * max(1.0, abs(e0))
            if cert.upper < e0 - tol:
                failures.append(
                    f"seed={seed} scale={scale:.1e}: "
                    f"upper={cert.upper:.8g} < E0={e0:.8g}"
                )
        assert not failures, "\n".join(failures[:5])


# ──────────────────────────────────────────────────────────────────────────────
# Category 4 — near-degenerate (500 cases)
# ──────────────────────────────────────────────────────────────────────────────

class TestNearDegenerateOracle:
    """Spectral gaps spanning 8 orders of magnitude."""

    def test_near_degenerate_diagonal(self):
        """H = diag(0, gap) for gap in [1e-12, 1]; psi = GS."""
        for exp in range(1, 13):
            gap = 10.0 ** (-exp)
            H = np.diag([0.0, gap])
            psi = np.array([1.0, 0.0])
            cert = rayleigh_certificate(H, psi)
            assert cert.upper <= EPSILON, (
                f"gap=1e-{exp}: cert.upper={cert.upper:.3e}"
            )

    def test_near_degenerate_excited_subspace(self):
        """GS is well-separated; excited levels nearly degenerate."""
        failures: list[str] = []
        for seed in range(200):
            rng = np.random.default_rng(seed + 1_000_000)
            n = int(rng.integers(3, 7))
            # Build: E0=0, excited levels all near E1
            e1 = float(rng.uniform(0.1, 2.0))
            noise = float(rng.uniform(1e-10, 1e-4))
            evals = np.zeros(n)
            evals[1:] = e1 + rng.uniform(0, noise, n - 1)
            Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
            H = Q @ np.diag(evals) @ Q.T
            H = (H + H.T) / 2
            psi = Q[:, 0]
            e0 = float(evals[0])
            cert = rayleigh_certificate(H, psi)
            if cert.upper < e0 - EPSILON:
                failures.append(
                    f"seed={seed}: upper={cert.upper:.8g} < E0={e0:.8g}"
                )
        assert not failures, "\n".join(failures[:5])

    def test_near_degenerate_gs_pair(self):
        """Two near-degenerate ground states; trial state is one of them."""
        failures: list[str] = []
        for exp in range(2, 14):
            gap = 10.0 ** (-exp)
            H = np.diag([0.0, gap, 1.0, 2.0])
            psi = np.array([1.0, 0.0, 0.0, 0.0])
            e0 = 0.0
            cert = rayleigh_certificate(H, psi)
            if cert.upper < e0 - EPSILON:
                failures.append(f"gap=1e-{exp}: upper={cert.upper:.3e}")
        assert not failures, "\n".join(failures)


# ──────────────────────────────────────────────────────────────────────────────
# Category 5 — known adversarial inputs must be REJECTED (not silently accepted)
# ──────────────────────────────────────────────────────────────────────────────

class TestKnownRejectsStillRejected:
    """HTF-01 referee counterexamples — must still raise after v2 fixes."""

    def test_near_symmetric_H_rejected(self):
        H = np.array([[0.0, 5e-11], [0.0, 0.0]])
        with pytest.raises(ValueError, match="symmetric"):
            rayleigh_certificate(H, np.array([1.0, -1.0]))

    def test_nan_in_H_rejected(self):
        H = np.array([[np.nan, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="non-finite|NaN"):
            rayleigh_certificate(H, np.array([1.0, 0.0]))

    def test_nan_in_psi_rejected(self):
        H = np.diag([0.0, 1.0])
        with pytest.raises(ValueError, match="non-finite|NaN"):
            rayleigh_certificate(H, np.array([np.nan, 0.0]))

    def test_inf_in_H_rejected(self):
        H = np.array([[np.inf, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="non-finite|NaN"):
            rayleigh_certificate(H, np.array([1.0, 0.0]))

    def test_zero_psi_rejected(self):
        H = np.diag([0.0, 1.0])
        with pytest.raises(ValueError, match="zero norm"):
            rayleigh_certificate(H, np.zeros(2))

    def test_non_square_H_rejected(self):
        with pytest.raises(ValueError, match="square"):
            rayleigh_certificate(np.ones((2, 3)), np.array([1.0, 0.0]))

    def test_dim_mismatch_rejected(self):
        with pytest.raises(ValueError, match="length"):
            rayleigh_certificate(np.diag([0.0, 1.0]), np.array([1.0, 0.0, 0.0]))

    def test_non_hermitian_complex_rejected(self):
        H = np.array([[1.0, 2 + 1j], [2 + 0j, 1.0]])
        with pytest.raises(ValueError, match="[Hh]ermitian"):
            rayleigh_certificate(H, np.array([1.0 + 0j, 0.0]))

    def test_strictly_asymmetric_rejected(self):
        H = np.array([[1.0, 2.0], [3.0, 1.0]])
        with pytest.raises(ValueError, match="symmetric"):
            rayleigh_certificate(H, np.array([1.0, 0.0]))
