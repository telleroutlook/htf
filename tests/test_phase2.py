"""Phase 2 tests: 1D lattice operators and certified mode."""
from __future__ import annotations

import numpy as np
import pytest

from htf import Box, Certificate, TensorFunctor, contract
from htf.lattice import (
    effect_box,
    heat_step_box,
    laplacian_box,
    site_wire,
    state_box,
)


# ─────────────────────── laplacian_box ─────────────────────────────

class TestLaplacianBox:
    def test_shape(self):
        n = 5
        box, L = laplacian_box(n)
        assert L.shape == (n, n)
        assert box.dom == (site_wire(n),)
        assert box.cod == (site_wire(n),)

    def test_tridiagonal_values(self):
        _, L = laplacian_box(4, dx=1.0)
        np.testing.assert_allclose(np.diag(L), -2.0)
        np.testing.assert_allclose(np.diag(L, 1), 1.0)
        np.testing.assert_allclose(np.diag(L, -1), 1.0)

    def test_dx_scaling(self):
        _, L1 = laplacian_box(4, dx=1.0)
        _, L2 = laplacian_box(4, dx=2.0)
        np.testing.assert_allclose(L2, L1 / 4.0)

    def test_symmetry(self):
        _, L = laplacian_box(6)
        np.testing.assert_allclose(L, L.T)


# ─────────────────────── heat_step_box ─────────────────────────────

class TestHeatStepBox:
    def test_shape(self):
        box, M = heat_step_box(6, D=0.1, dt=0.05)
        assert M.shape == (6, 6)

    def test_single_step_matches_direct(self):
        """HTF one-step result == direct matrix-vector product."""
        n, D, dt, dx = 8, 0.1, 0.05, 1.0
        u0 = np.sin(np.linspace(0, np.pi, n))

        _, L = laplacian_box(n, dx)
        u_ref = (np.eye(n) + dt * D * L) @ u0

        psi_b, psi_a = state_box("psi", u0)
        step_b, step_a = heat_step_box(n, D, dt, dx)
        diagram = psi_b >> step_b
        F = TensorFunctor({"psi": psi_a, "heat_step": step_a})
        np.testing.assert_allclose(contract(diagram, F), u_ref, rtol=1e-12)

    def test_multiple_steps_match_iteration(self):
        """Chaining N step boxes matches iterating the matrix N times."""
        n, D, dt, nsteps = 8, 0.1, 0.05, 5
        u0 = np.zeros(n)
        u0[n // 2] = 1.0

        _, M = heat_step_box(n, D, dt)
        u_ref = u0.copy()
        for _ in range(nsteps):
            u_ref = M @ u_ref

        psi_b, psi_a = state_box("psi", u0)
        step_b, step_a = heat_step_box(n, D, dt)
        diagram = psi_b
        for _ in range(nsteps):
            diagram = diagram >> step_b

        F = TensorFunctor({"psi": psi_a, "heat_step": step_a})
        np.testing.assert_allclose(contract(diagram, F), u_ref, rtol=1e-12)

    def test_scalar_amplitude_heat(self):
        """<v| M |u> computed via diagram matches direct inner product."""
        n, D, dt = 6, 0.1, 0.04
        u0 = np.random.default_rng(42).random(n)
        v = np.random.default_rng(7).random(n)

        _, M = heat_step_box(n, D, dt)
        ref = v @ M @ u0

        psi_b, psi_a = state_box("psi", u0)
        phi_b, phi_a = effect_box("phi", v)
        step_b, step_a = heat_step_box(n, D, dt)
        diagram = psi_b >> step_b >> phi_b
        F = TensorFunctor({"psi": psi_a, "heat_step": step_a, "phi": phi_a})
        result = contract(diagram, F)
        assert np.isclose(float(result), ref, rtol=1e-12)


# ─────────────────────── certified mode ────────────────────────────

class TestCertifiedMode:
    def _simple_diagram(self, n=4):
        u0 = np.array([1.0, 0.5, 0.25, 0.0])
        psi_b, psi_a = state_box("psi", u0)
        step_b, step_a = heat_step_box(n, 0.1, 0.05)
        diagram = psi_b >> step_b
        F = TensorFunctor({"psi": psi_a, "heat_step": step_a})
        return diagram, F

    def test_returns_certificate(self):
        d, F = self._simple_diagram()
        cert = contract(d, F, mode="certified")
        assert isinstance(cert, Certificate)

    def test_mode_field(self):
        d, F = self._simple_diagram()
        cert = contract(d, F, mode="certified")
        assert cert.mode == "certified"

    def test_backend_field(self):
        d, F = self._simple_diagram()
        cert = contract(d, F, mode="certified")
        assert cert.backend == "flint-arb"

    def test_error_bound_nonnegative(self):
        d, F = self._simple_diagram()
        cert = contract(d, F, mode="certified")
        assert cert.error_bound is not None
        assert cert.error_bound >= 0.0

    def test_midpoint_matches_float(self):
        """Certified midpoint must agree with float result within error_bound."""
        d, F = self._simple_diagram()
        float_result = contract(d, F, mode="float")
        cert = contract(d, F, mode="certified")
        np.testing.assert_allclose(
            cert.result, float_result, atol=cert.error_bound + 1e-14
        )

    def test_certified_scalar_amplitude(self):
        """Certified scalar amplitude (bra-ket diagram) agrees with float."""
        n = 4
        u0 = np.zeros(n); u0[0] = 1.0
        v = np.zeros(n); v[0] = 1.0
        psi_b, psi_a = state_box("psi", u0)
        phi_b, phi_a = effect_box("phi", v)
        step_b, step_a = heat_step_box(n, 0.1, 0.05)
        diagram = psi_b >> step_b >> phi_b
        F = TensorFunctor({"psi": psi_a, "heat_step": step_a, "phi": phi_a})

        float_val = float(contract(diagram, F, mode="float"))
        cert = contract(diagram, F, mode="certified")
        assert abs(float(cert.result) - float_val) <= cert.error_bound + 1e-14

    def test_error_bound_accumulates_with_steps(self):
        """More steps → larger (or equal) error bound."""
        n, D, dt = 6, 0.1, 0.04
        u0 = np.random.default_rng(0).random(n)
        psi_b, psi_a = state_box("psi", u0)
        step_b, step_a = heat_step_box(n, D, dt)
        F1 = TensorFunctor({"psi": psi_a, "heat_step": step_a})
        F10 = TensorFunctor({"psi": psi_a, "heat_step": step_a})

        d1 = psi_b >> step_b
        d10 = psi_b
        for _ in range(10):
            d10 = d10 >> step_b

        cert1 = contract(d1, F1, mode="certified")
        cert10 = contract(d10, F10, mode="certified")
        assert cert10.error_bound >= cert1.error_bound

    def test_certificate_serializable(self):
        """Certificate.to_json() round-trips without error."""
        import json
        d, F = self._simple_diagram()
        cert = contract(d, F, mode="certified")
        s = cert.to_json()
        obj = json.loads(s)
        assert obj["mode"] == "certified"
        assert obj["error_bound"] is not None
