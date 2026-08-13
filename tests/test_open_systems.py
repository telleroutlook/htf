"""Tests for htf/open_systems.py — open quantum systems and CPTP maps."""
import numpy as np
import pytest

from htf.open_systems import (
    check_density_matrix,
    check_kraus_completeness,
    choi_matrix,
    density_matrix_from_pure,
    lindblad_step,
    lindblad_superoperator,
    partial_trace,
    steady_state,
)

# ─────── Pauli matrices ────────────────────────────────────────────────────
I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)

ket0 = np.array([1.0, 0.0])  # |0⟩
ket1 = np.array([0.0, 1.0])  # |1⟩
bell = np.array([1, 0, 0, 1]) / np.sqrt(2)  # |Φ+⟩ = (|00⟩+|11⟩)/√2


# ─────────────────── TestDensityMatrixFromPure ────────────────────────────

class TestDensityMatrixFromPure:

    def test_shape(self):
        rho = density_matrix_from_pure(ket0)
        assert rho.shape == (2, 2)

    def test_pure_0_is_projector_0(self):
        rho = density_matrix_from_pure(ket0)
        expected = np.array([[1, 0], [0, 0]], dtype=complex)
        assert np.allclose(rho, expected, atol=1e-12)

    def test_pure_1_is_projector_1(self):
        rho = density_matrix_from_pure(ket1)
        expected = np.array([[0, 0], [0, 1]], dtype=complex)
        assert np.allclose(rho, expected, atol=1e-12)

    def test_hermitian(self):
        rho = density_matrix_from_pure([1, 1j])
        assert np.allclose(rho, rho.conj().T, atol=1e-12)

    def test_unit_trace(self):
        rho = density_matrix_from_pure([1, 2, 3])
        assert abs(np.trace(rho) - 1.0) < 1e-12

    def test_idempotent_pure_state(self):
        rho = density_matrix_from_pure(ket0)
        assert np.allclose(rho @ rho, rho, atol=1e-12)

    def test_normalisation_applied(self):
        rho_norm  = density_matrix_from_pure(ket0)
        rho_unnorm = density_matrix_from_pure(5.0 * ket0)
        assert np.allclose(rho_norm, rho_unnorm, atol=1e-12)

    def test_zero_vector_raises(self):
        with pytest.raises(ValueError, match="zero norm"):
            density_matrix_from_pure([0.0, 0.0])

    def test_superposition(self):
        psi = np.array([1, 1]) / np.sqrt(2)
        rho = density_matrix_from_pure(psi)
        assert np.allclose(rho, np.full((2, 2), 0.5), atol=1e-12)


# ─────────────────── TestPartialTrace ────────────────────────────────────

class TestPartialTrace:

    def test_bell_trace_B_is_maximally_mixed(self):
        rho = density_matrix_from_pure(bell)
        rho_A = partial_trace(rho, n_sites=2, keep_sites=[0])
        assert np.allclose(rho_A, I2 / 2, atol=1e-12)

    def test_bell_trace_A_is_maximally_mixed(self):
        rho = density_matrix_from_pure(bell)
        rho_B = partial_trace(rho, n_sites=2, keep_sites=[1])
        assert np.allclose(rho_B, I2 / 2, atol=1e-12)

    def test_product_state_trace_B_is_pure(self):
        psi = np.kron(ket0, ket1)
        rho = density_matrix_from_pure(psi)
        rho_A = partial_trace(rho, n_sites=2, keep_sites=[0])
        expected = density_matrix_from_pure(ket0)
        assert np.allclose(rho_A, expected, atol=1e-12)

    def test_product_state_trace_A_gives_second_qubit(self):
        psi = np.kron(ket0, ket1)
        rho = density_matrix_from_pure(psi)
        rho_B = partial_trace(rho, n_sites=2, keep_sites=[1])
        expected = density_matrix_from_pure(ket1)
        assert np.allclose(rho_B, expected, atol=1e-12)

    def test_trace_all_gives_scalar_one(self):
        rho = density_matrix_from_pure(bell)
        rho_scalar = partial_trace(rho, n_sites=2, keep_sites=[])
        assert np.allclose(rho_scalar, [[1.0]], atol=1e-12)

    def test_three_site_keep_first_two(self):
        psi = np.kron(np.kron(ket0, ket1), ket0)
        rho = density_matrix_from_pure(psi)
        rho_AB = partial_trace(rho, n_sites=3, keep_sites=[0, 1])
        expected = density_matrix_from_pure(np.kron(ket0, ket1))
        assert np.allclose(rho_AB, expected, atol=1e-12)

    def test_result_has_correct_shape(self):
        rho = density_matrix_from_pure(bell)
        rho_A = partial_trace(rho, n_sites=2, keep_sites=[0])
        assert rho_A.shape == (2, 2)

    def test_result_is_hermitian(self):
        rho = density_matrix_from_pure(bell)
        rho_A = partial_trace(rho, n_sites=2, keep_sites=[0])
        assert np.allclose(rho_A, rho_A.conj().T, atol=1e-12)

    def test_result_unit_trace(self):
        rho = density_matrix_from_pure(bell)
        rho_A = partial_trace(rho, n_sites=2, keep_sites=[0])
        assert abs(np.trace(rho_A) - 1.0) < 1e-12


# ─────────────────── TestCheckDensityMatrix ──────────────────────────────

class TestCheckDensityMatrix:

    def test_pure_state_all_passed(self):
        rho = density_matrix_from_pure(ket0)
        rep = check_density_matrix(rho)
        assert rep["all_passed"] is True

    def test_maximally_mixed_all_passed(self):
        rho = I2 / 2
        rep = check_density_matrix(rho)
        assert rep["all_passed"] is True

    def test_non_hermitian_fails(self):
        rho = np.array([[1, 0.5], [0, 0]], dtype=complex)  # not Hermitian
        rep = check_density_matrix(rho)
        assert not rep["hermitian"].passed

    def test_non_psd_fails(self):
        rho = np.array([[1.1, 0], [0, -0.1]], dtype=complex)  # negative eigenvalue
        rep = check_density_matrix(rho)
        assert not rep["psd"].passed

    def test_non_unit_trace_fails(self):
        rho = np.array([[0.6, 0], [0, 0.6]], dtype=complex)  # Tr = 1.2
        rep = check_density_matrix(rho)
        assert not rep["unit_trace"].passed

    def test_has_required_keys(self):
        rho = density_matrix_from_pure(ket0)
        rep = check_density_matrix(rho)
        for k in ("hermitian", "psd", "unit_trace", "all_passed"):
            assert k in rep

    def test_all_passed_is_bool(self):
        rho = density_matrix_from_pure(ket0)
        rep = check_density_matrix(rho)
        assert isinstance(rep["all_passed"], bool)

    def test_bell_state_dm_passes(self):
        rho = density_matrix_from_pure(bell)
        rep = check_density_matrix(rho)
        assert rep["all_passed"] is True


# ─────────────────── TestChoiMatrix ──────────────────────────────────────

class TestChoiMatrix:

    def test_identity_channel_shape(self):
        kraus = [I2]
        J = choi_matrix(kraus)
        assert J.shape == (4, 4)

    def test_identity_channel_is_maximally_entangled_projector(self):
        # For identity channel Φ(ρ) = ρ, J = |Φ+⟩⟨Φ+| * d
        kraus = [I2]
        J = choi_matrix(kraus)
        # J should be (|Φ+⟩⟨Φ+|) * d where d=2
        # |Φ+⟩⟨Φ+| = [[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]] / 2
        expected = np.array([[1, 0, 0, 1],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [1, 0, 0, 1]], dtype=complex)
        assert np.allclose(J, expected, atol=1e-12)

    def test_choi_is_psd_for_kraus_channel(self):
        # Any Kraus-given channel has PSD Choi matrix
        kraus = [np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]])]
        J = choi_matrix(kraus)
        evals = np.linalg.eigvalsh(J)
        assert np.all(evals >= -1e-10)

    def test_depolarizing_channel_trace_condition(self):
        # Fully depolarizing: K_mu = sigma_mu / 2
        kraus = [I2 / 2, X / 2, Y / 2, Z / 2]
        J = choi_matrix(kraus)
        # Block diagonal of J: Σ_i J[2i:2i+2, 2i:2i+2] should be I/2
        diag_sum = J[:2, :2] + J[2:, 2:]
        assert np.allclose(diag_sum, I2, atol=1e-12)


# ─────────────────── TestCheckKrausCompleteness ───────────────────────────

class TestCheckKrausCompleteness:

    def test_identity_channel_passes(self):
        r = check_kraus_completeness([I2])
        assert r.passed

    def test_amplitude_damping_passes(self):
        gamma = 0.3
        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
        r = check_kraus_completeness([K0, K1])
        assert r.passed

    def test_depolarizing_passes(self):
        kraus = [I2 / 2, X / 2, Y / 2, Z / 2]
        r = check_kraus_completeness(kraus)
        assert r.passed

    def test_incomplete_channel_fails(self):
        # Scale K_0 down so completeness fails
        r = check_kraus_completeness([0.5 * I2])
        assert not r.passed

    def test_defect_near_zero_for_complete_channel(self):
        r = check_kraus_completeness([I2])
        assert r.defect < 1e-12

    def test_property_name(self):
        r = check_kraus_completeness([I2])
        assert r.property_name == "kraus_completeness"

    def test_returns_structure_report(self):
        from htf.structure import StructureReport
        r = check_kraus_completeness([I2])
        assert isinstance(r, StructureReport)


# ─────────────────── TestLindbladSuperoperator ────────────────────────────

class TestLindbladSuperoperator:

    def test_shape_qubit(self):
        L_super = lindblad_superoperator(Z, [])
        assert L_super.shape == (4, 4)

    def test_no_dissipation_is_purely_imaginary_antisymmetric(self):
        # With no jump ops, L_super = -i(I⊗H - H.T⊗I)
        # For real symmetric H, this is skew-Hermitian
        H = np.diag([0.0, 1.0]).astype(complex)
        L_super = lindblad_superoperator(H, [])
        # L_super should be skew-Hermitian: L† = -L
        assert np.allclose(L_super + L_super.conj().T, np.zeros((4, 4)), atol=1e-12)

    def test_pure_decay_trace_preserving(self):
        # The trace of ρ should be preserved: Tr(dρ/dt) = 0
        # This means sum of each column of L_super over diagonal indices = 0
        gamma = 0.5
        L1 = np.sqrt(gamma) * np.array([[0, 1], [0, 0]])  # |0><1|
        H = np.zeros((2, 2))
        L_super = lindblad_superoperator(H, [L1])
        d = 2
        trace_vec = np.zeros(d * d)
        for i in range(d):
            trace_vec[i + i * d] = 1.0  # diagonal indices
        result = trace_vec @ L_super
        assert np.allclose(result, np.zeros(d * d), atol=1e-12)

    def test_zero_ham_zero_lindblad_is_zero(self):
        H = np.zeros((2, 2))
        L_super = lindblad_superoperator(H, [])
        assert np.allclose(L_super, np.zeros((4, 4)), atol=1e-12)


# ─────────────────── TestLindbladStep ─────────────────────────────────────

class TestLindbladStep:

    def test_shape(self):
        rho0 = density_matrix_from_pure(ket0)
        rho1 = lindblad_step(rho0, Z, [], dt=0.1)
        assert rho1.shape == (2, 2)

    def test_no_dissipation_is_unitary_evolution(self):
        # With no Lindblad ops, ρ(t) = e^{-iHt} ρ e^{iHt}
        H = 0.5 * Z.astype(complex)
        rho0 = density_matrix_from_pure(ket0)
        dt = 0.3
        rho_t = lindblad_step(rho0, H, [], dt=dt)
        # |0⟩ is an eigenstate of Z, so rho_t = rho0 (no mixing)
        assert np.allclose(rho_t, rho0, atol=1e-10)

    def test_trace_preserved(self):
        gamma = 0.5
        L1 = np.sqrt(gamma) * np.array([[0, 1], [0, 0]], dtype=complex)
        rho0 = density_matrix_from_pure(np.array([1, 1]) / np.sqrt(2))
        rho_t = lindblad_step(rho0, np.zeros((2, 2)), [L1], dt=0.5)
        assert abs(np.trace(rho_t) - 1.0) < 1e-10

    def test_dt_zero_gives_initial_state(self):
        rho0 = density_matrix_from_pure(ket0)
        rho_t = lindblad_step(rho0, Z, [], dt=0.0)
        assert np.allclose(rho_t, rho0, atol=1e-10)


# ─────────────────── TestSteadyState ─────────────────────────────────────

class TestSteadyState:

    def test_amplitude_damping_steady_state_is_ground(self):
        # L = |0><1|: decay to |0⟩ regardless of initial state
        L1 = np.array([[0, 1], [0, 0]], dtype=complex)  # |0><1|
        H = np.zeros((2, 2))
        rho_ss = steady_state(H, [L1])
        expected = density_matrix_from_pure(ket0)
        assert np.allclose(rho_ss, expected, atol=1e-8)

    def test_steady_state_unit_trace(self):
        L1 = np.array([[0, 1], [0, 0]], dtype=complex)
        rho_ss = steady_state(np.zeros((2, 2)), [L1])
        assert abs(np.trace(rho_ss) - 1.0) < 1e-8

    def test_steady_state_is_psd(self):
        L1 = np.array([[0, 1], [0, 0]], dtype=complex)
        rho_ss = steady_state(np.zeros((2, 2)), [L1])
        min_ev = float(np.linalg.eigvalsh(rho_ss).min())
        assert min_ev >= -1e-8

    def test_dephasing_steady_state_is_diagonal(self):
        # Pure dephasing: L = σ_z kills off-diagonal elements
        # H = ω/2 * σ_z, steady state = I/2 (maximally mixed)
        H = 0.5 * Z
        L1 = np.sqrt(2.0) * Z.astype(complex)  # strong dephasing
        rho_ss = steady_state(H, [L1])
        # Off-diagonal should be ~0 for strong dephasing
        assert abs(rho_ss[0, 1]) < 1e-6
        assert abs(rho_ss[1, 0]) < 1e-6

    def test_zero_lindblad_no_unique_ss(self):
        # With no Lindblad ops, every energy eigenstate is steady
        # lstsq will return something, but we don't assert a specific state
        H = Z.astype(complex)
        rho_ss = steady_state(H, [])
        # It should at least be unit-trace (from our constraint)
        assert abs(np.trace(rho_ss) - 1.0) < 1e-6

    def test_steady_state_satisfies_lindblad(self):
        # L(ρ_ss) should be ~0
        L1 = np.array([[0, 1], [0, 0]], dtype=complex)
        H = np.zeros((2, 2))
        rho_ss = steady_state(H, [L1])
        L_super = lindblad_superoperator(H, [L1])
        rho_vec = rho_ss.flatten(order='F')
        residual = np.linalg.norm(L_super @ rho_vec)
        assert residual < 1e-7
