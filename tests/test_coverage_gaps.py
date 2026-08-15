"""Coverage-gap tests for multiple htf modules.

Targets previously-uncovered lines in:
  inverse.py (71), gap.py (141-142), lanczos.py (243-244),
  variational.py (119-120), mps.py (165, 314), symmetric.py (65, 250),
  corpus.py (89-91), difficulty.py (80), mpo.py (312, 317),
  tebd.py (467), zx.py (77-78, 217-222, 237-240, 249-252, 321-324, 666,
  673, 733), cli.py (375-388).
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# inverse.py line 71 — unknown model raises in ham()
# ──────────────────────────────────────────────────────────────────────────────


class TestParametricHamUnknownModel:
    def test_ham_unknown_model_raises(self):
        from htf.inverse import ParametricHam

        h = ParametricHam("ising", 4)
        h.model = "bad_model"
        with pytest.raises(ValueError, match="Unknown model"):
            h.ham([1.0, 0.5])


# ──────────────────────────────────────────────────────────────────────────────
# gap.py lines 141-142 — trial_energy_difference raises when flint absent
# ──────────────────────────────────────────────────────────────────────────────


class TestTrialEnergyDifferenceNoFlint:
    def test_raises_import_error_without_flint(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "flint", None)
        from htf.gap import trial_energy_difference

        H = np.diag([0.0, 1.0, 2.0, 3.0])
        psi0 = np.array([1.0, 0.0, 0.0, 0.0])
        psi1 = np.array([0.0, 1.0, 0.0, 0.0])
        with pytest.raises(ImportError, match="python-flint"):
            trial_energy_difference(H, psi0, psi1)


# ──────────────────────────────────────────────────────────────────────────────
# lanczos.py lines 243-244 — temple_lanczos float-mode fallback when flint absent
# ──────────────────────────────────────────────────────────────────────────────


class TestTemplateLanczosNoFlint:
    def test_returns_float_cert_without_flint(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "flint", None)
        from htf.lanczos import temple_lanczos

        H = np.diag([0.0, 1.0, 2.0, 3.0])
        bounds = temple_lanczos(H, k=4, seed=0)
        assert math.isfinite(bounds.E0_upper)
        assert math.isfinite(bounds.E0_upper_error)


# ──────────────────────────────────────────────────────────────────────────────
# variational.py lines 119-120 — variational_bound raises when flint absent
# ──────────────────────────────────────────────────────────────────────────────


class TestVariationalBoundNoFlint:
    def test_raises_import_error_without_flint(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "flint", None)
        from htf.mera import random_mera
        from htf.variational import transverse_ising_ham, variational_bound

        H = transverse_ising_ham(4, J=1.0, h=0.5)
        mera = random_mera(4, chi=2, seed=0)
        with pytest.raises(ImportError, match="python-flint"):
            variational_bound(H, mera)


# ──────────────────────────────────────────────────────────────────────────────
# mps.py line 165 — mps_normalise raises for zero MPS
# mps.py line 314 — mps_apply_gate raises for 3-site gate
# ──────────────────────────────────────────────────────────────────────────────


class TestMPSEdgeCases:
    def _zero_mps(self):
        from htf.mps import MPS

        t = np.zeros((1, 2, 1))
        return MPS([t.copy(), t.copy()])

    def _unit_mps(self):
        from htf.mps import MPS

        t0 = np.zeros((1, 2, 1))
        t0[0, 0, 0] = 1.0
        t1 = np.zeros((1, 2, 1))
        t1[0, 0, 0] = 1.0
        return MPS([t0, t1])

    def test_normalise_zero_mps_raises(self):
        from htf.mps import mps_normalise

        with pytest.raises(ValueError, match="zero"):
            mps_normalise(self._zero_mps())

    def test_apply_gate_three_sites_raises(self):
        from htf.mps import MPS, mps_apply_gate

        t = np.zeros((1, 2, 1))
        t[0, 0, 0] = 1.0
        mps = MPS([t.copy(), t.copy(), t.copy()])
        gate = np.eye(8, dtype=float).reshape(2, 2, 2, 2, 2, 2)
        with pytest.raises(ValueError, match="supported"):
            mps_apply_gate(mps, gate, sites=[0, 1, 2])


# ──────────────────────────────────────────────────────────────────────────────
# symmetric.py line 65 — ChargedBasis.__repr__
# symmetric.py line 250 — _flat_charges([]) → [0]
# ──────────────────────────────────────────────────────────────────────────────


class TestSymmetricEdgeCases:
    def test_charged_basis_repr(self):
        from htf.symmetric import ChargedBasis

        b = ChargedBasis([(1, 1), (1, -1)])
        s = repr(b)
        assert "ChargedBasis" in s
        assert "1" in s

    def test_flat_charges_empty_returns_zero(self):
        from htf.symmetric import _flat_charges

        result = _flat_charges([])
        np.testing.assert_array_equal(result, np.array([0], dtype=int))


# ──────────────────────────────────────────────────────────────────────────────
# corpus.py lines 89-91 — CorpusCase.run() catches exception in certificate()
# ──────────────────────────────────────────────────────────────────────────────


class TestCorpusCaseExceptionPath:
    def test_run_returns_failed_result_on_exception(self):
        from htf.corpus import CorpusCase, CorpusCaseResult

        class _FailCase(CorpusCase):
            def certificate(self):
                raise RuntimeError("forced test failure")

        H = np.diag([0.0, 1.0])
        psi = np.array([1.0, 0.0])
        case = _FailCase(
            name="test-fail",
            description="failure test",
            tags=["test"],
            H=H,
            psi=psi,
            expected_E0=0.0,
            expected_upper=0.01,
        )
        result = case.run()
        assert isinstance(result, CorpusCaseResult)
        assert result.passed is False
        assert "forced test failure" in (result.error or "")
        assert result.backend == "error"


# ──────────────────────────────────────────────────────────────────────────────
# difficulty.py line 80 — entanglement_spectrum invalid cut
# ──────────────────────────────────────────────────────────────────────────────


class TestEntanglementSpectrumInvalidCut:
    def test_cut_zero_raises(self):
        from htf.difficulty import entanglement_spectrum

        psi = np.array([1.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="cut"):
            entanglement_spectrum(psi, n_sites=2, cut=0)

    def test_cut_equals_n_sites_raises(self):
        from htf.difficulty import entanglement_spectrum

        psi = np.array([1.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="cut"):
            entanglement_spectrum(psi, n_sites=2, cut=2)


# ──────────────────────────────────────────────────────────────────────────────
# mpo.py lines 312, 317 — MPOScalingReport.summary() branches
# ──────────────────────────────────────────────────────────────────────────────


class TestMPOScalingReportSummary:
    def _make_point(self, chi, energy):
        from htf.mpo import MPOChiPoint

        return MPOChiPoint(chi=chi, energy=energy, n_seeds_used=1, best_seed=0)

    def test_summary_with_extrapolation(self):
        from htf.mpo import MPOScalingReport

        report = MPOScalingReport(
            chi_points=[self._make_point(2, -3.0), self._make_point(4, -3.1)],
            E_extrapolated=-3.15,
            E_extrap_stderr=0.01,
            fit_exponent=1.5,
            notes="",
        )
        s = report.summary()
        assert "Power-law" in s
        assert "-3.15" in s

    def test_summary_with_notes(self):
        from htf.mpo import MPOScalingReport

        report = MPOScalingReport(
            chi_points=[self._make_point(2, -3.0)],
            notes="power-law fit failed: test",
        )
        s = report.summary()
        assert "Note:" in s
        assert "power-law fit failed" in s


# ──────────────────────────────────────────────────────────────────────────────
# tebd.py line 467 — _nn_energy returns 0.0 for zero-norm MPS
# ──────────────────────────────────────────────────────────────────────────────


class TestNNEnergyZeroNorm:
    def test_returns_zero_for_zero_mps(self):
        from htf.mps import MPS
        from htf.tebd import _nn_energy, tfim_bonds

        t = np.zeros((1, 2, 1))
        zero_mps = MPS([t.copy(), t.copy()])
        h_terms = tfim_bonds(2, J=1.0, h=0.5)
        result = _nn_energy(zero_mps, h_terms)
        assert result == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# zx.py — ZXNode repr, Y/Sdg/Tdg/unknown gates, H-box error, zero-leg tensor,
#          invalid input-node edge count in zx_to_matrix
# ──────────────────────────────────────────────────────────────────────────────


class TestZXNodeRepr:
    """Line 77-78: ZXNode.__repr__ with nonzero phase."""

    def test_repr_with_phase(self):
        from htf.zx import ZXNode, ZXNodeType

        node = ZXNode(node_id=1, kind=ZXNodeType.Z, phase=0.5)
        s = repr(node)
        assert "ZXNode" in s
        assert "0.5" in s

    def test_repr_zero_phase_no_phi(self):
        from htf.zx import ZXNode, ZXNodeType

        node = ZXNode(node_id=2, kind=ZXNodeType.X, phase=0.0)
        s = repr(node)
        assert "φ" not in s


class TestZXFromCircuitGates:
    """Lines 217-252: Y, Sdg, Tdg gates and unknown gate fallback."""

    def test_y_gate_adds_two_nodes(self):
        from htf.qasm import Gate
        from htf.zx import ZXNodeType, zx_from_circuit

        g = zx_from_circuit([Gate("y", [0])], n_qubits=1)
        kinds = [n.kind for n in g.nodes.values()]
        assert ZXNodeType.Z in kinds
        assert ZXNodeType.X in kinds

    def test_sdg_gate_produces_graph(self):
        from htf.qasm import Gate
        from htf.zx import ZXNodeType, zx_from_circuit

        g = zx_from_circuit([Gate("sdg", [0])], n_qubits=1)
        z_nodes = [n for n in g.nodes.values() if n.kind == ZXNodeType.Z]
        assert any(abs(n.phase - (-math.pi / 2)) < 1e-10 for n in z_nodes)

    def test_tdg_gate_produces_graph(self):
        from htf.qasm import Gate
        from htf.zx import ZXNodeType, zx_from_circuit

        g = zx_from_circuit([Gate("tdg", [0])], n_qubits=1)
        z_nodes = [n for n in g.nodes.values() if n.kind == ZXNodeType.Z]
        assert any(abs(n.phase - (-math.pi / 4)) < 1e-10 for n in z_nodes)

    def test_unknown_gate_adds_opaque_z_node(self):
        from htf.qasm import Gate
        from htf.zx import zx_from_circuit

        g = zx_from_circuit([Gate("custom_xyz", [0])], n_qubits=1)
        labels = [n.label for n in g.nodes.values()]
        assert any("custom_xyz" in lbl for lbl in labels)


class TestSpiderTensorEdgeCases:
    """Lines 666, 673: H-box wrong legs raises; zero-leg Z tensor."""

    def test_h_box_wrong_legs_raises(self):
        from htf.zx import ZXNodeType, _spider_tensor

        with pytest.raises(ValueError, match="2 legs"):
            _spider_tensor(ZXNodeType.H, 0.0, 3)

    def test_z_spider_zero_legs_returns_zero_tensor(self):
        from htf.zx import ZXNodeType, _spider_tensor

        T = _spider_tensor(ZXNodeType.Z, 0.0, 0)
        assert T.shape == ()
        assert T == 0.0


class TestZXToMatrixInvalidInputEdges:
    """Line 733: input node with extra edge raises ValueError."""

    def test_extra_input_edge_raises(self):
        from htf.qasm import Gate
        from htf.zx import zx_from_circuit, zx_to_matrix

        g = zx_from_circuit([Gate("h", [0])], n_qubits=1)
        # Add a second edge from the input node — makes _single_bond fail
        g.edges.append((g.inputs[0], g.outputs[0]))
        with pytest.raises(ValueError):
            zx_to_matrix(g)


# ──────────────────────────────────────────────────────────────────────────────
# zx.py simplification continue branches
# Lines 384 (spider_fusion), 427 (identity_removal), 459/473 (hadamard_cancel),
# 579/582 (pi_copy)
# ──────────────────────────────────────────────────────────────────────────────


class TestZXSimplificationContinues:
    """Directly call simplification rules with graphs that hit guard clauses."""

    def test_spider_fusion_skips_non_spider_nodes(self):
        # Line 384: continue when u.kind not in (Z, X)
        from htf.zx import ZXGraph, ZXNodeType, spider_fusion

        g = ZXGraph()
        inp = g.add_node(ZXNodeType.INPUT)
        out = g.add_node(ZXNodeType.OUTPUT)
        z = g.add_node(ZXNodeType.Z, phase=0.0)
        g.add_edge(inp, z)
        g.add_edge(z, out)
        g.inputs.append(inp)
        g.outputs.append(out)
        # INPUT/OUTPUT nodes → continue at line 384; Z node: no same-kind neighbor
        result = spider_fusion(g)
        assert result == 0

    def test_identity_removal_skips_nonzero_phase_node(self):
        # Line 427: continue when phase != 0 (Z node, degree 2, phase=π)
        from htf.zx import ZXGraph, ZXNodeType, identity_removal

        g = ZXGraph()
        a = g.add_node(ZXNodeType.INPUT)
        b = g.add_node(ZXNodeType.OUTPUT)
        z = g.add_node(ZXNodeType.Z, phase=math.pi)
        g.add_edge(a, z)
        g.add_edge(z, b)
        result = identity_removal(g)
        assert result == 0
        assert z in g.nodes  # node was not removed

    def test_hadamard_cancel_skips_non_h_node(self):
        # Line 459: continue when h1.kind != H (a Z node is processed first)
        from htf.zx import ZXGraph, ZXNodeType, hadamard_cancel

        g = ZXGraph()
        a = g.add_node(ZXNodeType.Z)
        b = g.add_node(ZXNodeType.Z)
        g.add_edge(a, b)
        result = hadamard_cancel(g)
        assert result == 0

    def test_hadamard_cancel_skips_h_connected_to_non_h(self):
        # Line 473: continue when H's neighbour is not H
        from htf.zx import ZXGraph, ZXNodeType, hadamard_cancel

        g = ZXGraph()
        h = g.add_node(ZXNodeType.H)
        z = g.add_node(ZXNodeType.Z)
        g.add_edge(h, z)
        result = hadamard_cancel(g)
        assert result == 0

    def test_pi_copy_skips_z_pi_with_multiple_neighbours(self):
        # Line 579: continue when Z(π) has more than 1 neighbour
        from htf.zx import ZXGraph, ZXNodeType, pi_copy

        g = ZXGraph()
        z = g.add_node(ZXNodeType.Z, phase=math.pi)
        x1 = g.add_node(ZXNodeType.X, phase=0.0)
        x2 = g.add_node(ZXNodeType.X, phase=0.0)
        g.add_edge(z, x1)
        g.add_edge(z, x2)
        result = pi_copy(g)
        assert result == 0

    def test_pi_copy_skips_z_pi_connected_to_non_x(self):
        # Line 582: continue when Z(π)'s sole neighbour is not X(0)
        from htf.zx import ZXGraph, ZXNodeType, pi_copy

        g = ZXGraph()
        z = g.add_node(ZXNodeType.Z, phase=math.pi)
        nb = g.add_node(ZXNodeType.Z, phase=0.0)  # Z, not X
        g.add_edge(z, nb)
        result = pi_copy(g)
        assert result == 0


# ──────────────────────────────────────────────────────────────────────────────
# cli.py lines 375-388 — rayleigh subcommand produces JSON output
# ──────────────────────────────────────────────────────────────────────────────


class TestCLIRayleighSubcommand:
    def test_rayleigh_produces_json(self, capsys):
        import json

        from htf.cli import main

        main(["rayleigh", "--model", "ising", "--n", "4", "--seed", "0"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "interval" in data
        assert data["interval"]["upper"] > data["interval"]["lower"]

    def test_rayleigh_full_flag(self, capsys):
        import json

        from htf.cli import main

        main(["rayleigh", "--model", "xx", "--n", "4", "--seed", "1", "--full"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "schema_version" in data
