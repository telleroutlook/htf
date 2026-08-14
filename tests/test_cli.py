import json

import pytest

from htf.cli import main


def test_version_prints_valid_json(capsys):
    main(["version"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "htf_version" in data


def test_version_htf_version_is_string(capsys):
    main(["version"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert isinstance(data["htf_version"], str)
    assert len(data["htf_version"]) > 0


def test_hello_prints_valid_json(capsys):
    main(["hello"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data is not None


def test_hello_output_has_mode_float(capsys):
    main(["hello"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["mode"] == "float"


def test_hello_result_close_to_one(capsys):
    main(["hello"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert abs(data["result"] - 1.0) < 1e-9


def test_hello_certificate_has_required_keys(capsys):
    main(["hello"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    for key in ("result", "mode", "error_bound", "notes"):
        assert key in data


def test_hello_error_bound_is_none(capsys):
    main(["hello"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["error_bound"] is None


def test_hello_notes_is_string(capsys):
    main(["hello"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert isinstance(data["notes"], str)
    assert len(data["notes"]) > 0


def test_no_args_raises_system_exit():
    with pytest.raises(SystemExit):
        main([])


def test_unknown_subcommand_raises_system_exit():
    with pytest.raises(SystemExit):
        main(["nonexistent"])


def test_version_output_has_no_extra_whitespace(capsys):
    main(["version"])
    captured = capsys.readouterr()
    # Output should be a single JSON line (parseable)
    data = json.loads(captured.out.strip())
    assert "htf_version" in data


def test_hello_stdout_only(capsys):
    main(["hello"])
    captured = capsys.readouterr()
    assert captured.err == ""


def test_version_stdout_only(capsys):
    main(["version"])
    captured = capsys.readouterr()
    assert captured.err == ""


# ── variational ───────────────────────────────────────────────────────────

class TestCLIVariational:

    def _run(self, capsys, extra=None):
        args = ["variational", "--n", "4", "--chi", "2", "--n-iter", "30", "--seed", "7"]
        if extra:
            args += extra
        main(args)
        return json.loads(capsys.readouterr().out)

    def test_valid_json(self, capsys):
        assert isinstance(self._run(capsys), dict)

    def test_has_certificate_key(self, capsys):
        assert "certificate" in self._run(capsys)

    def test_certificate_has_result(self, capsys):
        assert "result" in self._run(capsys)["certificate"]

    def test_certificate_has_error_bound(self, capsys):
        assert "error_bound" in self._run(capsys)["certificate"]

    def test_certificate_mode_certified(self, capsys):
        assert self._run(capsys)["certificate"]["mode"] == "certified"

    def test_n_sites_matches_arg(self, capsys):
        assert self._run(capsys)["n_sites"] == 4

    def test_chi_matches_arg(self, capsys):
        assert self._run(capsys)["chi"] == 2

    def test_model_label_present(self, capsys):
        assert "model" in self._run(capsys)

    def test_xx_model(self, capsys):
        data = self._run(capsys, ["--model", "xx"])
        assert "xx" in data["model"]

    def test_stdout_only(self, capsys):
        main(["variational", "--n", "4", "--chi", "2", "--n-iter", "20"])
        assert capsys.readouterr().err == ""


# ── gap ───────────────────────────────────────────────────────────────────

class TestCLIGap:

    def _run(self, capsys, extra=None):
        args = ["gap", "--n", "4", "--chi", "2", "--n-iter", "30", "--seed", "7"]
        if extra:
            args += extra
        main(args)
        return json.loads(capsys.readouterr().out)

    def test_valid_json(self, capsys):
        assert isinstance(self._run(capsys), dict)

    def test_has_gap_exact(self, capsys):
        data = self._run(capsys)
        assert "gap_exact" in data
        assert data["gap_exact"] > 0

    def test_has_E0_var(self, capsys):
        assert "E0_var" in self._run(capsys)

    def test_E1_var_above_E0_var(self, capsys):
        data = self._run(capsys)
        assert data["E1_var"] > data["E0_var"]

    def test_gap_var_is_difference(self, capsys):
        data = self._run(capsys)
        assert abs(data["gap_var"] - (data["E1_var"] - data["E0_var"])) < 1e-10

    def test_has_temple_lb(self, capsys):
        assert "temple_heuristic" in self._run(capsys)

    def test_has_gap_cert_with_keys(self, capsys):
        cert = self._run(capsys)["trial_energy_diff"]
        assert "result" in cert
        assert "error_bound" in cert
        assert cert["mode"] == "certified"

    def test_notes_contains_out(self, capsys):
        assert "[OUT]" in self._run(capsys)["notes"]

    def test_stdout_only(self, capsys):
        main(["gap", "--n", "4", "--chi", "2", "--n-iter", "20"])
        assert capsys.readouterr().err == ""


# ── difficulty ────────────────────────────────────────────────────────────

class TestCLIDifficulty:

    def _run(self, capsys, extra=None):
        args = ["difficulty", "--n", "4", "--n-iter", "30", "--seed", "7"]
        if extra:
            args += extra
        main(args)
        return json.loads(capsys.readouterr().out)

    def test_valid_json(self, capsys):
        assert isinstance(self._run(capsys), dict)

    def test_entanglement_profile_length(self, capsys):
        profile = self._run(capsys)["entanglement_profile"]
        assert isinstance(profile, list)
        assert len(profile) == 3  # n_sites - 1

    def test_has_max_entropy_nonneg(self, capsys):
        assert self._run(capsys)["max_entropy"] >= 0

    def test_likely_area_law_is_bool(self, capsys):
        assert isinstance(self._run(capsys)["likely_area_law"], bool)

    def test_n_sites_matches(self, capsys):
        assert self._run(capsys)["n_sites"] == 4

    def test_has_notes(self, capsys):
        assert isinstance(self._run(capsys)["notes"], str)

    def test_stdout_only(self, capsys):
        main(["difficulty", "--n", "4", "--n-iter", "20"])
        assert capsys.readouterr().err == ""


# ── os-check ──────────────────────────────────────────────────────────────

class TestCLIOsCheck:

    def _run(self, capsys, extra=None):
        args = ["os-check", "--n", "4"]
        if extra:
            args += extra
        main(args)
        return json.loads(capsys.readouterr().out)

    def test_valid_json(self, capsys):
        assert isinstance(self._run(capsys), dict)

    def test_all_passed_true_for_ising(self, capsys):
        assert self._run(capsys)["all_passed"] is True

    def test_transfer_positivity_passed(self, capsys):
        assert self._run(capsys)["transfer_positivity"]["passed"] is True

    def test_reflection_symmetry_passed(self, capsys):
        assert self._run(capsys)["reflection_symmetry"]["passed"] is True

    def test_os_gram_positivity_passed(self, capsys):
        assert self._run(capsys)["os_gram_positivity"]["passed"] is True

    def test_notes_contains_out(self, capsys):
        assert "[OUT]" in self._run(capsys)["notes"]

    def test_n_sites_matches(self, capsys):
        assert self._run(capsys)["n_sites"] == 4

    def test_default_beta_is_one(self, capsys):
        assert self._run(capsys)["beta"] == 1.0

    def test_custom_beta(self, capsys):
        data = self._run(capsys, ["--beta", "2.0"])
        assert data["beta"] == 2.0
        assert data["all_passed"] is True

    def test_xx_model_passes(self, capsys):
        assert self._run(capsys, ["--model", "xx"])["all_passed"] is True

    def test_defect_key_present(self, capsys):
        data = self._run(capsys)
        assert "defect" in data["transfer_positivity"]
        assert "defect" in data["reflection_symmetry"]

    def test_stdout_only(self, capsys):
        main(["os-check", "--n", "4"])
        assert capsys.readouterr().err == ""


# ── benchmark ─────────────────────────────────────────────────────────────

class TestCLIBenchmark:

    def _run(self, capsys, extra=None):
        args = ["benchmark", "--n", "4", "--chi", "2", "--n-iter", "20", "--seed", "0"]
        if extra:
            args += extra
        main(args)
        return json.loads(capsys.readouterr().out)

    def test_valid_json(self, capsys):
        assert isinstance(self._run(capsys), dict)

    def test_has_htf_version(self, capsys):
        assert "htf_version" in self._run(capsys)

    def test_has_results_list(self, capsys):
        data = self._run(capsys)
        assert isinstance(data["results"], list)
        assert len(data["results"]) == 2  # ising + xx by default

    def test_result_has_required_keys(self, capsys):
        result = self._run(capsys)["results"][0]
        for k in ("model", "E0_var", "gap_exact", "gap_cert_result",
                  "os_passed", "max_entropy", "likely_area_law"):
            assert k in result

    def test_single_model_flag(self, capsys):
        data = self._run(capsys, ["--models", "ising"])
        assert len(data["results"]) == 1
        assert data["results"][0]["model"] == "ising"

    def test_multiple_models_flag(self, capsys):
        data = self._run(capsys, ["--models", "ising", "xx"])
        models = {r["model"] for r in data["results"]}
        assert models == {"ising", "xx"}

    def test_n_sites_matches(self, capsys):
        assert self._run(capsys)["n_sites"] == 4

    def test_chi_matches(self, capsys):
        assert self._run(capsys)["chi"] == 2

    def test_os_passed_for_ising(self, capsys):
        data = self._run(capsys, ["--models", "ising"])
        assert data["results"][0]["os_passed"] is True

    def test_stdout_only(self, capsys):
        main(["benchmark", "--n", "4", "--n-iter", "15"])
        assert capsys.readouterr().err == ""


# ─────────────────────── TestLanczosCmd ──────────────────────────────────

class TestLanczosCmd:

    def _run(self, capsys, extra=None):
        args = ["lanczos", "--n", "4", "--k", "10"]
        main(args + (extra or []))
        return json.loads(capsys.readouterr().out)

    def test_returns_valid_json(self, capsys):
        data = self._run(capsys)
        assert isinstance(data, dict)

    def test_has_required_keys(self, capsys):
        data = self._run(capsys)
        for key in ("E0_upper", "E0_lower_heuristic", "E1_ritz", "temple_condition_met"):
            assert key in data

    def test_e0_upper_less_than_e1(self, capsys):
        data = self._run(capsys)
        assert data["E0_upper"] < data["E1_ritz"]

    def test_model_xx(self, capsys):
        data = self._run(capsys, ["--model", "xx"])
        assert "xx" in data["model"]

    def test_no_stderr(self, capsys):
        self._run(capsys)
        assert capsys.readouterr().err == ""


# ─────────────────────── TestQasmSimCmd ──────────────────────────────────

class TestQasmSimCmd:

    _BELL_QASM = (
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[2];\n"
        "h q[0];\ncx q[0], q[1];\n"
    )

    def _run_file(self, capsys, tmp_path, content=None, extra=None):
        p = tmp_path / "circuit.qasm"
        p.write_text(content or self._BELL_QASM)
        main(["qasm-sim", "--file", str(p)] + (extra or []))
        return json.loads(capsys.readouterr().out)

    def test_returns_valid_json(self, capsys, tmp_path):
        data = self._run_file(capsys, tmp_path)
        assert isinstance(data, dict)

    def test_has_unitary_keys(self, capsys, tmp_path):
        data = self._run_file(capsys, tmp_path)
        assert "unitary_real" in data
        assert "unitary_imag" in data

    def test_n_qubits_inferred(self, capsys, tmp_path):
        data = self._run_file(capsys, tmp_path)
        assert data["n_qubits"] == 2

    def test_unitary_is_correct_size(self, capsys, tmp_path):
        data = self._run_file(capsys, tmp_path)
        assert len(data["unitary_real"]) == 4
        assert len(data["unitary_real"][0]) == 4

    def test_unitary_is_unitary(self, capsys, tmp_path):
        import numpy as np
        data = self._run_file(capsys, tmp_path)
        U = np.array(data["unitary_real"]) + 1j * np.array(data["unitary_imag"])
        assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-10)


# ─────────────────────── TestZxSimplifyCmd ───────────────────────────────

class TestZxSimplifyCmd:

    _BELL_QASM = (
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[2];\n"
        "h q[0];\ncx q[0], q[1];\n"
    )

    def _run_file(self, capsys, tmp_path, content=None):
        p = tmp_path / "circuit.qasm"
        p.write_text(content or self._BELL_QASM)
        main(["zx-simplify", "--file", str(p)])
        return json.loads(capsys.readouterr().out)

    def test_returns_valid_json(self, capsys, tmp_path):
        data = self._run_file(capsys, tmp_path)
        assert isinstance(data, dict)

    def test_has_stats_keys(self, capsys, tmp_path):
        data = self._run_file(capsys, tmp_path)
        for key in ("nodes_before", "nodes_after", "rewrites_total", "rule_counts"):
            assert key in data

    def test_nodes_after_le_before(self, capsys, tmp_path):
        data = self._run_file(capsys, tmp_path)
        assert data["nodes_after"] <= data["nodes_before"]

    def test_n_qubits_inferred(self, capsys, tmp_path):
        data = self._run_file(capsys, tmp_path)
        assert data["n_qubits"] == 2


# ─────────────────────── TestInverseCmd ──────────────────────────────────

class TestInverseCmd:

    def _run(self, capsys, extra=None):
        args = ["inverse", "--n", "4", "--target-e0", "-1.5",
                "--n-restarts", "2", "--seed", "0"]
        main(args + (extra or []))
        return json.loads(capsys.readouterr().out)

    def test_returns_valid_json(self, capsys):
        data = self._run(capsys)
        assert isinstance(data, dict)

    def test_has_required_keys(self, capsys):
        data = self._run(capsys)
        for key in ("E0_achieved", "residual", "params_opt", "converged"):
            assert key in data

    def test_target_e0_matches(self, capsys):
        data = self._run(capsys)
        assert abs(data["target_e0"] - (-1.5)) < 1e-12

    def test_model_xx(self, capsys):
        data = self._run(capsys, ["--model", "xx"])
        assert "xx" in data["model"]


# ─────────────────────── TestLeanExportCmd ───────────────────────────────

class TestLeanExportCmd:

    def _run(self, capsys, tmp_path, extra=None):
        out = tmp_path / "out.lean"
        args = ["lean-export", "--n", "4", "--output", str(out)]
        main(args + (extra or []))
        return json.loads(capsys.readouterr().out), out

    def test_returns_valid_json(self, capsys, tmp_path):
        data, _ = self._run(capsys, tmp_path)
        assert isinstance(data, dict)

    def test_output_file_created(self, capsys, tmp_path):
        _, out = self._run(capsys, tmp_path)
        assert out.exists()

    def test_lean_file_has_namespace(self, capsys, tmp_path):
        _, out = self._run(capsys, tmp_path)
        content = out.read_text()
        assert "namespace HTF" in content

    def test_lean_file_has_sorry(self, capsys, tmp_path):
        _, out = self._run(capsys, tmp_path)
        content = out.read_text()
        assert "sorry" in content

    def test_json_has_e0_upper(self, capsys, tmp_path):
        data, _ = self._run(capsys, tmp_path)
        assert "E0_upper" in data
        assert isinstance(data["E0_upper"], float)


# ── C6: claim registry CLI command ───────────────────────────────────────────

class TestRegistryCommand:
    """htf registry must print the claim registry as valid JSON."""

    def test_prints_valid_json(self, capsys):
        main(["registry"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert isinstance(data, dict)

    def test_all_known_claims_present(self, capsys):
        main(["registry"])
        out = capsys.readouterr().out
        data = json.loads(out)
        for cid in ("rayleigh", "variational", "gap", "lanczos", "benchmark", "difficulty"):
            assert cid in data, f"claim_id {cid!r} missing from registry output"

    def test_each_claim_has_required_fields(self, capsys):
        main(["registry"])
        data = json.loads(capsys.readouterr().out)
        required = {"title", "assurance", "evidence_tier", "limitations"}
        for cid, info in data.items():
            missing = required - set(info)
            assert not missing, f"claim {cid!r} missing fields: {missing}"

    def test_assurance_values_are_valid(self, capsys):
        main(["registry"])
        data = json.loads(capsys.readouterr().out)
        valid = {"rigorous", "heuristic", "reproducible"}
        for cid, info in data.items():
            assert info["assurance"] in valid, (
                f"claim {cid!r} has invalid assurance {info['assurance']!r}"
            )

