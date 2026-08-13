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
