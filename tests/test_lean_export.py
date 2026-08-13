"""Tests for htf/lean_export.py — Lean 4 proof-assistant export (§4-L)."""

import pytest

from htf.certificate import Certificate
from htf.lean_export import (
    LeanExporter,
    certificate_to_lean,
    diagram_to_lean_type,
    export_lean,
    gap_report_to_lean,
    structure_report_to_lean,
)
from htf.structure import StructureReport
from htf.topology import Box, Id, Wire

# ─────────── certificate_to_lean ─────────────────────────────────────────

class TestCertificateToLean:

    def test_returns_string(self):
        cert = Certificate(result=-3.5, mode="float", error_bound=None)
        s = certificate_to_lean(cert, "e0")
        assert isinstance(s, str)

    def test_name_appears(self):
        cert = Certificate(result=-1.0, mode="certified", error_bound=1e-10)
        s = certificate_to_lean(cert, "my_bound")
        assert "my_bound" in s

    def test_value_appears(self):
        cert = Certificate(result=-3.141592, mode="float", error_bound=None)
        s = certificate_to_lean(cert, "pi_bound")
        assert "-3.141592" in s

    def test_contains_sorry(self):
        cert = Certificate(result=0.0, mode="float", error_bound=None)
        s = certificate_to_lean(cert, "zero")
        assert "sorry" in s

    def test_contains_theorem_keyword(self):
        cert = Certificate(result=1.0, mode="certified", error_bound=0.0)
        s = certificate_to_lean(cert, "one")
        assert "theorem" in s

    def test_contains_def_keyword(self):
        cert = Certificate(result=1.0, mode="certified", error_bound=0.0)
        s = certificate_to_lean(cert, "one")
        assert "def" in s

    def test_certified_mode_includes_rad(self):
        cert = Certificate(result=2.0, mode="certified", error_bound=0.001)
        s = certificate_to_lean(cert, "c")
        assert "0.001" in s

    def test_no_error_bound_uses_zero(self):
        cert = Certificate(result=0.5, mode="float", error_bound=None)
        s = certificate_to_lean(cert, "c")
        assert "0.0" in s


# ─────────── gap_report_to_lean ──────────────────────────────────────────

class TestGapReportToLean:

    def _report(self):
        return {
            "E0_exact": -4.0,
            "gap_exact": 1.5,
            "E0_variational": -3.9,
            "temple_lower_bound": -4.05,
            "temple_condition_met": True,
        }

    def test_returns_string(self):
        s = gap_report_to_lean(self._report(), "ising")
        assert isinstance(s, str)

    def test_name_appears(self):
        s = gap_report_to_lean(self._report(), "my_model")
        assert "my_model" in s

    def test_contains_variational_theorem(self):
        s = gap_report_to_lean(self._report(), "m")
        assert "variational_upper" in s

    def test_contains_temple_theorem_when_met(self):
        s = gap_report_to_lean(self._report(), "m")
        assert "temple_lower" in s

    def test_no_temple_theorem_when_not_met(self):
        r = self._report()
        r["temple_condition_met"] = False
        s = gap_report_to_lean(r, "m")
        assert "temple_lower" not in s

    def test_contains_sorry(self):
        s = gap_report_to_lean(self._report(), "m")
        assert "sorry" in s

    def test_e0_value_in_output(self):
        s = gap_report_to_lean(self._report(), "m")
        assert "-4.0" in s or "-4" in s


# ─────────── structure_report_to_lean ────────────────────────────────────

class TestStructureReportToLean:

    def _report(self, passed=True):
        return StructureReport(
            property_name="isometry",
            passed=passed,
            defect=1e-15,
            tolerance=1e-10,
        )

    def test_returns_string(self):
        s = structure_report_to_lean(self._report(), "iso")
        assert isinstance(s, str)

    def test_name_appears(self):
        s = structure_report_to_lean(self._report(), "my_prop")
        assert "my_prop" in s

    def test_true_when_passed(self):
        s = structure_report_to_lean(self._report(passed=True), "p")
        assert "True" in s

    def test_false_when_not_passed(self):
        s = structure_report_to_lean(self._report(passed=False), "p")
        assert "False" in s

    def test_contains_native_decide(self):
        s = structure_report_to_lean(self._report(), "p")
        assert "native_decide" in s

    def test_defect_value_appears(self):
        s = structure_report_to_lean(self._report(), "p")
        assert "1e-15" in s or "1.0e-15" in s or "1e" in s


# ─────────── diagram_to_lean_type ────────────────────────────────────────

class TestDiagramToLeanType:

    def test_returns_string(self):
        w = Wire("q", 2)
        d = Box("H", (w,), (w,))
        s = diagram_to_lean_type(d, "h_gate")
        assert isinstance(s, str)

    def test_name_appears(self):
        w = Wire("q", 2)
        d = Box("H", (w,), (w,))
        s = diagram_to_lean_type(d, "hadamard")
        assert "hadamard" in s

    def test_dim_appears(self):
        w = Wire("q", 2)
        d = Box("H", (w,), (w,))
        s = diagram_to_lean_type(d, "h")
        assert "2" in s

    def test_id_diagram(self):
        w = Wire("x", 3)
        d = Id((w,))
        s = diagram_to_lean_type(d, "identity")
        assert isinstance(s, str)

    def test_empty_type_uses_unit(self):
        d = Id(())
        s = diagram_to_lean_type(d, "empty")
        assert "Unit" in s


# ─────────── LeanExporter ────────────────────────────────────────────────

class TestLeanExporter:

    def test_source_contains_header(self):
        exp = LeanExporter()
        src = exp.source()
        assert "HTF" in src
        assert "namespace HTF" in src

    def test_source_contains_footer(self):
        exp = LeanExporter()
        src = exp.source()
        assert "end HTF" in src

    def test_source_contains_base_defs_by_default(self):
        exp = LeanExporter()
        src = exp.source()
        assert "CertInterval" in src

    def test_base_defs_can_be_disabled(self):
        exp = LeanExporter(include_base_defs=False)
        src = exp.source()
        assert "CertInterval" not in src

    def test_add_snippet(self):
        exp = LeanExporter()
        exp.add_snippet("-- my custom comment")
        src = exp.source()
        assert "my custom comment" in src

    def test_add_certificate(self):
        exp = LeanExporter()
        cert = Certificate(result=-2.5, mode="float", error_bound=None)
        exp.add_certificate(cert, "e0")
        src = exp.source()
        assert "e0" in src
        assert "-2.5" in src

    def test_add_gap_report(self):
        exp = LeanExporter()
        report = {"E0_exact": -1.0, "gap_exact": 0.5, "E0_variational": -0.9,
                  "temple_lower_bound": -1.1, "temple_condition_met": True}
        exp.add_gap_report(report, "model")
        src = exp.source()
        assert "model" in src

    def test_write_creates_file(self, tmp_path):
        exp = LeanExporter()
        exp.add_snippet("-- hello Lean")
        path = tmp_path / "test.lean"
        exp.write(path)
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "hello Lean" in content

    def test_preamble_included(self):
        exp = LeanExporter(preamble="Generated for the TFIM model.")
        src = exp.source()
        assert "TFIM" in src

    def test_multiple_snippets_ordered(self):
        exp = LeanExporter()
        exp.add_snippet("-- FIRST")
        exp.add_snippet("-- SECOND")
        src = exp.source()
        assert src.index("FIRST") < src.index("SECOND")


# ─────────── export_lean ─────────────────────────────────────────────────

class TestExportLean:

    def test_creates_file(self, tmp_path):
        cert = Certificate(result=-3.0, mode="float", error_bound=None)
        path = tmp_path / "out.lean"
        export_lean([("certificate", cert, "e0")], path)
        assert path.exists()

    def test_returns_source_string(self, tmp_path):
        cert = Certificate(result=0.0, mode="float", error_bound=None)
        path = tmp_path / "out.lean"
        src = export_lean([("certificate", cert, "zero")], path)
        assert isinstance(src, str)
        assert "zero" in src

    def test_unknown_kind_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown item kind"):
            export_lean([("bad_kind", None, "x")], tmp_path / "x.lean")

    def test_raw_snippet_kind(self, tmp_path):
        path = tmp_path / "out.lean"
        src = export_lean([("snippet", "-- raw snippet text", "n/a")], path)
        assert "raw snippet text" in src

    def test_lean_file_has_utf8_encoding(self, tmp_path):
        cert = Certificate(result=1.0, mode="certified", error_bound=1e-14,
                           notes="φ≈1.618")
        path = tmp_path / "utf8.lean"
        export_lean([("certificate", cert, "phi")], path)
        content = path.read_text(encoding="utf-8")
        assert "φ" in content or "phi" in content
