"""Certificate hygiene lint checks.

These tests catch patterns that produce false or misleading claims in
certificate assumptions text.  They mirror the finding from the HTF-06
Gate-A review (2026-08-15): using binary64 float arithmetic inside
assumption strings can produce statements like "<psi|psi> = 0 > 0" when
the true norm^2 underflows below the float64 minimum.

Rule: Assumption strings in certificate-generating code must be logically
derived from the exact-check results (which raised or passed), not from
float64 arithmetic that can silently under/overflow.
"""
from __future__ import annotations

import ast
import pathlib


_REPO_ROOT = pathlib.Path(__file__).parent.parent
_PRIMITIVES = _REPO_ROOT / "htf" / "_rayleigh_primitives.py"
_RAYLEIGH_CERT = _REPO_ROOT / "htf" / "rayleigh_cert.py"


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _fstring_has_float_format(node: ast.JoinedStr, src: str) -> str | None:
    """Return the offending source fragment if a FormattedValue uses a float format spec."""
    for part in node.values:
        if not isinstance(part, ast.FormattedValue):
            continue
        spec = part.format_spec
        if spec is None:
            continue
        if not isinstance(spec, ast.JoinedStr):
            continue
        # format_spec content as source
        spec_text = ast.get_source_segment(src, spec) or ""
        # float format indicators: f, g, e (with optional width/precision digits)
        for ch in ("f", "g", "e", "G", "E"):
            if ch in spec_text and any(c.isdigit() or c == "." for c in spec_text):
                frag = ast.get_source_segment(src, node) or "<unknown>"
                return frag
    return None


def _collect_float_vars_in_function(func: ast.FunctionDef) -> set[str]:
    """Return names that are assigned from float() calls in the function body."""
    names: set[str] = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not (isinstance(call.func, ast.Name) and call.func.id == "float"):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _fstring_uses_float_var(node: ast.JoinedStr, float_vars: set[str], src: str) -> str | None:
    """Return offending fragment if a FormattedValue references a known float64 variable."""
    for part in node.values:
        if not isinstance(part, ast.FormattedValue):
            continue
        if isinstance(part.value, ast.Name) and part.value.id in float_vars:
            frag = ast.get_source_segment(src, node) or "<unknown>"
            return frag
    return None


# ─────────────────────── test: _check_preconditions ─────────────────────────

class TestCheckPreconditionsHygiene:
    """Lint: _check_preconditions must not embed float64 arithmetic in assumption text."""

    def _setup(self):
        src = _PRIMITIVES.read_text(encoding="utf-8")
        tree = ast.parse(src)
        func = _find_function(tree, "_check_preconditions")
        assert func is not None, "_check_preconditions not found in _rayleigh_primitives.py"
        return src, func

    def test_no_float_format_spec_in_return_list(self):
        """f-strings in the returned assumptions list must not have float format specs."""
        src, func = self._setup()
        violations = []
        for node in ast.walk(func):
            if isinstance(node, ast.JoinedStr):
                frag = _fstring_has_float_format(node, src)
                if frag:
                    violations.append(frag)
        assert not violations, (
            "_check_preconditions contains f-strings with float format specs "
            "(binary64 arithmetic in assumption text can under/overflow):\n"
            + "\n".join(f"  {v}" for v in violations)
        )

    def test_float_vars_not_formatted_into_return(self):
        """Variables computed via float() must not be embedded in assumption f-strings."""
        src, func = self._setup()
        float_vars = _collect_float_vars_in_function(func)
        if not float_vars:
            return  # nothing to check
        violations = []
        for node in ast.walk(func):
            if isinstance(node, ast.JoinedStr):
                frag = _fstring_uses_float_var(node, float_vars, src)
                if frag:
                    violations.append(frag)
        assert not violations, (
            "_check_preconditions embeds float() results in assumption f-strings.  "
            "float() from numpy ops can under/overflow; use exact logical descriptions:\n"
            + "\n".join(f"  {v}" for v in violations)
        )

    def test_assumption_text_is_exact_logical(self):
        """Returned assumptions must describe exact logical checks, not float measurements."""
        src, func = self._setup()
        # Find all Return nodes containing a List
        bad = []
        for node in ast.walk(func):
            if not isinstance(node, ast.Return):
                continue
            if not isinstance(node.value, ast.List):
                continue
            for elt in node.value.elts:
                if not isinstance(elt, ast.JoinedStr):
                    continue
                # Check for numeric format specs
                frag = _fstring_has_float_format(elt, src)
                if frag:
                    bad.append(frag)
        assert not bad, (
            "Returned assumption list contains float-formatted f-strings:\n"
            + "\n".join(f"  {v}" for v in bad)
        )


# ─────────────────────── test: rayleigh_cert claims ─────────────────────────

class TestRayleighCertClaimsHygiene:
    """Lint: claim strings in rayleigh_cert.py must derive from verified upper, not float ops."""

    def _setup(self):
        src = _RAYLEIGH_CERT.read_text(encoding="utf-8")
        tree = ast.parse(src)
        return src, tree

    def test_no_raw_float_in_assumptions_assignments(self):
        """Assumption lists in rayleigh_cert.py must not embed raw float() arithmetic."""
        src, tree = self._setup()
        violations = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.List):
                continue
            for elt in node.elts:
                if not isinstance(elt, ast.JoinedStr):
                    continue
                frag = _fstring_has_float_format(elt, src)
                if frag:
                    violations.append(frag)
        assert not violations, (
            "rayleigh_cert.py contains float-formatted f-strings in list literals "
            "(possible assumption text):\n"
            + "\n".join(f"  {v}" for v in violations)
        )
