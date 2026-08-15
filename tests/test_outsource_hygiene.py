"""Lint checks for outsource task documents.

Rule: Every fenced code block in outsource/*.md must be complete and
runnable — no ellipsis placeholders.  The HTF-06 Gate-A review was
BLOCKED partly because code sections used '...' to indicate omitted
implementation, preventing the reviewer from verifying the actual logic.

Detected patterns (any one triggers a failure):
  - A line whose only content is '...' (bare Python ellipsis / omitted body)
  - A line containing only a comment that is '# ...' or '# …' (Unicode ellipsis)
  - A line matching '...  # <any comment>' standing in for omitted code

These patterns are common in "relevant excerpt" style documentation.
outsource documents must instead show the complete, copy-pasteable code.
"""
from __future__ import annotations

import pathlib
import re

import pytest

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_OUTSOURCE_DIR = _REPO_ROOT / "outsource"

# Patterns that indicate an ellipsis placeholder inside a code block.
# Match the whole stripped line so we don't flag '...' inside a longer expression.
_BARE_ELLIPSIS = re.compile(r"^\s*\.\.\.\s*$")
_COMMENT_ELLIPSIS = re.compile(r"^\s*#\s*\.\.\.[\s]*$")
_UNICODE_COMMENT_ELLIPSIS = re.compile(r"^\s*#\s*…[\s]*$")
_TRAILING_ELLIPSIS_STUB = re.compile(r"^\s*(def |class )[^:]+:\s*\.\.\.\s*$")


def _violations_in_document(md_path: pathlib.Path) -> list[str]:
    """Return a list of 'file:line: <offending line>' strings."""
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    in_code_block = False
    fence_marker = ""
    results: list[str] = []

    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Track fenced code block entry/exit.
        if not in_code_block:
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_code_block = True
                fence_marker = stripped[:3]
            continue

        # Inside a code block — check for exit first.
        if stripped == fence_marker or stripped.startswith(fence_marker + " "):
            in_code_block = False
            continue

        # Check for ellipsis placeholder patterns.
        if (
            _BARE_ELLIPSIS.match(line)
            or _COMMENT_ELLIPSIS.match(line)
            or _UNICODE_COMMENT_ELLIPSIS.match(line)
            or _TRAILING_ELLIPSIS_STUB.match(line)
        ):
            rel = md_path.relative_to(_REPO_ROOT)
            results.append(f"{rel}:{lineno}: {line.rstrip()!r}")

    return results


def _outsource_md_files() -> list[pathlib.Path]:
    if not _OUTSOURCE_DIR.exists():
        return []
    return sorted(_OUTSOURCE_DIR.rglob("*.md"))


# Build parametrised cases at collection time so each document is its own
# test item; an empty directory simply produces zero items (no xfail).
_MD_FILES = _outsource_md_files()


@pytest.mark.parametrize("md_path", _MD_FILES, ids=[p.name for p in _MD_FILES])
def test_no_ellipsis_in_code_blocks(md_path: pathlib.Path) -> None:
    """outsource docs must show complete code — no '...' placeholder lines."""
    violations = _violations_in_document(md_path)
    assert not violations, (
        f"{md_path.name} contains ellipsis placeholders in code blocks.\n"
        "outsource documents must be self-contained: show the full, runnable\n"
        "code so reviewers can verify every line without access to the repo.\n\n"
        "Offending lines:\n" + "\n".join(f"  {v}" for v in violations)
    )


def test_outsource_directory_exists() -> None:
    """outsource/ directory must exist (sanity check for the test itself)."""
    assert _OUTSOURCE_DIR.is_dir(), (
        f"Expected outsource/ directory at {_OUTSOURCE_DIR}; not found."
    )
