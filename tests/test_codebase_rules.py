"""House rules no off-the-shelf linter enforces, checked on every test run."""

from __future__ import annotations

import ast
import io
import subprocess
import tokenize
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_FILES = sorted([*(REPO_ROOT / "src").rglob("*.py"), *(REPO_ROOT / "tests").rglob("*.py")])
PRAGMAS = ("noqa", "type: ignore", "pyright: ignore")
ALLOWED_MARKDOWN = {"README.md", "AGENTS.md", "TODO.md"}
ALLOWED_MARKDOWN_DIRS = {"docs"}


def _comments(path: Path) -> list[str]:
    tokens = tokenize.generate_tokens(io.StringIO(path.read_text()).readline)
    return [
        f"{path.relative_to(REPO_ROOT)}:{tok.start[0]}: {tok.string}"
        for tok in tokens
        if tok.type == tokenize.COMMENT and not any(pragma in tok.string for pragma in PRAGMAS)
    ]


@pytest.mark.parametrize("path", PYTHON_FILES, ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_no_comments(path: Path) -> None:
    """Explain intent with names and docstrings; only tool pragmas may use ``#``."""
    assert _comments(path) == []


def test_markdown_only_in_allowed_places() -> None:
    """New prose goes in the existing docs, not in stray markdown files."""
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.md"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    stray = [
        name for name in listed if name not in ALLOWED_MARKDOWN and Path(name).parts[0] not in ALLOWED_MARKDOWN_DIRS
    ]
    assert stray == []


TEST_FILES = sorted((REPO_ROOT / "tests").rglob("*.py"))
TEST_DOUBLES = {"monkeypatch", "SimpleNamespace", "MagicMock", "Mock", "patch"}


@pytest.mark.parametrize("path", TEST_FILES, ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_tests_use_real_objects(path: Path) -> None:
    """Tests exercise the real code; patching and stand-in objects test the stand-in instead."""
    tree = ast.parse(path.read_text())
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    used |= {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}
    used |= {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    fakes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name.startswith("Fake")}
    if path.name == "test_codebase_rules.py":
        used -= TEST_DOUBLES
    assert sorted((used & TEST_DOUBLES) | fakes) == []
