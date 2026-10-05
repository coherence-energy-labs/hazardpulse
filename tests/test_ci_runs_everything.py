"""CI runs every test file. On 2026-10-05 the hand-picked lists in tests.yml had left 39 of the 88 test
files out of CI -- every integrity test of that day's audit among them -- so a fix could regress silently."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_test_workflow_runs_the_whole_tests_directory():
    text = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    runs = re.findall(r"python -m pytest[^\n]*", text)
    assert any(re.search(r"\stests/\s*$", r) for r in runs), runs
    # no list of individual files anywhere (a list is how files got left out)
    assert not re.search(r"pytest[^\n]*\\\n\s+tests/test_", text)


def test_every_test_module_is_collectable_by_name():
    """A file pytest would not collect (wrong prefix) is a test that never runs."""
    for p in (ROOT / "tests").glob("*.py"):
        if p.name in ("__init__.py", "conftest.py"):
            continue
        body = p.read_text(encoding="utf-8")
        if re.search(r"^def test_", body, re.M):
            assert p.name.startswith("test_"), f"{p.name} has tests but pytest will not collect it"
