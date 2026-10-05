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


#: packages a test may import-or-skip although CI does not install them, each for a stated reason
NOT_IN_CI = {
    "omega": "the omega_one sibling repository, not on PyPI (tests/test_forest_serve.py, test_train_best_tabular.py)",
    "xgboost": "used only with omega in tests/test_forest_serve.py, which skips without omega anyway",
}
#: import name -> pip name, where they differ
PIP_NAME = {"sklearn": "scikit-learn", "PIL": "pillow"}


def _ci_installs() -> set[str]:
    text = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    pkgs = {w for line in re.findall(r"pip install ([^\n]+)", text) for w in line.split() if not w.startswith("-")}
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    deps = project.split("dependencies = [", 1)[1].split("]", 1)[0]       # `pip install -e .` installs these
    pkgs |= {re.split(r"[<>=!~\[ ]", d.strip().strip('",'))[0].lower() for d in deps.splitlines() if d.strip()}
    return {p.lower() for p in pkgs}


def test_no_test_is_skipped_in_ci_for_want_of_a_package():
    """``-rs`` showed none of this: CI installed five packages, and tests/test_lgbm_payload.py -- the served
    tornado model's payload -- skipped its whole module there for want of lightgbm, unseen (2026-10-05)."""
    installed = _ci_installs()
    assert {"numpy", "scipy", "pytest", "scikit-learn", "cryptography"} <= installed, installed   # the parser reads
    missing = {}
    for p in sorted((ROOT / "tests").glob("test_*.py")):
        for mod in re.findall(r"importorskip\(\s*[\"']([\w.]+)", p.read_text(encoding="utf-8")):
            top = mod.split(".")[0]
            if top not in NOT_IN_CI and PIP_NAME.get(top, top).lower() not in installed:
                missing.setdefault(PIP_NAME.get(top, top), []).append(p.name)
    assert not missing, f"tests skip in CI for want of these packages (install them in tests.yml): {missing}"


def test_every_test_module_is_collectable_by_name():
    """A file pytest would not collect (wrong prefix) is a test that never runs."""
    for p in (ROOT / "tests").glob("*.py"):
        if p.name in ("__init__.py", "conftest.py"):
            continue
        body = p.read_text(encoding="utf-8")
        if re.search(r"^def test_", body, re.M):
            assert p.name.startswith("test_"), f"{p.name} has tests but pytest will not collect it"
