"""The append-only gate compares against the right state in every way CI runs it.

Given ``--base origin/main``, a push to main has ``HEAD == origin/main``: comparing a commit with itself can
never fail, so a push that shrank a ledger passed. And when a scorer pushed while the tests ran, ``origin/main``
was LATER than ``HEAD`` and one record read as destroyed: the merge of PR #28 failed that way on 2026-10-05.
Each case runs the real script in a throwaway git repository.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LEDGER = "dist/data/evidence/prediction-ledger.json"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True,
                          stdin=subprocess.DEVNULL).stdout.strip()


@pytest.fixture()
def repo(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    r = tmp_path / "r"
    (r / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "check_append_only_monotonic.py", r / "scripts")
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "t")
    _git(r, "config", "commit.gpgsign", "false")
    return r


def commit(repo: Path, n: int, msg: str) -> str:
    (repo / LEDGER).parent.mkdir(parents=True, exist_ok=True)
    (repo / LEDGER).write_text(json.dumps({"mode": "append_only", "made_by": msg, "entries": list(range(n))}),
                               encoding="utf-8")                 # made_by: every commit has content
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def gate(repo: Path, base: str, head: str) -> tuple[int, str]:
    p = subprocess.run([sys.executable, "scripts/check_append_only_monotonic.py", "--base", base, "--head", head],
                       cwd=repo, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    return p.returncode, p.stdout + p.stderr


def test_a_branch_regenerated_at_its_fork_point_is_refused(repo):
    fork = commit(repo, 2, "fork point")
    live = commit(repo, 3, "a scorer adds a record")
    _git(repo, "checkout", "-q", "-b", "stale", fork)
    stale = commit(repo, 2, "regenerated from the fork point")      # not in main's history
    code, out = gate(repo, live, stale)
    assert code == 1 and "1 DESTROYED" in out
    _git(repo, "checkout", "-q", "-b", "grows", live)
    assert gate(repo, live, commit(repo, 4, "adds one"))[0] == 0


def test_a_push_that_shrank_a_ledger_is_refused_though_it_is_main_itself(repo):
    commit(repo, 3, "three records")
    head = commit(repo, 2, "a push that lost one")
    code, out = gate(repo, head, head)                              # CI on a push: HEAD == origin/main
    assert code == 1 and "1 DESTROYED" in out and "built on" in out


def test_main_moving_on_during_the_run_is_not_a_shrink(repo):
    commit(repo, 3, "before")
    pushed = commit(repo, 3, "the pushed commit")
    later = commit(repo, 4, "a scorer pushed while the tests ran")
    code, out = gate(repo, later, pushed)
    assert code == 0 and "built on" in out
    # ...but the pushed commit is still checked: had IT lost a record, the overlap would not hide it
    _git(repo, "checkout", "-q", "-b", "x", pushed + "^")
    lossy = commit(repo, 2, "lost one")
    commit(repo, 5, "later")
    assert gate(repo, "HEAD", lossy)[0] == 1


def test_a_root_commit_has_nothing_to_lose(repo):
    root = commit(repo, 1, "root")
    code, out = gate(repo, root, root)
    assert code == 0 and "root commit" in out
