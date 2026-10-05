"""Scoring jobs never lose a forecast to a moved branch, and never force over another writer.

2026-10-03: a dispatched hurricane run waited in the scoring concurrency queue, checked out the
commit that TRIGGERED it, and its push was rejected -- its forecasts were lost. The workflows now
check out the branch tip at job start and push through scripts/ci/push_with_rebase.sh."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ci" / "push_with_rebase.sh"
SCORERS = ("tornado-score", "earthquake-score", "hurricane-score", "verification-score", "cross-modality-analyses")


def _bash() -> str | None:
    # On Windows a bare "bash" resolves to WSL's (CreateProcess searches System32 first); use Git's.
    if sys.platform == "win32":
        git = shutil.which("git")
        if not git:
            return None
        for root in list(Path(git).parents)[:3]:        # ...\Git\cmd\git.exe or ...\Git\mingw64\bin\git.exe
            for cand in (root / "bin" / "bash.exe", root / "usr" / "bin" / "bash.exe"):
                if cand.exists():
                    return str(cand)
        return None
    return shutil.which("bash")


def _git(cwd: Path, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, (args, r.stdout, r.stderr)
    return r.stdout


def _clone(remote: Path, where: Path) -> Path:
    subprocess.run(["git", "clone", "-q", str(remote), str(where)], check=True, capture_output=True,
                   stdin=subprocess.DEVNULL)
    _git(where, "config", "user.email", "t@example.com")
    _git(where, "config", "user.name", "t")
    _git(where, "config", "core.autocrlf", "false")
    return where


@pytest.fixture
def repos(tmp_path):
    if not _bash() or not shutil.which("git"):
        pytest.skip("git and a POSIX bash are needed")
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True, capture_output=True,
                   stdin=subprocess.DEVNULL)
    seed = _clone(remote, tmp_path / "seed")
    (seed / "shared.txt").write_text("v0\n", encoding="utf-8")
    _git(seed, "add", "shared.txt")
    _git(seed, "commit", "-q", "-m", "seed")
    _git(seed, "push", "-q", "origin", "HEAD:main")
    a = _clone(remote, tmp_path / "a")      # the writer that lands first
    b = _clone(remote, tmp_path / "b")      # the stale job: cloned before a's push
    return remote, a, b


def _run_script(cwd: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, GITHUB_REF_NAME="main", PUSH_REMOTE="origin")
    return subprocess.run([_bash(), str(SCRIPT)], cwd=cwd, env=env, capture_output=True, text=True,
                          stdin=subprocess.DEVNULL)


def test_a_stale_job_rebases_onto_the_moved_branch_and_both_commits_land(repos):
    remote, a, b = repos
    (a / "eq.json").write_text("earthquake\n", encoding="utf-8")
    _git(a, "add", "eq.json"); _git(a, "commit", "-q", "-m", "earthquake data"); _git(a, "push", "-q", "origin", "HEAD:main")
    (b / "hu.json").write_text("hurricane\n", encoding="utf-8")
    _git(b, "add", "hu.json"); _git(b, "commit", "-q", "-m", "hurricane data")
    r = _run_script(b)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "rebasing onto" in r.stdout
    log = _git(a, "ls-remote", str(remote), "refs/heads/main")
    head = log.split()[0]
    msgs = _git(b, "log", "--format=%s", head)
    assert "hurricane data" in msgs and "earthquake data" in msgs


def test_files_the_job_changed_but_does_not_commit_do_not_block_the_rebase(repos):
    """2026-10-03T16:33Z: an earthquake run lost its forecast to "cannot pull with rebase: You have
    unstaged changes" (a tracked file it modified outside dist/), reported as a rebase conflict."""
    remote, a, b = repos
    (a / "eq.json").write_text("earthquake\n", encoding="utf-8")
    _git(a, "add", "eq.json"); _git(a, "commit", "-q", "-m", "earthquake data"); _git(a, "push", "-q", "origin", "HEAD:main")
    (b / "hu.json").write_text("hurricane\n", encoding="utf-8")
    _git(b, "add", "hu.json"); _git(b, "commit", "-q", "-m", "hurricane data")
    (b / "shared.txt").write_text("touched by the job, not committed\n", encoding="utf-8")   # unstaged
    r = _run_script(b)
    assert r.returncode == 0, r.stdout + r.stderr
    head = _git(a, "ls-remote", str(remote), "refs/heads/main").split()[0]
    assert "hurricane data" in _git(b, "log", "--format=%s", head)
    assert (b / "shared.txt").read_text(encoding="utf-8") == "touched by the job, not committed\n"   # put back


def test_a_conflicting_writer_fails_loudly_and_nothing_is_forced(repos):
    remote, a, b = repos
    (a / "shared.txt").write_text("from a\n", encoding="utf-8")
    _git(a, "commit", "-q", "-am", "a changes shared"); _git(a, "push", "-q", "origin", "HEAD:main")
    a_head = _git(a, "rev-parse", "HEAD").strip()
    (b / "shared.txt").write_text("from b\n", encoding="utf-8")
    _git(b, "commit", "-q", "-am", "b changes shared")
    r = _run_script(b)
    assert r.returncode == 1
    assert "rebase conflict" in r.stdout
    assert _git(a, "ls-remote", str(remote), "refs/heads/main").split()[0] == a_head   # a's work stands
    assert not (b / ".git" / "rebase-merge").exists() and not (b / ".git" / "rebase-apply").exists()


def test_every_scoring_workflow_checks_out_the_tip_and_pushes_through_the_script():
    # plain text, not a YAML parser: CI installs no PyYAML, and a skipped guard guards nothing
    for name in SCORERS:
        lines = (ROOT / ".github" / "workflows" / f"{name}.yml").read_text(encoding="utf-8").splitlines()
        uses = [i for i, l in enumerate(lines) if "uses: actions/checkout@" in l]
        assert uses, name
        for i in uses:
            block = [l.strip() for l in lines[i + 1:i + 4]]
            assert "with:" in block and "ref: ${{ github.ref }}" in block, (name, block)
        assert any(l.strip() == "bash scripts/ci/push_with_rebase.sh" for l in lines), name
        assert not any(l.strip() == "git push" for l in lines), name


def test_every_workflow_that_commits_site_files_deploys_them():
    """A bot push does not trigger deploy.yml, so a workflow that commits dist/ must deploy itself,
    after its push (found 2026-10-03: the earthquake model switch sat undeployed)."""
    for path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        commits_dist = any("git add" in l and "dist/" in l for l in text.splitlines())
        if not commits_dist:
            continue
        assert "cloudflare/wrangler-action@" in text, path.name
        assert text.index("push_with_rebase.sh") < text.index("cloudflare/wrangler-action@"), path.name
