"""The SHIPS text archive (docs/MODEL_IMPROVEMENT_LEDGER.md, H4): every listed text kept once,
byte-reproducibly, nothing else written, and the hurricane workflow runs it and commits it."""
from __future__ import annotations

import gzip
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("archive_ships_t", ROOT / "scripts" / "archive_ships_text.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


INDEX = ('<a href="?C=N;O=D">Name</a> <a href="/atcf/">Parent</a> '
         '<a href="26100306EP1826_ships.txt">26100306EP1826_ships.txt</a> '
         '<a href="26100312EP1826_ships.txt">26100312EP1826_ships.txt</a> '
         '<a href="../../etc/26100318EP1826_ships.txt">x</a> <a href="notes.txt">notes</a>')


def test_every_listed_text_is_kept_once_and_nothing_else(tmp_path):
    m = _mod()
    assert m.listed(INDEX) == ["26100306EP1826_ships.txt", "26100312EP1826_ships.txt"]
    calls = []

    def fetch(url):
        calls.append(url)
        return f"SHIPS for {url.rsplit('/', 1)[1]}\n".encode()
    first = m.sync(INDEX, fetch, root=tmp_path)
    assert first == {"listed": 2, "already": 0, "archived_now": 2, "failed": []}
    p = tmp_path / "2026" / "26100306EP1826_ships.txt.gz"
    assert gzip.decompress(p.read_bytes()) == b"SHIPS for 26100306EP1826_ships.txt\n"
    before = p.read_bytes()
    again = m.sync(INDEX, fetch, root=tmp_path)                     # idempotent: nothing refetched
    assert again["archived_now"] == 0 and again["already"] == 2 and len(calls) == 2
    assert p.read_bytes() == before == m.deterministic_gzip(b"SHIPS for 26100306EP1826_ships.txt\n")
    assert sorted(x.name for x in tmp_path.rglob("*") if x.is_file()) == [
        "26100306EP1826_ships.txt.gz", "26100312EP1826_ships.txt.gz"]


def test_a_failed_or_empty_fetch_is_reported_and_not_written(tmp_path):
    m = _mod()

    def fetch(url):
        if url.endswith("26100306EP1826_ships.txt"):
            raise OSError("404")
        return b"   \n"
    res = m.sync(INDEX, fetch, root=tmp_path)
    assert res["archived_now"] == 0 and len(res["failed"]) == 2
    assert not any(tmp_path.rglob("*.gz"))
    with pytest.raises(ValueError):
        m.archive_path("../../evil_ships.txt", tmp_path)


def test_the_hurricane_workflow_archives_and_commits_the_texts():
    wf = (ROOT / ".github" / "workflows" / "hurricane-score.yml").read_text(encoding="utf-8")
    assert "python -u scripts/archive_ships_text.py" in wf
    assert "results/hurricane_ships_archive" in wf.split("Check for changes", 1)[1].split("Commit and push", 1)[0]
    assert "git add -f dist/ results/hurricane_ships_archive/" in wf
