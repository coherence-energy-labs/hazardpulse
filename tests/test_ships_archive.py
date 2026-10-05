"""The SHIPS text archive (docs/MODEL_IMPROVEMENT_LEDGER.md, H4): every listed text kept, byte-reproducibly,
EVERY version of it with its retrieval time (NHC rewrites texts after first publishing them), nothing ever
overwritten or deleted, and the hurricane workflow runs it and commits it."""
from __future__ import annotations

import datetime as dt
import gzip
import importlib.util
import json
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
NOW = dt.datetime(2026, 10, 3, 19, 7, 41)


def _files(root: Path) -> list[str]:
    return sorted(x.relative_to(root).as_posix() for x in root.rglob("*") if x.is_file())


def test_every_listed_text_is_kept_once_and_nothing_else(tmp_path):
    m = _mod()
    assert m.listed(INDEX) == ["26100306EP1826_ships.txt", "26100312EP1826_ships.txt"]
    calls = []

    def fetch(url):
        calls.append(url)
        return f"SHIPS for {url.rsplit('/', 1)[1]}\n".encode()
    first = m.sync(INDEX, fetch, root=tmp_path, now=NOW)
    assert (first["listed"], first["already"], first["archived_now"], first["failed"]) == (2, 0, 2, [])
    p = tmp_path / "2026" / "26100306EP1826_ships.txt.gz"
    assert gzip.decompress(p.read_bytes()) == b"SHIPS for 26100306EP1826_ships.txt\n"
    before = p.read_bytes()
    again = m.sync(INDEX, fetch, root=tmp_path, now=NOW + dt.timedelta(hours=1))   # nothing has changed
    assert again["archived_now"] == 0 and again["already"] == 2 and len(calls) == 2
    assert p.read_bytes() == before == m.deterministic_gzip(b"SHIPS for 26100306EP1826_ships.txt\n")
    assert _files(tmp_path) == ["2026/26100306EP1826_ships.txt.gz", "2026/26100312EP1826_ships.txt.gz",
                                "2026/versions.jsonl"]


def _listing(*rows: tuple[str, str]) -> str:
    return "\n".join(f'<a href="{n}">{n[:20]}..&gt;</a> {stamp}  9.0K  ' for n, stamp in rows)


RACHEL = "26100318EP1826_ships.txt"
PRELIM = b"... DTOPS:     0.0%    0.0%\n       SDCON:     1.4%\n RII 30/24 12.6\n"
FINAL = b"... RII 30/24 12.6\n"


def test_a_text_nhc_rewrites_keeps_every_version_with_its_retrieval_time(tmp_path):
    """Rachel 2026-10-03 18Z: archived at 18:47 with a DTOPS line, rewritten by NHC at 18:50 without it.
    The archive kept only the first version seen; both are kept now, and the first is never rewritten."""
    m = _mod()
    server = {RACHEL: PRELIM}
    fetch = lambda url: server[url.rsplit("/", 1)[1]]                                      # noqa: E731
    t1 = dt.datetime(2026, 10, 3, 18, 47, 2)
    m.sync(_listing((RACHEL, "2026-10-03 18:46")), fetch, root=tmp_path, now=t1)
    server[RACHEL] = FINAL                                                                  # NHC rewrites it
    t2 = dt.datetime(2026, 10, 3, 23, 58, 6)
    res = m.sync(_listing((RACHEL, "2026-10-03 18:50")), fetch, root=tmp_path, now=t2)
    assert res["new_versions"] == 1 and res["rechecked"] == 1
    base = tmp_path / "2026" / f"{RACHEL}.gz"
    assert gzip.decompress(base.read_bytes()) == PRELIM                                    # never rewritten
    vs = m.versions(RACHEL, tmp_path)
    assert [(v["retrieved_at"], v["listed_modified"]) for v in vs] == [
        ("2026-10-03T18:47:02Z", "2026-10-03 18:46"), ("2026-10-03T23:58:06Z", "2026-10-03 18:50")]
    assert gzip.decompress((tmp_path / vs[1]["file"]).read_bytes()) == FINAL
    assert vs[1]["file"] == f"2026/versions/{RACHEL}.20261003T235806Z.gz"

    # unchanged listing: nothing is read again; a third rewrite is a third version; nothing is lost
    calls = []
    counting = lambda url: calls.append(url) or server[url.rsplit("/", 1)[1]]              # noqa: E731
    assert m.sync(_listing((RACHEL, "2026-10-03 18:50")), counting, root=tmp_path,
                  now=dt.datetime(2026, 10, 4, 6, 1))["rechecked"] == 0 and calls == []
    server[RACHEL] = FINAL + b" (third)\n"
    m.sync(_listing((RACHEL, "2026-10-04 07:12")), counting, root=tmp_path, now=dt.datetime(2026, 10, 4, 12, 40))
    assert len(m.versions(RACHEL, tmp_path)) == 3 and len(_files(tmp_path)) == 4            # 3 texts + the log


def test_the_rewritten_text_is_kept_beside_the_first(tmp_path):
    """The bare contract, on the real clock: after NHC rewrites a text, the archive holds both texts."""
    m = _mod()
    server = {RACHEL: PRELIM}
    fetch = lambda url: server[url.rsplit("/", 1)[1]]                                      # noqa: E731
    m.sync(_listing((RACHEL, "2026-10-03 18:46")), fetch, root=tmp_path)
    server[RACHEL] = FINAL
    m.sync(_listing((RACHEL, "2099-01-01 00:00")), fetch, root=tmp_path)                  # modified after our read
    assert sorted(gzip.decompress(p.read_bytes()) for p in tmp_path.rglob("*.gz")) == sorted([PRELIM, FINAL])


def test_a_reread_that_finds_the_same_text_is_logged_not_stored_again(tmp_path):
    m = _mod()
    fetch = lambda url: PRELIM                                                              # noqa: E731
    m.sync(_listing((RACHEL, "2026-10-03 18:46")), fetch, root=tmp_path, now=dt.datetime(2026, 10, 3, 18, 46, 30))
    # read in the same minute it was last modified: it may have changed after the read, so it is read again
    res = m.sync(_listing((RACHEL, "2026-10-03 18:46")), fetch, root=tmp_path, now=dt.datetime(2026, 10, 4, 0, 3))
    assert res["rechecked"] == 1 and res["unchanged"] == 1 and res["new_versions"] == 0
    log = m.manifest(tmp_path)[RACHEL]
    assert [e["kind"] for e in log] == ["version", "seen"] and log[1]["file"] == log[0]["file"]


def test_files_archived_before_the_log_are_reread_only_when_the_listing_says_they_changed(tmp_path):
    m = _mod()
    legacy = [{"name": RACHEL, "kind": "version", "retrieved_at": None, "listed_modified": None,
               "archived_by_commit_at": "2026-10-03T18:47:31Z"}]
    assert m.needs_read(RACHEL, legacy, "2026-10-03 18:50", NOW) is True                   # modified after the commit
    assert m.needs_read(RACHEL, legacy, "2026-10-03 18:41", NOW) is False
    assert m.needs_read(RACHEL, [], "2026-10-03 18:41", NOW) is True                       # not even logged: read once
    seen = [{"retrieved_at": "2026-10-03T19:00:00Z", "listed_modified": "2026-10-03 18:50"}]
    assert m.needs_read(RACHEL, seen, None, dt.datetime(2026, 10, 4, 2)) is True           # no times: recent, 7 h
    assert m.needs_read(RACHEL, seen, None, dt.datetime(2026, 10, 3, 21)) is False         # ... but not every run
    assert m.needs_read(RACHEL, seen, None, dt.datetime(2026, 10, 8)) is False             # old cycle: settled


def test_a_text_archived_by_an_older_run_is_logged_and_its_rewrite_kept(tmp_path):
    """A base file written before the log existed (an older run, after the backfill) is read once: the
    stored first version is logged as it is, the server's current text beside it."""
    m = _mod()
    base = m.archive_path(RACHEL, tmp_path)
    base.parent.mkdir(parents=True)
    base.write_bytes(m.deterministic_gzip(PRELIM))
    res = m.sync(_listing((RACHEL, "2026-10-03 18:50")), lambda url: FINAL, root=tmp_path,
                 now=dt.datetime(2026, 10, 5, 6, 47))
    assert res["rechecked"] == 1 and res["new_versions"] == 1
    vs = m.versions(RACHEL, tmp_path)
    assert [v["retrieved_at"] for v in vs] == [None, "2026-10-05T06:47:00Z"]
    assert gzip.decompress(base.read_bytes()) == PRELIM and "not recorded" in vs[0]["note"]


def test_a_failed_or_empty_fetch_is_reported_and_not_written(tmp_path):
    m = _mod()

    def fetch(url):
        if url.endswith("26100306EP1826_ships.txt"):
            raise OSError("404")
        return b"   \n"
    res = m.sync(INDEX, fetch, root=tmp_path, now=NOW)
    assert res["archived_now"] == 0 and len(res["failed"]) == 2
    assert not any(tmp_path.rglob("*.gz")) and not any(tmp_path.rglob("*.jsonl"))
    with pytest.raises(ValueError):
        m.archive_path("../../evil_ships.txt", tmp_path)


def test_the_committed_archive_and_its_log_agree():
    """Every logged file exists with the bytes its SHA-256 names and says when it was read (the backfill
    gave the texts archived before the log the time of the commit that added them -- an upper bound on
    when they were read); every later version is logged. A base text archived by an older run and not
    logged yet is read once by the next run (test above), so it is allowed here."""
    m = _mod()
    root = ROOT / "results" / "hurricane_ships_archive"
    if not root.exists():
        pytest.skip("archive not in this checkout")
    log = m.manifest(root)
    bases = {p.name[:-3] for p in root.glob("*/*_ships.txt.gz")}
    assert not bases or log, "the archive holds texts but logs none"
    logged_files = set()
    for name, entries in log.items():
        for e in entries:
            p = root / e["file"]
            assert p.exists(), e
            logged_files.add(e["file"])
            if e.get("kind") == "version":
                assert m._stored_sha(p) == e["sha256"], e
                assert (e.get("retrieved_at") or e.get("archived_by_commit_at")
                        or "not recorded" in e.get("note", "")), e
    versions = {p.relative_to(root).as_posix() for p in root.glob(f"*/{m.VERSIONS_DIR}/*.gz")}
    assert versions <= logged_files


def test_the_hurricane_workflow_archives_and_commits_the_texts():
    wf = (ROOT / ".github" / "workflows" / "hurricane-score.yml").read_text(encoding="utf-8")
    assert "python -u scripts/archive_ships_text.py" in wf
    assert "results/hurricane_ships_archive" in wf.split("Check for changes", 1)[1].split("Commit and push", 1)[0]
    assert "git add -f dist/ results/hurricane_ships_archive/" in wf
