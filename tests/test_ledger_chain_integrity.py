"""NEGATIVE tests for the append-only hash chains: each forgery must be REJECTED.

Before these existed the only assertion about the ledgers anywhere in `tests/` was
`test_earthquake_workflows.py::test_append_ledger_skips_duplicate_forecasts`, which checks
`lines[1]["prev_hash"] == lines[0]["hash"]` -- a POSITIVE linkage assertion. Nothing asserted that
anything is rejected, so the entire append-only property was untested.

Every test here builds a forgery with the production writer's own digest, confirms the OLD
read-time check (`build_site_artifacts._count_link_mismatches`, prev_hash-vs-hash only) reports
ZERO mismatches for it, and then requires the new verifier to REJECT it. Without the
old-check-is-blind assertion these would be strawmen: they have to reproduce the defect that was
actually measured, not an easier one.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, relative_path: str):
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


verifier = _load_script_module("hazardpulse_verify_ledger_chain", "scripts/verify_ledger_chain.py")
site = _load_script_module("hazardpulse_build_site_artifacts_ledger_test",
                           "scripts/build_site_artifacts.py")

EQ_SPEC = next(s for s in verifier.LEDGERS if "earthquake" in s.path)
TO_SPEC = next(s for s in verifier.LEDGERS if "tornado" in s.path)
SPECS = pytest.mark.parametrize("spec", verifier.LEDGERS, ids=[s.path.split("/")[-1]
                                                              for s in verifier.LEDGERS])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def build_chain(spec, n: int = 8, probability: float = 0.10) -> list[dict]:
    """Chain built exactly the way the production appenders build one."""
    rows: list[dict] = []
    previous = verifier.GENESIS
    for i in range(n):
        entry = {
            "forecast_id": f"fx_fcst_2026010{i}_0000",
            "timestamp": f"2026-01-{i + 1:02d}T00:00:00Z",
            "model_version": "test_v1",
            "top_probability": round(probability + i / 100, 4),
            "prev_hash": previous,
        }
        entry["hash"] = spec.digest(entry)
        rows.append(entry)
        previous = entry["hash"]
    return rows


def anchor_for(spec, rows: list[dict]) -> dict:
    report = verifier.Report(path=spec.path)
    verifier.verify_rows(rows, spec, report)
    assert report.ok, f"fixture chain does not verify: {report.violations}"
    return {
        "n_entries": report.n_rows,
        "first_hash": report.first_hash,
        "head_hash": report.head_hash,
        "chain_commitment": report.chain_commitment,
    }


def verify(spec, rows: list[dict], anchor: dict | None) -> verifier.Report:
    report = verifier.Report(path=spec.path)
    verifier.verify_rows(rows, spec, report)
    verifier.verify_anchor(report, anchor)
    return report


def old_check_mismatches(rows: list[dict], tmp_path: Path) -> int:
    """The check that shipped: compares stored prev_hash to the previous stored hash. Nothing else."""
    path = tmp_path / "ledger.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    _n_rows, mismatches = site._count_link_mismatches(path)
    return mismatches


# ---------------------------------------------------------------------------
# anti-vacuity: this file must not be able to pass by testing nothing
# ---------------------------------------------------------------------------


def test_self_test_of_the_verifier_passes():
    """The shipped `--self-test` is the repo's convention (check_append_only_monotonic.py)."""
    proc = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "verify_ledger_chain.py"),
                           "--self-test"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SELF-TEST GREEN" in proc.stdout


def test_the_suite_is_not_vacuous():
    """A negative suite that discovers zero ledgers passes for any input. Refuse that."""
    assert len(verifier.LEDGERS) >= 2, "expected both live hash chains to be registered"
    paths = {s.path for s in verifier.LEDGERS}
    assert "dist/data/earthquake-ledger.jsonl" in paths
    assert "dist/data/tornado-ledger.jsonl" in paths
    for spec in verifier.LEDGERS:
        rows = build_chain(spec)
        assert len(rows) == 8
        report = verify(spec, rows, anchor_for(spec, rows))
        assert report.ok, f"an HONEST chain was rejected, so every rejection below is meaningless: "\
                          f"{report.violations}"
        assert report.n_rows == 8


@SPECS
def test_digest_spec_matches_the_production_writer(spec):
    """The verifier must recompute the SAME digest the writer computes, not a plausible one."""
    entry = {"forecast_id": "fx", "timestamp": "2026-01-01T00:00:00Z", "prev_hash": verifier.GENESIS}
    import hashlib

    kwargs = {"sort_keys": True}
    if spec.compact:
        kwargs["separators"] = (",", ":")
    expected = hashlib.sha256(json.dumps(entry, **kwargs).encode("utf-8")).hexdigest()
    assert spec.digest(entry) == expected
    # and the two ledgers really do disagree, so a single shared spec would be wrong
    assert EQ_SPEC.compact is False and TO_SPEC.compact is True


# ---------------------------------------------------------------------------
# the four forgeries
# ---------------------------------------------------------------------------


@SPECS
def test_rejects_mutated_middle_entry(spec, tmp_path):
    """Rewrite one entry's probability, touch nothing else. The old check saw 0 mismatches."""
    rows = build_chain(spec)
    anchor = anchor_for(spec, rows)
    forged = copy.deepcopy(rows)
    forged[3]["top_probability"] = 0.99
    forged[3]["timestamp"] = "2026-01-04T00:00:00Z"

    assert old_check_mismatches(forged, tmp_path) == 0, "fixture is a strawman: the OLD check caught it"

    report = verify(spec, forged, anchor)
    assert not report.ok
    assert any("FORGED CONTENT" in v and "row 3" in v for v in report.violations), report.violations


@SPECS
def test_rejects_backdated_entry(spec, tmp_path):
    """Backdating is the specific lie the product's claim rules out."""
    rows = build_chain(spec)
    anchor = anchor_for(spec, rows)
    forged = copy.deepcopy(rows)
    forged[4]["timestamp"] = "2020-01-01T00:00:00Z"

    assert old_check_mismatches(forged, tmp_path) == 0
    report = verify(spec, forged, anchor)
    assert not report.ok
    assert any("FORGED CONTENT" in v for v in report.violations)


@SPECS
def test_rejects_truncated_tail(spec, tmp_path):
    """Dropping the tail leaves a chain that is internally PERFECT. Only the anchor catches it."""
    rows = build_chain(spec)
    anchor = anchor_for(spec, rows)
    truncated = copy.deepcopy(rows)[:5]

    assert old_check_mismatches(truncated, tmp_path) == 0

    per_row = verifier.Report(path=spec.path)
    verifier.verify_rows(truncated, spec, per_row)
    assert per_row.ok, "the truncated chain should pass every per-row check -- that is the point"

    report = verify(spec, truncated, anchor)
    assert not report.ok
    assert any("ANCHOR MISMATCH on n_entries" in v for v in report.violations), report.violations
    assert any("REMOVED" in v for v in report.violations)


@SPECS
def test_rejects_wholly_forged_chain(spec, tmp_path):
    """A complete rewrite, re-chained from genesis. Self-consistent, and entirely fabricated."""
    rows = build_chain(spec)
    anchor = anchor_for(spec, rows)
    forged = build_chain(spec, n=8, probability=0.90)

    assert old_check_mismatches(forged, tmp_path) == 0
    per_row = verifier.Report(path=spec.path)
    verifier.verify_rows(forged, spec, per_row)
    assert per_row.ok, "a re-chained forgery is internally consistent by construction"

    report = verify(spec, forged, anchor)
    assert not report.ok
    assert any("ANCHOR MISMATCH" in v for v in report.violations), report.violations


@SPECS
def test_rejects_reordered_entries(spec, tmp_path):
    rows = build_chain(spec)
    anchor = anchor_for(spec, rows)
    swapped = copy.deepcopy(rows)
    swapped[2], swapped[3] = swapped[3], swapped[2]

    report = verify(spec, swapped, anchor)
    assert not report.ok
    assert any("BROKEN LINK" in v for v in report.violations), report.violations


@SPECS
def test_rejects_relinked_deletion(spec, tmp_path):
    """Delete a middle entry and RE-POINT the successor, so the linkage still reads clean."""
    rows = build_chain(spec)
    anchor = anchor_for(spec, rows)
    forged = copy.deepcopy(rows)
    removed = forged.pop(4)
    forged[4]["prev_hash"] = removed["prev_hash"]

    assert old_check_mismatches(forged, tmp_path) == 0, "fixture is a strawman"
    report = verify(spec, forged, anchor)
    assert not report.ok
    assert any("FORGED CONTENT" in v for v in report.violations), report.violations


@SPECS
def test_rejects_new_row_that_leaves_a_field_uncovered(spec, tmp_path):
    """The legacy `forecast_id` gap is PINNED: a NEW row may not reintroduce an uncovered field.

    The pin is by INDEX (13 rows on earthquake, 170 on tornado), so the fixture uses a spec whose
    window is 1 row -- appending 170 fixture rows would test the same rule far more slowly.
    """
    spec = verifier.LedgerSpec(path=spec.path, writer=spec.writer, compact=spec.compact,
                               legacy_uncovered_rows=1)
    rows = build_chain(spec)
    tail = {
        "timestamp": "2026-02-01T00:00:00Z",
        "model_version": "test_v1",
        "top_probability": 0.5,
        "prev_hash": rows[-1]["hash"],
    }
    tail["hash"] = spec.digest(tail)          # digest taken BEFORE forecast_id is attached
    tail["forecast_id"] = "fx_fcst_20260201_0000"
    rows.append(tail)

    assert old_check_mismatches(rows, tmp_path) == 0
    report = verifier.Report(path=spec.path)
    verifier.verify_rows(rows, spec, report)
    assert not report.ok
    assert any("must cover every field it carries" in v for v in report.violations), report.violations


@SPECS
def test_anchor_binds_fields_the_write_time_digest_does_not_cover(spec):
    """Legacy rows do not hash `forecast_id`. Mutating it must still be caught -- by the anchor."""
    rows = build_chain(spec, n=3)
    legacy = []
    previous = verifier.GENESIS
    for i in range(3):
        entry = {"timestamp": f"2026-01-0{i + 1}T00:00:00Z", "model_version": "test_v1",
                 "top_probability": 0.1, "prev_hash": previous}
        entry["hash"] = spec.digest(entry)
        entry["forecast_id"] = f"legacy_{i}"      # retrofitted AFTER hashing, exactly as in the live file
        legacy.append(entry)
        previous = entry["hash"]

    spec_legacy = verifier.LedgerSpec(path=spec.path, writer=spec.writer, compact=spec.compact,
                                      legacy_uncovered_rows=3)
    anchor = anchor_for(spec_legacy, legacy)

    tampered = copy.deepcopy(legacy)
    tampered[1]["forecast_id"] = "legacy_ANYTHING_I_LIKE"

    per_row = verifier.Report(path=spec.path)
    verifier.verify_rows(tampered, spec_legacy, per_row)
    assert per_row.ok, "the write-time digest genuinely does not cover forecast_id on legacy rows"

    report = verify(spec_legacy, tampered, anchor)
    assert not report.ok
    assert any("chain_commitment" in v for v in report.violations), report.violations


@SPECS
def test_missing_anchor_is_fatal(spec):
    rows = build_chain(spec)
    report = verify(spec, rows, None)
    assert not report.ok
    assert any("NO ANCHOR" in v for v in report.violations)


@SPECS
def test_broken_genesis_is_rejected(spec):
    rows = build_chain(spec)
    forged = copy.deepcopy(rows)
    forged[0]["prev_hash"] = "f" * 64
    report = verify(spec, forged, anchor_for(spec, rows))
    assert not report.ok
    assert any("genesis" in v for v in report.violations), report.violations


# ---------------------------------------------------------------------------
# the anchor must be refreshed BY THE WRITER, or it goes stale on the first real append
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "script,module_name",
    [("scripts/fetch_and_score_earthquake.py", "hazardpulse_eq_anchor_wiring"),
     ("scripts/fetch_and_score_tornado.py", "hazardpulse_to_anchor_wiring")],
)
def test_writers_expose_the_anchor_refresh(script, module_name):
    """Both appenders must call refresh_ledger_anchor; a workflow step could be forgotten."""
    source = (PROJECT_ROOT / script).read_text(encoding="utf-8")
    assert "def refresh_ledger_anchor(" in source, f"{script} has no anchor refresh"
    assert source.count("refresh_ledger_anchor()") >= 1, f"{script} never CALLS it"


def test_earthquake_production_append_refreshes_the_anchor(tmp_path, monkeypatch):
    """Appending to the PRODUCTION ledger re-anchors; an explicit test/replay path does not."""
    module = _load_script_module("hazardpulse_eq_append_anchor", "scripts/fetch_and_score_earthquake.py")
    calls: list[int] = []
    monkeypatch.setattr(module, "refresh_ledger_anchor", lambda: calls.append(1))

    side_ledger = tmp_path / "side.jsonl"
    scored = [{"lat": 1.0, "lon": 2.0, "probability": 0.3, "conditions_met": 2, "max_mag": 5.0}]
    issued = __import__("datetime").datetime(2026, 4, 2, tzinfo=__import__("datetime").timezone.utc)
    module.append_ledger(scored, issued, forecast_id="eq_fcst_20260402_0000",
                         ledger_path=side_ledger)
    assert calls == [], "an explicit --ledger-path must not touch the production anchor"

    production = tmp_path / "earthquake-ledger.jsonl"
    production.write_text("", encoding="utf-8")   # the writer fails closed if it does not exist
    monkeypatch.setattr(module, "LEDGER_PATH", production)
    module.append_ledger(scored, issued, forecast_id="eq_fcst_20260402_0600",
                         ledger_path=production)
    assert calls == [1], "the production append did NOT refresh the anchor"


def test_anchor_refuses_to_certify_a_forged_ledger(tmp_path, monkeypatch):
    """--update-anchor must not launder a broken chain into a fresh commitment."""
    spec = verifier.LedgerSpec(path="ledger.jsonl", writer="test", compact=True,
                               legacy_uncovered_rows=0)
    rows = build_chain(spec)
    rows[2]["top_probability"] = 0.99                      # forged, hashes untouched
    (tmp_path / "ledger.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    monkeypatch.setattr(verifier, "LEDGERS", (spec,))
    monkeypatch.setattr(verifier, "ANCHOR_PATH", tmp_path / "anchor.json")

    with pytest.raises(SystemExit) as excinfo:
        verifier.refresh_anchor()
    assert "REFUSING to anchor" in str(excinfo.value)
    assert not (tmp_path / "anchor.json").exists()


# ---------------------------------------------------------------------------
# the live artifacts
# ---------------------------------------------------------------------------


def _live_rows(spec) -> list[dict]:
    text, origin = verifier.read_source(spec.path, None)
    assert text is not None, (
        f"{spec.path} is readable from neither the worktree nor git. This test would otherwise "
        f"pass by checking nothing.")
    report = verifier.Report(path=spec.path)
    rows = verifier.parse_rows(text, report)
    assert not report.violations, f"{spec.path} ({origin}) has unparseable lines: {report.violations}"
    return rows


@SPECS
def test_live_ledger_recomputes_end_to_end(spec):
    """Every entry of the real chain must recompute. This is what the site claims and never checked."""
    rows = _live_rows(spec)
    assert len(rows) > 100, f"only {len(rows)} rows: this is not the live ledger"
    report = verifier.Report(path=spec.path)
    verifier.verify_rows(rows, spec, report)
    assert report.ok, report.violations[:10]
    assert report.n_legacy == spec.legacy_uncovered_rows, (
        f"{report.n_legacy} rows fail to cover `forecast_id`, but {spec.legacy_uncovered_rows} are "
        f"pinned. If the count GREW, a writer regressed; if it SHRANK, re-pin it.")


@SPECS
def test_live_ledger_matches_the_committed_anchor(spec):
    rows = _live_rows(spec)
    anchors = verifier.load_anchors(None)
    assert spec.path in anchors, (
        f"no anchor for {spec.path} in {verifier.ANCHOR_PATH}. Run --update-anchor.")
    report = verify(spec, rows, anchors[spec.path])
    assert report.ok, report.violations[:10]


@SPECS
def test_mutating_a_live_entry_is_detected(spec, tmp_path):
    """The measured exploit, replayed against the REAL chain in a temp copy: entry 249 -> 0.99, 2020."""
    rows = _live_rows(spec)
    anchors = verifier.load_anchors(None)
    index = min(249, len(rows) - 1)

    forged = copy.deepcopy(rows)
    forged[index]["top_probability"] = 0.99
    forged[index]["timestamp"] = "2020-01-01T00:00:00Z"

    assert old_check_mismatches(forged, tmp_path) == 0, (
        "the OLD check must be blind to this, or the reported defect is not reproduced")

    report = verify(spec, forged, anchors[spec.path])
    assert not report.ok
    assert any(f"row {index}" in v and "FORGED CONTENT" in v for v in report.violations), \
        report.violations[:5]


@SPECS
def test_dropping_50_live_entries_is_detected(spec, tmp_path):
    rows = _live_rows(spec)
    anchors = verifier.load_anchors(None)
    truncated = copy.deepcopy(rows)[:-50]

    assert old_check_mismatches(truncated, tmp_path) == 0
    per_row = verifier.Report(path=spec.path)
    verifier.verify_rows(truncated, spec, per_row)
    assert per_row.ok, "a truncated real chain still passes every per-row check"

    report = verify(spec, truncated, anchors[spec.path])
    assert not report.ok
    assert any("ANCHOR MISMATCH on n_entries" in v for v in report.violations)
