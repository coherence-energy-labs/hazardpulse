"""An issued hurricane record is never rewritten with different content (its ledger entry hashes the id, the storms
and the catch-up records). 2026-10-10: a sanitized live-storms.json that still named the 03:39Z forecast made the
next scorer run rewrite that record (-Infinity -> null), and every verification run failed its record audit."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _bsa():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("bsa_frozen_t", ROOT / "scripts" / "build_site_artifacts.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_an_issued_record_keeps_its_content_and_may_gain_display_fields(tmp_path):
    bsa = _bsa()
    p = tmp_path / "hu_fcst_20261010_0337.json"
    issued = {"forecast_id": "hu_fcst_20261010_0337", "storms": [{"storm_id": "AL092026", "x": float("-inf")}]}
    assert not bsa._hurricane_content_changed(p, issued)                     # nothing issued yet: write it
    p.write_text(json.dumps(issued), encoding="utf-8")                         # as the scorer writes it (-Infinity)
    same_plus = dict(issued, top_probability=0.0, source_artifacts=["/data/live-storms.json"])
    assert not bsa._hurricane_content_changed(p, same_plus)                  # display fields only: allowed
    sanitized = {"forecast_id": issued["forecast_id"], "storms": [{"storm_id": "AL092026", "x": None}]}
    assert bsa._hurricane_content_changed(p, sanitized)                      # -inf -> null is a content change
    with_catch_up = dict(issued, shadow_catch_up=[{"storm_id": "AL092026"}])
    assert bsa._hurricane_content_changed(p, with_catch_up)
