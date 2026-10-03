"""The site build writes signed/hashed JSON as LF bytes on every platform and never rewrites an
unchanged file (a Windows build rewrote two frozen replays to CRLF; .gitattributes pins them)."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _bsa():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("bsa_byte_stable", ROOT / "scripts" / "build_site_artifacts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_json_is_written_as_lf_bytes_and_an_unchanged_file_is_left_alone(tmp_path):
    bsa = _bsa()
    p = tmp_path / "replay" / "to_fcst_x.json"
    payload = {"forecast_id": "to_fcst_x", "storms": [{"p": 0.1}]}
    bsa._write_json(p, payload)
    data = p.read_bytes()
    assert bytes([13, 10]) not in data and data.endswith(bytes([10]))
    os.utime(p, (1_000_000_000, 1_000_000_000))
    bsa._write_json(p, payload)                      # same content: not touched
    assert p.stat().st_mtime == 1_000_000_000
    bsa._write_json(p, {**payload, "storms": []})    # new content: written
    assert p.stat().st_mtime != 1_000_000_000 and b'"storms": []' in p.read_bytes()


def test_the_replay_and_evidence_files_are_pinned_against_line_ending_conversion():
    attrs = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "dist/data/replay/** -text" in attrs and "dist/data/evidence/** -text" in attrs
