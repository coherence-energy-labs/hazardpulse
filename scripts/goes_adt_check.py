"""Program G1 input check (amendment 12b): the collected eye decisions against CIMSS ADT's operational scene type.

    PYTHONPATH=src python scripts/goes_adt_check.py <shard dir> [--adt-dir DIR]

ADT (the Advanced Dvorak Technique, run in real time by CIMSS on every storm, every 30 min) is an objective eye /
no-eye oracle independent of this pipeline. Each collected crop is matched to the storm's ADT record nearest its
scan (within 15 min; LAND records excluded). No RI label and no outcome is read: this is an input check, and it runs
before any G1 outcome.

The gates, fixed before the 12b collection (the closed-ring rule scored 0.839, 0.02, 9 km, +0.955 and 27,199 on
the crops collected under the old detector):
- HSS of the eye flag against ADT's EYE scene on 2024-2026 (the seasons the thresholds were not fitted on) >= 0.80;
- eye fraction below 50 kt (ADT's intensity) <= 0.03 -- the weak-storm defect amendment 12b removed;
- our feature-image centre within a median 15 km of ADT's on ADT eye scenes;
- Spearman(our eye temperature, ADT's) on ADT eye scenes >= 0.90 -- geolocation and calibration;
- at least 20,000 matched crops, so the check cannot pass on nothing.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hazardpulse.hurricane import goes_abi as g  # noqa: E402

ADT_URL = "https://tropic.ssec.wisc.edu/real-time/adt/archive{year}/{num}{basin}-list.txt"
BASIN = {"AL": "L", "EP": "E", "CP": "C"}
MATCH_MIN = 15.0
GATES = {"hss_2024_2026": 0.80, "eye_frac_below_50kt": 0.03, "centre_km_median": 15.0, "teye_spearman": 0.90,
         "matched": 20000}

_MON = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}
# date time CI MSLP Vmax ... flags | eye-region T, mean cloud T, scene | RMW field (one or two tokens), MW score |
# lat, lon (deg W), fix method, satellite, view angle. MSLP/Vmax are integers in older ADT versions; scenes may be
# EYE/P (pinhole) or EYE/L (large).
ROW = re.compile(r"^(\d{4})([A-Z]{3})(\d{2}) (\d{2})(\d{2})(\d{2})\s+(\d+\.\d)\s+(\d+(?:\.\d)?)\s+(\d+(?:\.\d)?)\s.*?"
                 r"(?:ON|OFF|FLG)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+([A-Z]+(?:/[A-Z])?)\s+.*?"
                 r"(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+([A-Z]+)\s+\S+\s+\d+\.\d")
DATA_ROW = re.compile(r"^\d{4}[A-Z]{3}\d{2} \d{6} ")


def parse_adt(text: str) -> list[dict]:
    """ADT history records. Raises if any data row does not parse: a silently dropped row (a first parser lost 16 %
    of them, all EYE/P, EYE/L and integer-wind rows) biases the oracle."""
    out = []
    for line in text.splitlines():
        if not DATA_ROW.match(line):
            continue
        m = ROW.match(line)
        if not m:
            raise ValueError(f"unparsed ADT row: {line!r}")
        t = dt.datetime(int(m[1]), _MON[m[2]], int(m[3]), int(m[4]), int(m[5]), int(m[6]))
        out.append({"time": t, "vmax": float(m[9]), "eye_c": float(m[10]), "cloud_c": float(m[11]),
                    "scene": "EYE" if m[12].startswith("EYE") else m[12], "lat": float(m[13]), "lon": -float(m[14])})
    return out


def fetch_adt(storm: str, adt_dir: Path) -> str | None:
    dst = adt_dir / f"{storm}.txt"
    if dst.exists():
        return dst.read_text(encoding="utf-8", errors="replace")
    url = ADT_URL.format(year=storm[4:], num=storm[2:4], basin=BASIN[storm[:2]])
    for i in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "hazardpulse-g1"}),
                                        timeout=60) as r:
                text = r.read().decode("utf-8", "replace")
            dst.write_text(text, encoding="utf-8")
            return text
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None                    # ADT has no history for this storm: it is left out, and counted
            time.sleep(2 * (i + 1))
        except Exception:  # noqa: BLE001 -- retried
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"could not fetch {url}")


def km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(((lon2 - lon1 + 180) % 360) - 180)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(min(1.0, a)))


def hss(pred: np.ndarray, truth: np.ndarray) -> float:
    tp, fp = int((pred & truth).sum()), int((pred & ~truth).sum())
    fn, tn = int((~pred & truth).sum()), int((~pred & ~truth).sum())
    den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return 2.0 * (tp * tn - fp * fn) / den if den else math.nan


def match(shard_dir: str, adt_dir: Path) -> dict[str, np.ndarray]:
    adt_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, list[dict] | None] = {}
    rows = []
    for p in sorted(Path(shard_dir).rglob("goes_shard_*.npz")):
        with np.load(p) as z:
            if "teye" not in z.files:
                raise SystemExit(f"{p}: written before amendment 12b's eye detector")
            for k, st, scan, la, lo, eye, te in zip(z["keys"], z["status"], z["scan"], z["lat"], z["lon"], z["eye"],
                                                     z["teye"]):
                if str(st) != "ok":
                    continue
                storm, hour = str(k).split("_")[:2]
                if storm not in cache:
                    text = fetch_adt(storm, adt_dir)
                    cache[storm] = parse_adt(text) if text else None
                recs = cache[storm]
                if not recs:
                    continue
                ts = g.scan_start(str(scan))
                a = min(recs, key=lambda r: abs((r["time"] - ts).total_seconds()))
                if abs((a["time"] - ts).total_seconds()) > MATCH_MIN * 60 or a["scene"] == "LAND":
                    continue
                rows.append((int(hour[:4]), a["vmax"], a["scene"] == "EYE", a["eye_c"], bool(eye), float(te),
                             km(float(la), float(lo), a["lat"], a["lon"])))
    n_storms = sum(1 for v in cache.values() if v)
    print(f"ADT histories: {n_storms} of {len(cache)} storms", flush=True)
    arr = np.array(rows, dtype=float).reshape(-1, 7)
    return {"year": arr[:, 0], "vmax": arr[:, 1], "adt_eye": arr[:, 2] > 0, "adt_eye_c": arr[:, 3],
            "eye": arr[:, 4] > 0, "teye": arr[:, 5], "dist_km": arr[:, 6]}


def check(m: dict[str, np.ndarray]) -> dict:
    from scipy.stats import spearmanr
    held = m["year"] >= 2024
    e = m["adt_eye"] & np.isfinite(m["teye"]) & (m["adt_eye_c"] < 90)
    weak = m["vmax"] < 50
    got = {"matched": int(len(m["year"])), "hss_2024_2026": hss(m["eye"][held], m["adt_eye"][held]),
           "eye_frac_below_50kt": float(m["eye"][weak].mean()) if weak.any() else math.nan,
           "centre_km_median": float(np.median(m["dist_km"][m["adt_eye"]])) if m["adt_eye"].any() else math.nan,
           "teye_spearman": float(spearmanr(m["teye"][e], m["adt_eye_c"][e]).statistic) if e.sum() > 10 else math.nan,
           "hss_all": hss(m["eye"], m["adt_eye"])}
    by_kt = {}
    for lo, hi in ((0, 50), (50, 64), (64, 83), (83, 96), (96, 113), (113, 200)):
        b = (m["vmax"] >= lo) & (m["vmax"] < hi)
        by_kt[f"{lo}-{hi}"] = {"n": int(b.sum()), "ours": float(m["eye"][b].mean()) if b.any() else None,
                               "adt": float(m["adt_eye"][b].mean()) if b.any() else None}
    passed = {
        "matched": got["matched"] >= GATES["matched"],
        "hss_2024_2026": got["hss_2024_2026"] >= GATES["hss_2024_2026"],
        "eye_frac_below_50kt": got["eye_frac_below_50kt"] <= GATES["eye_frac_below_50kt"],
        "centre_km_median": got["centre_km_median"] <= GATES["centre_km_median"],
        "teye_spearman": got["teye_spearman"] >= GATES["teye_spearman"],
    }   # a NaN compares False: a gate with nothing to measure fails
    return {"measured": got, "gates": GATES, "passed": passed, "all_passed": all(passed.values()),
            "eye_fraction_by_adt_kt": by_kt}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shard_dir")
    ap.add_argument("--adt-dir", default=str(ROOT / ".cache" / "adt"))
    ap.add_argument("--out", default=str(ROOT / "results" / "goes" / "adt_check.json"))
    a = ap.parse_args(argv)
    res = check(match(a.shard_dir, Path(a.adt_dir)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, ok in res["passed"].items():
        print(f"{'PASS' if ok else 'FAIL'} {k}: {res['measured'][k]:.4g} (gate {GATES[k]})")
    print(f"{'ALL PASS' if res['all_passed'] else 'FAILED'}; wrote {a.out}")
    return 0 if res["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
