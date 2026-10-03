"""Print the corrected programme's results (results/lab_avail) as markdown, straight from the files.

    python scripts/audit_20261001/report_finals.py [results_dir]

Used for docs/TORNADO_MODEL_PROGRAM.md's amendment-8 outcome so no number there is typed by hand.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(d: Path, name: str):
    p = d / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _ci(c, nd=4, signed=False):
    if not c:
        return ""
    f = (lambda v: f"{v:+.{nd}f}") if signed else (lambda v: f"{v:.{nd}f}")
    return f" [{f(c[0])}, {f(c[1])}]"


def main() -> int:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "lab_avail"
    print("| Final run (2025, every storm observation) | AUC [95% CI] | PR-AUC | BSS | n / tornadic |")
    print("|---|---|---|---|---|")
    for run in ("v3_plus_W", "v3_primary", "v3_plus_W_30", "v3_plus_W_90", "v3_plus_W_ef2"):
        f = _read(d, f"final_{run}.json")
        if not f:
            print(f"| {run} | (not run) | | | |")
            continue
        t = f["final_2025"]
        print(f"| {run} ({f['exp']['label']}) | {t['auc']:.4f}{_ci(t['auc_ci'])} | {t['pr_auc']:.4f} | "
              f"{t['bss']:+.4f} | {t['n']:,} / {t['pos']:,} |")
    base = (_read(d, "baselines_storm_60.json") or {}).get("final") or {}
    for k, r in base.items():
        print(f"| baseline {k} | {r['auc']:.4f}{_ci(r.get('auc_ci'))} | {r.get('pr_auc', float('nan')):.4f} | "
              f"{r.get('bss', float('nan')):+.4f} | {r.get('n', 0):,} / {r.get('pos', 0):,} |")
    print()
    print("| Paired by day (2025) | label | dAUC [95% CI] | dBrier [95% CI] |")
    print("|---|---|---|---|")
    for p in sorted(d.glob("compare_*_final_*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        print(f"| {r['b']} - {r['a']} | {r['label']} | {r['delta_auc']:+.4f}{_ci(r['delta_auc_ci'], signed=True)} | "
              f"{r['delta_brier']:+.2e}{_ci(r['delta_brier_ci'], 6, signed=True)} |")
    print()
    print("| At the NWS warnings' false-alarm rate (2025) | NWS POD | model POD | dPOD [95% CI] |")
    print("|---|---|---|---|")
    for p in sorted(d.glob("nws_bar_*_final_*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        print(f"| {r['model']} | {r['nws_pod']:.4f} | {r['model_pod_at_nws_pofd']:.4f} | "
              f"{r['delta_pod']:+.4f}{_ci(r.get('delta_pod_ci'), signed=True)} |")
    st = _read(d, "stress_v3_plus_W_final_storm_60.json")
    if st:
        print()
        print("| Stratum (2025) | n / tornadic | v3_plus_W AUC | ProbTor (Platt) AUC |")
        print("|---|---|---|---|")
        for k, s in st.items():
            m, pt = s.get("v3_plus_W") or {}, s.get("probtor_platt") or {}
            print(f"| {k} | {s.get('n', 0):,} / {s.get('pos', 0):,} | {m.get('auc', float('nan')):.4f}{_ci(m.get('ci'))} | "
                  f"{pt.get('auc', float('nan')):.4f}{_ci(pt.get('ci'))} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
