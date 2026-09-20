"""Variability context for the MIP measures (Section 3.5 of the paper).

Reads the per-case output of eval_mip_contamination.py and reports, for each
direction, the mean and the case-to-case standard deviation of the per-case
difference (retained model minus seed), whether its sign is consistent across
cases, and the change in bright false positives / negatives relative to the
bright ground-truth pixels.

Usage:
    python mip_percase_stats.py <fold> <configuration>
Output:
    results/analysis/mip_percase_stats_fold<k>_<configuration>.json
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import ANALYSIS_DIR  # noqa: E402

FOLD = int(sys.argv[1]) if len(sys.argv) > 1 else 1
CONFIG = sys.argv[2] if len(sys.argv) > 2 else "strict-combined"

SRC = ANALYSIS_DIR / f"mip_contamination_fold{FOLD}_{CONFIG}.json"
OUT = ANALYSIS_DIR / f"mip_percase_stats_fold{FOLD}_{CONFIG}.json"

d = json.loads(SRC.read_text(encoding="utf-8"))
seed = d["seed"]["per_case"]
ret = d[CONFIG]["per_case"]
cases = sorted(seed.keys())

res = {"fold": FOLD, "configuration": CONFIG, "cases": cases, "metrics": {}}
for met in ["contamination_axial", "contamination_coronal", "contamination_sagittal",
            "loss_axial", "loss_coronal", "loss_sagittal"]:
    diffs = [ret[c][met] - seed[c][met] for c in cases]
    res["metrics"][met] = {
        "seed_mean": st.mean(seed[c][met] for c in cases),
        "diff_per_case": dict(zip(cases, diffs)),
        "diff_mean": st.mean(diffs),
        "diff_sd_across_cases": st.stdev(diffs) if len(diffs) > 1 else 0.0,
        "sign_consistent": all(x > 0 for x in diffs) or all(x < 0 for x in diffs),
    }

bt = sum(seed[c]["bright_total"] for c in cases)
res["bright"] = {
    "total": bt,
    "delta_fp": sum(ret[c]["bright_fp"] for c in cases)
                - sum(seed[c]["bright_fp"] for c in cases),
    "delta_fn": sum(ret[c]["bright_fn"] for c in cases)
                - sum(seed[c]["bright_fn"] for c in cases),
}
res["bright"]["delta_fp_pct"] = 100 * res["bright"]["delta_fp"] / bt
res["bright"]["delta_fn_pct"] = 100 * res["bright"]["delta_fn"] / bt

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
for met, m in res["metrics"].items():
    print(f"{met:24s} diff_mean={m['diff_mean']:+.6f}  "
          f"SD={m['diff_sd_across_cases']:.6f}  "
          f"sign_consistent={m['sign_consistent']}")
print(f"bright: dFP={res['bright']['delta_fp_pct']:+.3f}%  "
      f"dFN={res['bright']['delta_fn_pct']:+.3f}%")
print("saved:", OUT)
