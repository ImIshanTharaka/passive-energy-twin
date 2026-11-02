"""
s0_check.py - Sub-task 3.1 step 6: gate check S0 (plan §6).

The untuned 2.8 Random Forest (300 trees, 28 raw inputs) is re-run through the NEW Phase 3
harness (surrogate.cv + surrogate.metrics). Its grouped-CV design skill must equal the 2.8
result (outputs/eda/eda_metrics.json, copied into config.yaml) within +/- 0.002 for every
target. A pass proves the harness is a faithful, model-agnostic replacement for eda_lib, so all
later Phase 3 numbers are comparable with Phase 2.8.

    ./venv/bin/python surrogate/s0_check.py

Writes surrogate/outputs/s0/s0_result.json and s0_report.md. Training split only.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from surrogate import cv, metrics as M
from surrogate.config import ALL_TARGETS, CONFIG, PRIMARY_TARGETS
from surrogate.paths import OUT

OUT_DIR = OUT / "s0"


def main() -> int:
    t0 = time.time()
    ref = CONFIG["reference_2_8"]
    tol = ref["tolerance"]
    params = dict(ref["rf_params"])

    df, cols = cv.load_training()
    heat = M.heating_climates(df)
    print(f"Training rows {len(df)}, inputs {len(cols['inputs'])}, heating climates {len(heat)}")
    print(f"RF {params['n_estimators']} trees x {CONFIG['protocol']['grouped_cv']['n_folds']} folds "
          f"x {len(ALL_TARGETS)} targets ...", flush=True)

    oof = cv.run_grouped(df, df[cols["inputs"]], ALL_TARGETS, lambda t: RandomForestRegressor(**params))
    summary, _ = M.summarise(oof, heat)
    summary = summary.set_index("target").loc[ALL_TARGETS]

    rows = []
    for t in ALL_TARGETS:
        new, old = float(summary.at[t, "design_skill"]), float(ref["design_skill"][t])
        rows.append({"target": t, "role": "primary" if t in PRIMARY_TARGETS else "secondary",
                     "skill_2_8": old, "skill_new": new, "diff": new - old, "pass": abs(new - old) <= tol,
                     "pooled_r2": float(summary.at[t, "r2"]),
                     "spearman_median": float(summary.at[t, "spearman_median"]),
                     "top_hit_mean": float(summary.at[t, "top_hit_mean"])})
    res = pd.DataFrame(rows)
    status = "PASS" if res["pass"].all() else "FAIL"
    runtime = time.time() - t0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    record = {"check": "S0", "status": status, "tolerance": tol, "created_utc": datetime.now(timezone.utc).isoformat(),
              "dataset_sha256_training": CONFIG["dataset_sha256"]["training"], "rf_params": params,
              "heating_climates": heat, "runtime_s": round(runtime, 1), "results": rows,
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              **{p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn")}}}
    (OUT_DIR / "s0_result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    lines = ["# S0 reproduction check (Sub-task 3.1 step 6)", "",
             f"**{status}** - untuned 2.8 RF re-run through the Phase 3 harness; design skill must match 2.8 "
             f"within ± {tol}.", "",
             f"Run {record['created_utc'][:19]} UTC, {runtime:.0f} s, dataset sha256 "
             f"`{record['dataset_sha256_training'][:12]}…`, scikit-learn {version('scikit-learn')}.", "",
             "| target | role | 2.8 skill | new skill | difference | result | pooled R² | Spearman (median) | top-10 % hit |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['target']} | {r['role']} | {r['skill_2_8']:.6f} | {r['skill_new']:.6f} | "
                     f"{r['diff']:+.6f} | {'PASS' if r['pass'] else 'FAIL'} | {r['pooled_r2']:.3f} | "
                     f"{r['spearman_median']:.3f} | {r['top_hit_mean']:.3f} |")
    lines += ["", "Spearman and top-10 % hit rate are new Phase 3 metrics (within-climate ranking; "
              "heating targets over the 7 heating climates) - recorded here as the untuned baseline."]
    (OUT_DIR / "s0_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(res[["target", "skill_2_8", "skill_new", "diff", "pass"]].to_string(index=False, float_format="%.6f"))
    print(f"\nS0 {status}  ({runtime:.0f} s) - wrote {OUT_DIR.relative_to(OUT.parent.parent)}/s0_result.json, s0_report.md")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
