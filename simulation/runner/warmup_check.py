#!/usr/bin/env python3
"""
warmup_check.py - Sensitivity of the targets to the warmup limit (Sub-task 2.6).

Two campaign runs (main_B0573_chicago, main_B0600_chicago) reported
"CheckWarmupConvergence: Loads Initialization, Zone=ZONE_F0_CORE did not converge
after 25 warmup days": the loads tolerance was missed narrowly (cooling 4.005 % and
heating 4.23 % vs 4 %). Both are 1-storey ~9,000 m2 designs with a heavily
insulated roof and slab and low infiltration - a very slow core zone (EnergyPlus'
case 1, high effective thermal mass).

This script re-simulates selected runs with a higher Building.Maximum Number of
Warmup Days (only that field differs from the generator) and compares the 10 ML
targets with the campaign result. The campaign data are NOT changed: the reruns
go to outputs/warmup_check/ and the comparison to warmup_check.md there.

USAGE (from the repository root, on the Mac)
-----
  python runner/warmup_check.py                                    # the two flagged runs, 50 days
  python runner/warmup_check.py --run-ids main_B0573_chicago --days 100
  python runner/warmup_check.py --compare-only                     # re-make the table
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_RUNS = ["main_B0573_chicago", "main_B0600_chicago"]
OUT = ROOT / "outputs" / "warmup_check"
ENV_DAYS = "WARMUP_CHECK_DAYS"          # read inside the worker processes


def warmup_build(params: dict, idd: str) -> str:
    """batch_runner builder: the 2.2 generator with only Maximum Number of Warmup Days changed."""
    from idf_generator import build_idf
    idf = build_idf(params, idd)
    bld = idf.idfobjects["BUILDING"][0]
    bld.Maximum_Number_of_Warmup_Days = int(os.environ.get(ENV_DAYS, "50"))
    idf.outputtype = "nocomment"
    return idf.idfstr()


def compare(run_ids: list[str], days: int) -> str:
    import pandas as pd
    from runner import err_parser, output_parser
    output_parser.parse_batch(OUT)
    err_parser.aggregate(OUT)
    new = pd.read_csv(OUT / "results.csv").set_index("run_id")
    old = pd.concat([pd.read_csv(ROOT / "outputs" / b / "results.csv") for b in ("main", "holdout")
                     if (ROOT / "outputs" / b / "results.csv").exists()]).set_index("run_id")
    es = pd.read_csv(OUT / "err_summary.csv").set_index("run_id")
    lines = [f"# Warmup sensitivity check (Maximum Number of Warmup Days 25 → {days})", "",
             "Campaign value (25 days) vs re-run; difference in % of the campaign value.", ""]
    worst = 0.0
    for r in run_ids:
        if r not in new.index or r not in old.index:
            lines.append(f"- {r}: missing ({'re-run' if r not in new.index else 'campaign'} result)")
            continue
        sev = int(es.loc[r, "reported_severe"]) if r in es.index else -1
        lines += [f"## {r}", "", f"Severe errors in the re-run: **{sev}**", "",
                  "| target | 25 days | " + f"{days} days | Δ % |", "|---|---|---|---|"]
        for name, col in output_parser.TARGETS.items():
            a, b = float(old.loc[r, col]), float(new.loc[r, col])
            pct = 100 * (b - a) / a if a else 0.0
            worst = max(worst, abs(pct))
            lines.append(f"| {name} | {a:.4f} | {b:.4f} | {pct:+.3f} |")
        lines.append("")
    lines.append(f"**Largest absolute difference: {worst:.3f} %.**")
    text = "\n".join(lines) + "\n"
    (OUT / "warmup_check.md").write_text(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-ids", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--days", type=int, default=50)
    ap.add_argument("--compare-only", action="store_true")
    a = ap.parse_args(argv)

    if not a.compare_only:
        from climate.climate_features import load_climate_registry
        from runner.batch_runner import RunnerConfig, energyplus_exe, run_batch
        from sampling.sampler import idf_params, load_samples, resolve_idd
        from sampling.transform import load_design_space
        os.environ[ENV_DAYS] = str(a.days)
        space = load_design_space()
        rows = [r for f in ("samples_main.csv", "samples_holdout.csv")
                for r in load_samples(ROOT / "sampling" / f, space) if r["run_id"] in set(a.run_ids)]
        if len(rows) != len(set(a.run_ids)):
            print("unknown run_ids:", sorted(set(a.run_ids) - {r["run_id"] for r in rows}))
            return 2
        registry = load_climate_registry()
        if OUT.exists() and (OUT / "run_log.csv").exists():
            (OUT / "run_log.csv").rename(OUT / "run_log_previous.csv")   # always re-run
        cfg = RunnerConfig(energyplus=energyplus_exe(), idd=resolve_idd(), out_dir=str(OUT),
                           epw_paths={k: str(v.epw_path) for k, v in registry.items()},
                           builder="runner.warmup_check:warmup_build")
        run_batch(rows, lambda r: idf_params(r, space), cfg, workers=min(8, len(rows)))
    print(compare(a.run_ids, a.days))
    return 0


if __name__ == "__main__":
    sys.exit(main())
