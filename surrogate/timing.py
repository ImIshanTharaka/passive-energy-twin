"""
timing.py - Sub-task 3.7: speed of the final surrogates on the Mac (gate S8). Never reads the holdout.

Plan: claude/Phase3.7_Execution_Plan.md decision I6; settings in config.yaml `final_test_3_7.timing`.

In plain words
    How long does the surrogate take for ONE design (the web tool, Phase 5) and how many designs per
    second can it do in a batch (the optimiser, Phase 4)? The timed path is the full one a user
    triggers: the 28 raw inputs -> feature engineering -> all four fitted models -> all 10 outputs.
    Compared with EnergyPlus (16.7 s per run; 2.3 s per run in the parallel campaign).

    load      model load time (unpickling + SHA-256 check), cold
    single    3 warm-up calls, then 500 calls, one design per call (different training designs)
              -> median and 95th percentile in ms; with and without the 90 % intervals
    batch     10,000 and 100,000 designs (training rows resampled), 5 repeats each -> median
    variants  primary picks (gate) and the all-MLP alternative (report, for the 3.8 decision)
    S8        FAIL unless median single < 50 ms AND 100,000-design batch >= 10,000 designs/s (primary)

    caffeinate -i ./venv/bin/python surrogate/timing.py          # official (Mac, on power, apps closed)
    ./venv/bin/python surrogate/timing.py --dev                  # quick code check (fewer calls)

Writes outputs/timing/timing_result.json and timing_report.md (re-runnable: speed is not a holdout result).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from surrogate import explain as X
from surrogate import features as FE
from surrogate import final_test as FT
from surrogate.config import ALL_TARGETS, CONFIG, CONFIG_PATH
from surrogate.paths import OUT

CFG = CONFIG["final_test_3_7"]["timing"]
S8 = CONFIG["gate"]["S8"]
SEED = int(CFG["seed"])
DEV = {"single_calls": 40, "batch_sizes": [2000, 10000], "batch_repeats": 2}


def out_dir(dev: bool) -> Path:
    d = OUT / ("timing_dev" if dev else "timing")
    d.mkdir(parents=True, exist_ok=True)
    return d


def settings(dev: bool) -> dict:
    s = {"warmup_calls": int(CFG["warmup_calls"]), "single_calls": int(CFG["single_calls"]),
         "batch_sizes": [int(v) for v in CFG["batch_sizes"]], "batch_repeats": int(CFG["batch_repeats"])}
    return {**s, **DEV} if dev else s


class Timed:
    """The deployable prediction path for one variant: raw inputs (climate_key + 28) -> 10 outputs."""

    def __init__(self, fm: X.FinalModels, iv: dict, variant: str):
        self.fm, self.iv, self.variant = fm, iv, variant
        self.table = FT.index_table_with()

    def predict(self, raw: pd.DataFrame) -> pd.DataFrame:
        # Features are engineered once per chunk and shared by the fitted objects (FinalModels.predict
        # re-engineers them per object: same numbers, checked in test_final_test, but ~4x the feature time).
        return FT.predict_variant(self.fm, raw, self.table, self.variant)

    def predict_with_intervals(self, raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        pred = self.predict(raw)
        Xm = FT.matrices_with(raw, self.table)["C-G+I"]
        bounds = {t: self.iv[t]["standard"].interval(Xm, pred[t].to_numpy(float)) for t in ALL_TARGETS}
        return pred, bounds


def design_pool() -> pd.DataFrame:
    from surrogate import cv
    df, _ = cv.load_training()
    return df[["climate_key", *FE.RAW_INPUTS]].reset_index(drop=True)


def time_single(fn, pool: pd.DataFrame, calls: int, warmup: int) -> dict:
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(pool))
    for i in range(warmup):
        fn(pool.iloc[[order[i]]])
    ms = []
    for k in range(calls):
        i = order[(warmup + k) % len(pool)]
        row = pool.iloc[[i]]
        t0 = time.perf_counter()
        fn(row)
        ms.append(1e3 * (time.perf_counter() - t0))
    ms = np.asarray(ms)
    return {"calls": calls, "median_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95)),
            "mean_ms": float(ms.mean()), "max_ms": float(ms.max())}


def time_batch(fn, pool: pd.DataFrame, size: int, repeats: int) -> dict:
    rng = np.random.default_rng(SEED + size)
    batch = pool.iloc[rng.integers(0, len(pool), size)].reset_index(drop=True)
    fn(batch.iloc[:min(size, 100)])                       # warm-up
    secs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(batch)
        secs.append(time.perf_counter() - t0)
    n_out = len(out[0] if isinstance(out, tuple) else out)
    if n_out != size:
        raise AssertionError(f"batch of {size} returned {n_out} rows")
    med = float(np.median(secs))
    return {"designs": size, "repeats": repeats, "median_s": med, "designs_per_s": size / med,
            "us_per_design": 1e6 * med / size, "all_s": [float(s) for s in secs]}


def evaluate_s8(single_median_ms: float, batch_per_s: float) -> dict:
    ok_s = single_median_ms < float(S8["single_design_ms_max"])
    ok_b = batch_per_s >= float(S8["batch_designs_per_s_min"])
    return {"status": "PASS" if ok_s and ok_b else "FAIL", "severity": S8["severity"],
            "single_median_ms": single_median_ms, "single_max_ms": float(S8["single_design_ms_max"]),
            "batch_designs_per_s": batch_per_s, "batch_min_per_s": float(S8["batch_designs_per_s_min"])}


def machine() -> dict:
    info = {"platform": platform.platform(), "processor": platform.processor(), "machine": platform.machine(),
            "cpu_count": os.cpu_count()}
    try:
        import subprocess
        info["cpu_brand"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                           text=True).stdout.strip() or None
    except OSError:
        info["cpu_brand"] = None
    from surrogate.mlp import _torch
    info["torch_threads"] = _torch().get_num_threads()
    return info


def run(dev: bool, log=print) -> dict:
    s = settings(dev)
    t0 = time.perf_counter()
    fm = X.final_models(dev=False)
    load_models_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    iv, _ = FT.load_intervals(dev)
    load_iv_s = time.perf_counter() - t0
    log(f"load: models {load_models_s:.2f} s, interval models {load_iv_s:.2f} s")
    if FT.index_table_with().shape[0] != 10:
        raise AssertionError("timing uses the 10 training climates only")
    m = machine()
    if m["torch_threads"] != int(CFG["torch_threads"]):
        raise AssertionError(f"torch threads {m['torch_threads']} != {CFG['torch_threads']}")
    pool = design_pool()
    res = {"variants": {}}
    for v in CFG["variants"]:
        tm = Timed(fm, iv, v)
        r = {"single": time_single(tm.predict, pool, s["single_calls"], s["warmup_calls"]),
             "single_with_intervals": time_single(tm.predict_with_intervals, pool, s["single_calls"],
                                                  s["warmup_calls"]),
             "batch": [time_batch(tm.predict, pool, n, s["batch_repeats"]) for n in s["batch_sizes"]]}
        r["batch_with_intervals"] = [time_batch(tm.predict_with_intervals, pool, s["batch_sizes"][0],
                                                s["batch_repeats"])]
        res["variants"][v] = r
        b = r["batch"][-1]
        log(f"{v:8s} single median {r['single']['median_ms']:.1f} ms (p95 {r['single']['p95_ms']:.1f}) | "
            f"with intervals {r['single_with_intervals']['median_ms']:.1f} ms | batch {b['designs']:,}: "
            f"{b['designs_per_s']:,.0f} designs/s")
    # where the time goes (primary): features once, then each fitted object on ready-made features
    table = FT.index_table_with()
    n_small = max(20, s["single_calls"] // 5)
    t_feat = time_single(lambda r: FT.matrices_with(r, table), pool, n_small, s["warmup_calls"])
    per_obj, per_obj_batch = {}, {}
    big = pool.iloc[np.random.default_rng(SEED).integers(0, len(pool), s["batch_sizes"][0])].reset_index(drop=True)
    Xbig = FT.matrices_with(big, table)
    for name in fm.objects:
        per_obj[name] = time_single(lambda r, n=name: fm._predict_chunk(n, FT.matrices_with(r, table)), pool,
                                    n_small, s["warmup_calls"])["median_ms"] - t_feat["median_ms"]
        t0 = time.perf_counter()
        fm._predict_chunk(name, Xbig)
        per_obj_batch[name] = time.perf_counter() - t0
    prim = res["variants"]["primary"]
    gate_batch = next(b for b in prim["batch"] if b["designs"] == (int(CFG["s8"]["batch_size_for_gate"])
                                                                  if not dev else s["batch_sizes"][-1]))
    s8 = evaluate_s8(prim["single"]["median_ms"], gate_batch["designs_per_s"])
    ep = CFG["energyplus_s_per_run"]
    speedup = {"single_vs_energyplus_run": float(ep["single"]) / (prim["single"]["median_ms"] / 1e3),
               "batch_vs_campaign_throughput": float(ep["campaign_throughput"]) * gate_batch["designs_per_s"],
               "energyplus_s_per_run": ep}
    result = {"official": not dev, "dev": dev, "created_utc": datetime.now(timezone.utc).isoformat(),
              "config_md5": FT.md5_text(CONFIG_PATH), "git": FT.git_state(), "settings": s, "machine": m,
              "environment": FT.environment(), "load_s": {"models": load_models_s, "intervals": load_iv_s},
              "single_ms_per_object_primary": per_obj, "single_ms_feature_engineering": t_feat["median_ms"],
              "batch_s_per_object_primary": {"designs": s["batch_sizes"][0], **per_obj_batch},
              **res, "speedup": speedup, "s8": s8}
    d = out_dir(dev)
    (d / "timing_result.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    write_report(d, result)
    log(f"S8 {s8['status']}: single {s8['single_median_ms']:.1f} ms (< {s8['single_max_ms']:.0f}), batch "
        f"{s8['batch_designs_per_s']:,.0f} designs/s (>= {s8['batch_min_per_s']:,.0f}) | speed-up vs EnergyPlus "
        f"{speedup['single_vs_energyplus_run']:,.0f}x single, {speedup['batch_vs_campaign_throughput']:,.0f}x batch")
    return result


def write_report(d: Path, r: dict) -> Path:
    L = [f"# Surrogate speed (Sub-task 3.7){' — DEV (code check only)' if r['dev'] else ''}", "",
         f"{r['created_utc']} · {r['machine'].get('cpu_brand') or r['machine']['processor']} · "
         f"{r['machine']['cpu_count']} cores · torch threads {r['machine']['torch_threads']} · "
         f"commit `{(r['git'] or {}).get('commit')}`", "",
         f"Model load: {r['load_s']['models']:.2f} s (+ interval models {r['load_s']['intervals']:.2f} s).", "",
         "| Variant | One design, median (ms) | p95 (ms) | With 90 % intervals, median (ms) | "
         + " | ".join(f"Batch {b['designs']:,} (designs/s)" for b in r["variants"]["primary"]["batch"]) + " |",
         "|---|---|---|---|" + "---|" * len(r["variants"]["primary"]["batch"])]
    for v, x in r["variants"].items():
        L.append(f"| {v} | {x['single']['median_ms']:.1f} | {x['single']['p95_ms']:.1f} | "
                 f"{x['single_with_intervals']['median_ms']:.1f} | "
                 + " | ".join(f"{b['designs_per_s']:,.0f}" for b in x["batch"]) + " |")
    L += ["", "Single-design time per fitted object (primary, median ms): "
          + ", ".join(f"{k} {v:.1f}" for k, v in r["single_ms_per_object_primary"].items())
          + f"; feature engineering alone {r['single_ms_feature_engineering']:.1f} ms.", "",
          f"Batch of {r['batch_s_per_object_primary']['designs']:,} designs, seconds per fitted object (features ready): "
          + ", ".join(f"{k} {v:.2f}" for k, v in r["batch_s_per_object_primary"].items() if k != "designs") + ".", "",
          f"Speed-up vs EnergyPlus: {r['speedup']['single_vs_energyplus_run']:,.0f}× for one design "
          f"(16.7 s per run); {r['speedup']['batch_vs_campaign_throughput']:,.0f}× in batch "
          f"(2.3 s per run in the parallel campaign).", "",
          f"**S8 {r['s8']['status']}** (one design < {r['s8']['single_max_ms']:.0f} ms and batch ≥ "
          f"{r['s8']['batch_min_per_s']:,.0f} designs/s, primary picks).", ""]
    p = d / "timing_report.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sub-task 3.7 surrogate timing (gate S8)")
    ap.add_argument("--dev", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    run(a.dev)
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
