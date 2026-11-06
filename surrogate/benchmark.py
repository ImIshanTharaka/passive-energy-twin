#!/usr/bin/env python3
"""
benchmark.py - Sub-task 3.3: model benchmark and selection (plan M4, M10; claude/Phase3.3_Execution_Plan.md E1-E6).

Stages (run in this order; each writes its own files, `select` combines them):

    probe     timing test: one fold per model (defaults + mid-range settings, GPR on 63 and top-15
              features) -> timing_probe.json with run-time estimates and the E5 fallback decision
    untuned   run A: 5 models x {R, C-G} x 3 primary targets, default settings, grouped CV;
              then the D5 transfer check (E3) -> untuned_table.csv, feature_sets.json
    d5        only if D5 triggers for a model: 3.2 ablation (8 sets, defaults) for that model (--model)
    tuned     run C: tuned settings (tuning/best_params.json; GPR defaults, E5) scored by grouped CV
              (seed 42), fresh folds (seed 7, E4) and LOCO -> tuned_table.csv, oof_tuned_grouped.csv.gz
    secondary run D: 7 secondary targets, tuned LightGBM + best other family -> secondary_table.csv
    select    M10 per target (primary provisional, secondary final), CV-stage S1/S2,
              report + figures -> selection.json, benchmark_report.md, *.png/.pdf

    ./venv/bin/python surrogate/benchmark.py --stage probe
    ./venv/bin/python surrogate/benchmark.py --stage untuned
    ./venv/bin/python surrogate/benchmark.py --stage tuned [--gpr-top-k 15]
    ./venv/bin/python surrogate/benchmark.py --stage secondary
    ./venv/bin/python surrogate/benchmark.py --stage select
    ./venv/bin/python surrogate/benchmark.py --stage untuned --dev     # small: 25 % of designs, small models

Writes surrogate/outputs/benchmark/ (dev: benchmark_dev/). Training split only.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from surrogate import cv, features as FE, metrics as M, models
from surrogate.config import CONFIG, PRIMARY_TARGETS, SECONDARY_TARGETS, ALL_TARGETS
from surrogate.paths import OUT

CFG = CONFIG["benchmark_3_3"]
SEL = CONFIG["selection"]
GATE = CONFIG["gate"]
SHORT = {"heating_demand_kwh_m2": "Heating", "cooling_demand_kwh_m2": "Cooling", "eui_kwh_m2": "EUI",
         "peak_heating_w_m2": "Peak heating", "peak_cooling_w_m2": "Peak cooling",
         "zone_cooling_core_kwh_m2": "Zone core", "zone_cooling_south_kwh_m2": "Zone south",
         "zone_cooling_east_kwh_m2": "Zone east", "zone_cooling_north_kwh_m2": "Zone north",
         "zone_cooling_west_kwh_m2": "Zone west"}
# categorical slots 1-5 of the validated palette, fixed per model (entity, not rank); contrast WARN for
# slots 3-5 -> every chart also uses a marker shape per model and direct labels / tables.
COLORS = {"rf": "#2a78d6", "lgbm": "#eb6834", "xgb": "#1baf7a", "gpr": "#eda100", "mlp": "#e87ba4"}
MARKERS = {"rf": "o", "lgbm": "s", "xgb": "^", "gpr": "D", "mlp": "v"}
GROUPED_COLS = ("design_skill", "r2", "mae", "rmse", "cv_rmse", "spearman_median", "top_hit_mean")
LOCO_COLS = ("median_abs_bias_pct", "max_abs_bias_pct", "worst_climate", "mape_pct", "bias_removed_r2",
             "spearman_median", "top_hit_mean")
DEV_DESIGN_SHARE = 0.25


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8a8a85")
    ax.tick_params(colors="#55554f", labelsize=8)
    ax.grid(axis="y", color="#e4e3dd", linewidth=0.6)
    ax.set_axisbelow(True)


def out_dir(dev: bool) -> Path:
    d = OUT / ("benchmark_dev" if dev else "benchmark")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── data ────────────────────────────────────────────────────────────────────────────────────

def load(dev: bool):
    """(df, F, heat). Dev: a fixed 25 % sample of designs (all their climates) for quick code checks."""
    df, _ = cv.load_training()
    if dev:
        ids = np.sort(df["design_id"].astype(str).unique())
        keep = set(np.random.default_rng(0).choice(ids, int(DEV_DESIGN_SHARE * len(ids)), replace=False))
        df = df[df["design_id"].astype(str).isin(keep)].reset_index(drop=True)
    heat = M.heating_climates(df)
    return df, FE.engineer(df), heat


def dev_params(model: str, params: dict, dev: bool) -> dict:
    from surrogate.tune import dev_shrink
    if not dev:
        return dict(params)
    p = dev_shrink(model, params)
    if model == "gpr":
        p["top_k"] = 5
    return p


# ── scoring helpers ─────────────────────────────────────────────────────────────────────────

def oof_for(model: str, params_by_target: dict, df, X, targets: list[str], folds, scheme: str) -> pd.DataFrame:
    """Out-of-fold predictions. Single-output models may have different settings per target;
    the MLP is one multi-output net (params_by_target[None]) trained on all 10 targets."""
    if model in models.MULTI_OUTPUT:
        oof = cv.cross_val_predict_multi(df, X, ALL_TARGETS, models.make_multi(model, params_by_target[None]),
                                         folds, scheme)
        return oof[oof["target"].isin(targets)].reset_index(drop=True)
    parts = [cv.cross_val_predict(df, X, [t], models.make(model, params_by_target[t]), folds, scheme) for t in targets]
    return pd.concat(parts, ignore_index=True)


def fit_seconds_per_fold(oof: pd.DataFrame) -> pd.Series:
    return oof.groupby(["target", "fold"])["fit_s"].first().groupby("target").mean()


def grouped_rows(oof, heat, prefix="grouped") -> tuple[dict, pd.DataFrame]:
    summ, pc = M.summarise(oof, heat)
    fit = fit_seconds_per_fold(oof)
    rows = {}
    for _, r in summ.iterrows():
        rows[r["target"]] = {f"{prefix}_{c}": float(r[c]) for c in GROUPED_COLS}
        rows[r["target"]][f"{prefix}_fit_s_per_fold"] = float(fit[r["target"]])
    return rows, pc


def loco_rows(oof, heat) -> tuple[dict, pd.DataFrame]:
    summ, tab = M.loco_summary(oof, heat)
    rows = {r["target"]: {f"loco_{c}": r[c] for c in LOCO_COLS} for _, r in summ.iterrows()}
    return rows, tab


def environment() -> dict:
    pkgs = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost", "optuna", "torch", "matplotlib")
    env = {"python": platform.python_version(), "platform": platform.platform()}
    for p in pkgs:
        try:
            env[p] = version(p)
        except Exception:                               # noqa: BLE001
            env[p] = None
    return env


def record_stage(d: Path, stage: str, info: dict, dev: bool):
    path = d / "benchmark_result.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"stages": {}}
    data["stages"][stage] = {"official": not dev, "dev": dev, "created_utc": datetime.now(timezone.utc).isoformat(),
                             "dataset_sha256_training": CONFIG["dataset_sha256"]["training"],
                             "environment": environment(), **info}
    path.write_text(json.dumps(data, indent=2, default=float), encoding="utf-8")


# ── D5 transfer check (plan E3) ─────────────────────────────────────────────────────────────

def d5_check(untuned: pd.DataFrame) -> dict:
    """Per model in d5.models: mean untuned grouped skill over primary targets on the selected set vs R.
    triggered = the selected set does NOT score higher than R."""
    fs, ref = CFG["feature_set"], CFG["reference_set"]
    out = {}
    for m in CFG["d5"]["models"]:
        g = untuned[untuned["model"] == m]
        s_fs = float(g.loc[g["feature_set"] == fs, "grouped_design_skill"].mean())
        s_ref = float(g.loc[g["feature_set"] == ref, "grouped_design_skill"].mean())
        out[m] = {fs: s_fs, ref: s_ref, "triggered": not (s_fs > s_ref)}
    return out


# ── M10 selection (pre-registered) ──────────────────────────────────────────────────────────

def select_target(cands: pd.DataFrame, target: str) -> dict:
    """M10 for one target. cands: one row per candidate with model, feature_set, grouped_design_skill
    (and loco_mape_pct for primary targets)."""
    primary = target in PRIMARY_TARGETS
    bar = SEL["primary_skill_min"] if primary else SEL["secondary_skill_min"]
    c = cands.reset_index(drop=True)
    passing = c[c["grouped_design_skill"] >= bar]
    rec = {"target": target, "kind": "primary" if primary else "secondary", "skill_bar": bar,
           "n_candidates": len(c), "n_passing": len(passing),
           "status": CFG["selection_status"]["primary" if primary else "secondary"]}
    if passing.empty:
        best = c.loc[c["grouped_design_skill"].idxmax()]
        rec.update(model=best["model"], feature_set=best["feature_set"], rule="no candidate passes: best grouped skill",
                   gate_fail=True)
        return rec
    if primary:
        key, best_val = "loco_mape_pct", passing["loco_mape_pct"].min()
        tied = passing[passing[key] <= best_val + SEL["tie_mape_points"] + 1e-12]
        rule = f"lowest LOCO MAPE (ties within {SEL['tie_mape_points']} points -> LightGBM)"
        best = passing.loc[passing[key].idxmin()]
    else:
        key, best_val = "grouped_design_skill", passing["grouped_design_skill"].max()
        tied = passing[passing[key] >= best_val - SEL["tie_skill"] - 1e-12]
        rule = f"highest grouped skill (ties within {SEL['tie_skill']} -> LightGBM)"
        best = passing.loc[passing[key].idxmax()]
    chosen = best
    if SEL["tie_break"] == "lightgbm" and (tied["model"] == "lgbm").any() and best["model"] != "lgbm":
        chosen = tied[tied["model"] == "lgbm"].iloc[0]
        rule += "; LightGBM within the tie margin"
    rec.update(model=chosen["model"], feature_set=chosen["feature_set"], rule=rule, gate_fail=False,
               value=float(chosen[key]), best_model_by_value=best["model"], best_value=float(best_val),
               tied_models=sorted(tied["model"].tolist()))
    return rec


def gate_cv_stage(sel: dict, table: pd.DataFrame) -> dict:
    """S1/S2 on cross-validation for the selected candidates (binding gate is evaluated in 3.7)."""
    out = {}
    for t, r in sel.items():
        row = table[(table["target"] == t) & (table["model"] == r["model"]) & (table["feature_set"] == r["feature_set"])].iloc[0]
        if t in PRIMARY_TARGETS:
            ok = row["grouped_design_skill"] >= GATE["S1"]["design_skill_min"] and row["grouped_r2"] >= GATE["S1"]["pooled_r2_min"]
            out[t] = {"check": "S1", "design_skill": float(row["grouped_design_skill"]), "pooled_r2": float(row["grouped_r2"]),
                      "result": "PASS" if ok else "FAIL"}
        else:
            ok = row["grouped_design_skill"] >= GATE["S2"]["design_skill_min"]
            out[t] = {"check": "S2", "design_skill": float(row["grouped_design_skill"]), "result": "PASS" if ok else "WARN"}
    return out


# ── stages ──────────────────────────────────────────────────────────────────────────────────

def stage_probe(dev: bool, log=print) -> dict:
    """One fold (fold 0), EUI: default and mid-range settings per model; GPR also with top-15 features."""
    df, F, heat = load(dev)
    X = F[FE.feature_set(CFG["feature_set"])]
    f = cv.grouped_folds(df)[0]
    y = df["eui_kwh_m2"].to_numpy()
    Y = df[ALL_TARGETS].to_numpy()
    groups = df["design_id"].astype(str).to_numpy()
    mid = {"rf": {"n_estimators": 500, "max_features": 0.5, "min_samples_leaf": 2, "max_depth": None, "max_samples": 0.75},
           "lgbm": {"n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 32, "min_child_samples": 20, "subsample": 0.8,
                    "colsample_bytree": 0.65, "reg_alpha": 1e-3, "reg_lambda": 1e-3},
           "xgb": {"n_estimators": 1000, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 3.0, "subsample": 0.8,
                   "colsample_bytree": 0.65, "reg_alpha": 1e-3, "reg_lambda": 1e-3},
           "mlp": {"n_layers": 3, "width": 256, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 128}}
    configs = [(m, "default", dev_params(m, models.default_params(m), dev)) for m in models.MODELS]
    configs += [(m, "mid", dev_params(m, models.tuned_params(m, p), dev)) for m, p in mid.items()]
    configs.append(("gpr", "top15", dev_params("gpr", {**models.default_params("gpr"), "top_k": CFG["gpr_fallback"]["top_k"]}, dev)))
    times = {}
    for m, label, p in configs:
        t0 = time.perf_counter()
        if m in models.MULTI_OUTPUT:
            est = models.make_multi(m, p)()
            est.fit(X.iloc[f.train], Y[f.train], groups=groups[f.train])
        else:
            est = models.make(m, p)("eui_kwh_m2")
            est.fit(X.iloc[f.train], y[f.train])
        fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        est.predict(X.iloc[f.test])
        pred = time.perf_counter() - t0
        times[f"{m}_{label}"] = {"fit_s": round(fit, 2), "predict_s": round(pred, 3)}
        log(f"  {m:5s} {label:8s} fit {fit:8.1f} s   predict {pred:6.3f} s")
    t = {k: v["fit_s"] for k, v in times.items()}
    nP, nS, B = len(PRIMARY_TARGETS), len(SECONDARY_TARGETS), CFG["tuning"]["trials"]
    gpr_limit = 60 * CFG["gpr_fallback"]["max_fit_minutes"]
    fallback = t["gpr_default"] > gpr_limit
    gpr_fit = t["gpr_top15"] if fallback else t["gpr_default"]
    est_h = {
        "A_untuned": (sum(t[f"{m}_default"] for m in ("rf", "xgb", "lgbm", "gpr")) * nP * 5 * 2 + t["mlp_default"] * 5 * 2
                      - (t["gpr_default"] - gpr_fit) * nP * 5 * 2) / 3600,
        "B_tuning": (sum(B[m] * nP * 5 * t[f"{m}_mid"] for m in ("rf", "xgb", "lgbm")) + B["mlp"] * 5 * t["mlp_mid"]) / 3600,
        "C_tuned": (sum(t[f"{m}_mid"] * nP * 20 for m in ("rf", "xgb", "lgbm")) + t["mlp_mid"] * 20 + gpr_fit * nP * 20 * 1.1) / 3600,
        "D_secondary": (B["lgbm"] * nS * 5 * t["lgbm_mid"] + nS * 5 * t["lgbm_mid"]) / 3600,
    }
    est_h["total"] = sum(est_h.values())
    info = {"fold": "grouped fold 0", "target": "eui_kwh_m2", "train_rows": int(len(f.train)), "times": times,
            "gpr_fallback_rule_s": gpr_limit, "gpr_fallback_needed": bool(fallback),
            "estimate_hours": {k: round(v, 2) for k, v in est_h.items()},
            "note": "rough: mid-range settings stand for an average tuning trial; real trials vary 0.2x-3x"}
    d = out_dir(dev)
    (d / "timing_probe.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    record_stage(d, "probe", {"gpr_fallback_needed": bool(fallback)}, dev)
    log(f"GPR full fit {t['gpr_default']:.0f} s vs limit {gpr_limit} s -> fallback {'NEEDED (top-15)' if fallback else 'not needed'}")
    log("estimated hours: " + ", ".join(f"{k} {v:.1f}" for k, v in est_h.items()))
    return info


def stage_untuned(dev: bool, log=print):
    df, F, heat = load(dev)
    folds = cv.grouped_folds(df)
    rows, pcs = [], []
    for m in models.MODELS:
        p = dev_params(m, models.default_params(m), dev)
        pbt = {None: p} if m in models.MULTI_OUTPUT else {t: p for t in PRIMARY_TARGETS}
        for fs in (CFG["reference_set"], CFG["feature_set"]):
            X = F[FE.feature_set(fs)]
            t0 = time.time()
            oof = oof_for(m, pbt, df, X, PRIMARY_TARGETS, folds, "grouped")
            g, pc = grouped_rows(oof, heat)
            for t in PRIMARY_TARGETS:
                rows.append({"model": m, "feature_set": fs, "tuned": False, "target": t, "n_features": X.shape[1], **g[t]})
            pcs.append(pc.assign(model=m, feature_set=fs, tuned=False))
            log(f"  {m:5s} {fs:4s} untuned  " + "  ".join(f"{SHORT[t]} {g[t]['grouped_design_skill']:.3f}" for t in PRIMARY_TARGETS)
                + f"   {time.time() - t0:6.1f} s")
    table = pd.DataFrame(rows)
    d = out_dir(dev)
    table.to_csv(d / "untuned_table.csv", index=False)
    pd.concat(pcs, ignore_index=True).to_csv(d / "untuned_per_climate.csv", index=False)
    d5 = d5_check(table)
    fsets = {m: CFG["feature_set"] for m in models.MODELS}
    (d / "feature_sets.json").write_text(json.dumps({"feature_sets": fsets, "d5": d5,
                                                     "note": "D5 triggered models get their own set from --stage d5"},
                                                    indent=2), encoding="utf-8")
    record_stage(d, "untuned", {"d5": d5, "params": {m: dev_params(m, models.default_params(m), dev) for m in models.MODELS}}, dev)
    for m, r in d5.items():
        log(f"  D5 {m}: C-G {r[CFG['feature_set']]:.4f} vs R {r[CFG['reference_set']]:.4f} -> "
            f"{'TRIGGERED: run --stage d5 --model ' + m if r['triggered'] else 'ok (C-G transfers)'}")
    return table, d5


def stage_d5(model: str, dev: bool, log=print):
    from surrogate.ablation import choose_winner
    df, F, heat = load(dev)
    folds = cv.grouped_folds(df)
    p = dev_params(model, models.default_params(model), dev)
    pbt = {None: p} if model in models.MULTI_OUTPUT else {t: p for t in PRIMARY_TARGETS}
    rows = []
    for fs in FE.FEATURE_SETS:
        X = F[FE.feature_set(fs)]
        g, _ = grouped_rows(oof_for(model, pbt, df, X, PRIMARY_TARGETS, folds, "grouped"), heat)
        rows += [{"feature_set": fs, "model": model, "target": t, "n_features": X.shape[1], **g[t]} for t in PRIMARY_TARGETS]
        log(f"  {model} {fs:4s} " + "  ".join(f"{SHORT[t]} {g[t]['grouped_design_skill']:.3f}" for t in PRIMARY_TARGETS))
    table = pd.DataFrame(rows)
    scores, winner = choose_winner(table)
    d = out_dir(dev)
    table.to_csv(d / f"d5_ablation_{model}.csv", index=False)
    path = d / "feature_sets.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["feature_sets"][model] = winner
    data.setdefault("d5_winners", {})[model] = {"winner": winner, "scores": scores.to_dict(orient="records")}
    path.write_text(json.dumps(data, indent=2, default=float), encoding="utf-8")
    record_stage(d, f"d5_{model}", {"winner": winner}, dev)
    log(f"  D5 ablation for {model}: winner {winner}")
    return winner


def feature_sets(dev: bool) -> dict:
    path = out_dir(dev) / "feature_sets.json"
    if not path.exists():
        return {m: CFG["feature_set"] for m in models.MODELS}
    return json.loads(path.read_text(encoding="utf-8"))["feature_sets"]


def tuned_params_for(model: str, targets: list[str], dev: bool, gpr_top_k=None, allow_missing=False,
                     feature_set: str | None = None) -> dict:
    """{target: params} (or {None: params} for the MLP) from tuning/best_params.json; GPR = defaults (E5)."""
    from surrogate.tune import load_best, study_name, MULTI_STUDY
    if model == "gpr":
        p = models.default_params("gpr")
        if gpr_top_k:
            p["top_k"] = int(gpr_top_k)
        p = dev_params("gpr", p, dev)
        return {t: p for t in targets}
    best = load_best(dev)
    if model in models.MULTI_OUTPUT:
        keys = {None: study_name(model, MULTI_STUDY)}
    else:
        keys = {t: study_name(model, t) for t in targets}
    out = {}
    for t, k in keys.items():
        if k not in best:
            if allow_missing:
                out[t] = dev_params(model, models.default_params(model), dev)
                continue
            raise SystemExit(f"no tuned settings for {k}: run surrogate/tune.py --model {model}"
                             f"{' --targets secondary' if t in SECONDARY_TARGETS else ''} first")
        if feature_set and best[k].get("feature_set") != feature_set:
            msg = (f"{k} was tuned on {best[k].get('feature_set')} but {model} is scored on {feature_set} "
                   "(feature_sets.json); re-tune after the D5 ablation")
            if not dev:
                raise SystemExit(msg)
            print(f"  [DEV] warning: {msg}", flush=True)
        out[t] = best[k]["params"]
    return out


def stage_tuned(dev: bool, gpr_top_k=None, log=print):
    df, F, heat = load(dev)
    fsets = feature_sets(dev)
    folds = {"grouped": cv.grouped_folds(df), "fresh": cv.grouped_folds(df, seed=CFG["fresh_fold_seed"]), "loco": cv.loco_folds(df)}
    rows, pcs, oofs, params_used = [], [], [], {}
    for m in models.MODELS:
        X = F[FE.feature_set(fsets[m])]
        pbt = tuned_params_for(m, PRIMARY_TARGETS, dev, gpr_top_k, allow_missing=dev, feature_set=fsets[m])
        params_used[m] = {str(k): v for k, v in pbt.items()}
        res = {t: {"model": m, "feature_set": fsets[m], "tuned": m in models.TUNABLE, "target": t, "n_features": X.shape[1]}
               for t in PRIMARY_TARGETS}
        t0 = time.time()
        oof = oof_for(m, pbt, df, X, PRIMARY_TARGETS, folds["grouped"], "grouped")
        g, pc = grouped_rows(oof, heat)
        oofs.append(oof.assign(model=m))
        pcs.append(pc.assign(model=m, scheme="grouped"))
        fr, _ = grouped_rows(oof_for(m, pbt, df, X, PRIMARY_TARGETS, folds["fresh"], "grouped"), heat, prefix="fresh")
        lo, tab = loco_rows(oof_for(m, pbt, df, X, PRIMARY_TARGETS, folds["loco"], "loco"), heat)
        pcs.append(tab.assign(model=m, scheme="loco"))
        for t in PRIMARY_TARGETS:
            res[t].update(g[t])
            res[t].update(fr[t])
            res[t].update(lo[t])
            res[t]["optimism_skill"] = res[t]["grouped_design_skill"] - res[t]["fresh_design_skill"]
        rows += list(res.values())
        log(f"  {m:5s} tuned  skill " + "  ".join(f"{SHORT[t]} {g[t]['grouped_design_skill']:.3f}" for t in PRIMARY_TARGETS)
            + "   LOCO MAPE " + "  ".join(f"{lo[t]['loco_mape_pct']:.1f}%" for t in PRIMARY_TARGETS)
            + f"   {time.time() - t0:6.0f} s")
    table = pd.DataFrame(rows)
    d = out_dir(dev)
    table.to_csv(d / "tuned_table.csv", index=False)
    pd.concat(pcs, ignore_index=True).to_csv(d / "tuned_per_climate.csv", index=False)
    pd.concat(oofs, ignore_index=True).to_csv(d / "oof_tuned_grouped.csv.gz", index=False)
    record_stage(d, "tuned", {"gpr_top_k": gpr_top_k, "feature_sets": fsets, "params": params_used}, dev)
    return table


def best_other(tuned: pd.DataFrame) -> str:
    s = tuned[tuned["model"] != "lgbm"].groupby("model")["grouped_design_skill"].mean()
    return str(s.idxmax())


def stage_secondary(dev: bool, gpr_top_k=None, log=print):
    d = out_dir(dev)
    tuned = pd.read_csv(d / "tuned_table.csv")
    other = best_other(tuned)
    df, F, heat = load(dev)
    fsets = feature_sets(dev)
    folds = cv.grouped_folds(df)
    rows, pcs, oofs = [], [], []
    for m in ("lgbm", other):
        X = F[FE.feature_set(fsets[m])]
        pbt = tuned_params_for(m, SECONDARY_TARGETS, dev, gpr_top_k, allow_missing=dev, feature_set=fsets[m])
        t0 = time.time()
        oof = oof_for(m, pbt, df, X, SECONDARY_TARGETS, folds, "grouped")
        g, pc = grouped_rows(oof, heat)
        rows += [{"model": m, "feature_set": fsets[m], "tuned": m in models.TUNABLE, "target": t,
                  "n_features": X.shape[1], **g[t]} for t in SECONDARY_TARGETS]
        pcs.append(pc.assign(model=m))
        oofs.append(oof.assign(model=m))
        log(f"  {m:5s} secondary  " + "  ".join(f"{SHORT[t]} {g[t]['grouped_design_skill']:.3f}" for t in SECONDARY_TARGETS)
            + f"   {time.time() - t0:6.0f} s")
    table = pd.DataFrame(rows)
    table.to_csv(d / "secondary_table.csv", index=False)
    pd.concat(pcs, ignore_index=True).to_csv(d / "secondary_per_climate.csv", index=False)
    pd.concat(oofs, ignore_index=True).to_csv(d / "oof_secondary_grouped.csv.gz", index=False)
    record_stage(d, "secondary", {"models": ["lgbm", other], "best_other": other}, dev)
    return table


def stage_select(dev: bool, log=print):
    d = out_dir(dev)
    tuned = pd.read_csv(d / "tuned_table.csv")
    sec = pd.read_csv(d / "secondary_table.csv") if (d / "secondary_table.csv").exists() else pd.DataFrame()
    sel = {t: select_target(tuned[tuned["target"] == t], t) for t in PRIMARY_TARGETS}
    if not sec.empty:
        sel.update({t: select_target(sec[sec["target"] == t], t) for t in SECONDARY_TARGETS})
    both = pd.concat([tuned, sec], ignore_index=True)
    gate = gate_cv_stage(sel, both)
    d5 = json.loads((d / "feature_sets.json").read_text(encoding="utf-8")) if (d / "feature_sets.json").exists() else {}
    result = {"official": not dev, "created_utc": datetime.now(timezone.utc).isoformat(),
              "dataset_sha256_training": CONFIG["dataset_sha256"]["training"],
              "rule": {"selection": SEL, "status": CFG["selection_status"]},
              "selection": sel, "gate_cv_stage": gate, "d5": d5.get("d5"), "feature_sets": d5.get("feature_sets"),
              "best_other_family": best_other(tuned)}
    (d / "selection.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    untuned = pd.read_csv(d / "untuned_table.csv") if (d / "untuned_table.csv").exists() else pd.DataFrame()
    write_report(d, untuned, tuned, sec, result)
    figures(d, untuned, tuned, sel)
    record_stage(d, "select", {"selection": {t: r["model"] for t, r in sel.items()}}, dev)
    for t, r in sel.items():
        log(f"  {SHORT[t]:13s} -> {models.LABEL[r['model']]:22s} ({r['status']}; {gate[t]['check']} {gate[t]['result']})")
    return result


# ── report and figures ──────────────────────────────────────────────────────────────────────

def _f(x, d=3):
    try:
        return "—" if x is None or not np.isfinite(float(x)) else f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return str(x)


def write_report(d: Path, untuned, tuned, sec, result):
    P = PRIMARY_TARGETS
    L = [f"# Model benchmark (Sub-task 3.3){' — DEV RUN, not official' if not result['official'] else ''}", "",
         f"Built {result['created_utc'][:19]} UTC, dataset sha256 `{result['dataset_sha256_training'][:12]}…`. "
         "Grouped CV = the 2.8 folds (seed 42); design skill = share of the differences between designs within a "
         "climate that the model explains (heating targets over the 7 heating climates). Training split only.", "",
         "## 1. RQ1 grid — grouped-CV design skill (higher is better)", "",
         "| model | " + " | ".join(f"{SHORT[t]} R untuned | {SHORT[t]} C-G untuned | {SHORT[t]} tuned" for t in P) + " |",
         "|---|" + "---|" * (3 * len(P))]
    for m in models.MODELS:
        vals = []
        for t in P:
            for fs in (CFG["reference_set"], CFG["feature_set"]):
                s = untuned[(untuned.model == m) & (untuned.feature_set == fs) & (untuned.target == t)] if not untuned.empty else untuned
                vals.append(_f(s["grouped_design_skill"].iloc[0]) if len(s) else "—")
            s = tuned[(tuned.model == m) & (tuned.target == t)]
            vals.append(_f(s["grouped_design_skill"].iloc[0]) if len(s) else "—")
        L.append(f"| {models.LABEL[m]} | " + " | ".join(vals) + " |")
    L += ["", "GPR is not tuned by Optuna (E5): its 'tuned' column uses marginal-likelihood fitting only.", ""]

    def block(title, col, d_=3, table=tuned, targets=P):
        out = [f"## {title}", "", "| model | " + " | ".join(SHORT[t] for t in targets) + " |", "|---|" + "---|" * len(targets)]
        for m in table["model"].unique():
            vals = []
            for t in targets:
                s = table[(table.model == m) & (table.target == t)]
                vals.append(_f(s[col].iloc[0], d_) if len(s) and col in s else "—")
            out.append(f"| {models.LABEL[m]} | " + " | ".join(vals) + " |")
        return out + [""]

    L += block("2. Tuned — pooled R²", "grouped_r2")
    L += block("3. Tuned — MAE (kWh/m² or W/m²)", "grouped_mae", 2)
    L += block("4. Tuned — RMSE", "grouped_rmse", 2)
    L += block("5. Tuned — CV(RMSE)", "grouped_cv_rmse", 3)
    L += block("6. Tuned — within-climate ranking (median Spearman ρ)", "grouped_spearman_median")
    L += block("7. Tuned — top-10 % hit rate", "grouped_top_hit_mean")
    L += block("8. Fresh-fold check (E4) — design skill on GroupKFold seed 7", "fresh_design_skill")
    L += block("8b. Tuning optimism = skill(seed 42) − skill(seed 7)", "optimism_skill", 4)
    L += ["Positive = the seed-42 number is flattered by tuning on those folds; a few thousandths is normal.", ""]
    L += block("9. Unseen climate (LOCO) — MAPE % (used by M10 for primary targets)", "loco_mape_pct", 1)
    L += block("9b. LOCO median |climate bias| %", "loco_median_abs_bias_pct", 1)
    L += block("9c. LOCO bias-removed R² (ranking in an unseen climate)", "loco_bias_removed_r2")
    L += block("10. Training time — mean fit seconds per fold", "grouped_fit_s_per_fold", 1)
    L += ["MLP: one fit trains all 10 targets, so its time is the whole network's.", ""]
    if result.get("d5"):
        L += ["## 11. D5 transfer check (E3) — mean untuned skill over 3 primary targets", "",
              "| model | C-G | R | triggered |", "|---|---|---|---|"]
        for m, r in result["d5"].items():
            L.append(f"| {models.LABEL[m]} | {_f(r['C-G'], 4)} | {_f(r['R'], 4)} | {'YES' if r['triggered'] else 'no'} |")
        L.append("")
    if not sec.empty:
        L += block("12. Secondary targets — design skill (LightGBM + best other family)", "grouped_design_skill",
                   table=sec, targets=SECONDARY_TARGETS)
    L += ["## 13. Selection (M10, pre-registered) and CV-stage gate", "",
          "| target | selected | feature set | status | rule | S1/S2 (CV stage) |", "|---|---|---|---|---|---|"]
    for t, r in result["selection"].items():
        g = result["gate_cv_stage"][t]
        L.append(f"| {SHORT[t]} | {models.LABEL[r['model']]} | {r['feature_set']} | {r['status']} | {r['rule']} | "
                 f"{g['check']} {g['result']} (skill {_f(g['design_skill'])}"
                 + (f", R² {_f(g['pooled_r2'])}" if "pooled_r2" in g else "") + ") |")
    L += ["", "Primary selections are provisional until 3.4 adds the physics target forms (E1). The binding gate "
          "(S1–S8) is evaluated in 3.7 on the holdout."]
    (d / "benchmark_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def figures(d: Path, untuned, tuned, sel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(d / f"{name}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # RQ1 grid: per target, skill for each model in three columns (R untuned, C-G untuned, tuned)
    cols = ["R\nuntuned", "C-G\nuntuned", "C-G\ntuned"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for j, t in enumerate(PRIMARY_TARGETS):
        ax = axes[j]
        _style(ax)
        for k, m in enumerate(models.MODELS):
            vals = []
            for fs in (CFG["reference_set"], CFG["feature_set"]):
                s = untuned[(untuned.model == m) & (untuned.feature_set == fs) & (untuned.target == t)] if not untuned.empty else untuned
                vals.append(float(s["grouped_design_skill"].iloc[0]) if len(s) else np.nan)
            s = tuned[(tuned.model == m) & (tuned.target == t)]
            vals.append(float(s["grouped_design_skill"].iloc[0]) if len(s) else np.nan)
            x = np.arange(3) + (k - 2) * 0.09
            ax.plot(x, vals, "-", color=COLORS[m], lw=2, alpha=0.9)
            ax.plot(x, vals, MARKERS[m], color=COLORS[m], ms=8, mec="white", mew=1.5,
                    label=models.LABEL[m] if j == 0 else None)
        ax.axhline(SEL["primary_skill_min"], color="#8a8a85", lw=0.8, ls="--")
        ax.text(-0.36, SEL["primary_skill_min"] + 0.005, "S1 bar 0.90", fontsize=7, color="#55554f", va="bottom", ha="left")
        ax.set_xticks(range(3), cols, fontsize=7.5)
        ax.set_xlim(-0.4, 2.4)
        ax.set_title(SHORT[t], fontsize=10, loc="left", color="#22221f")
    axes[0].set_ylabel("Grouped-CV design skill", fontsize=8.5, color="#3a3a36")
    fig.legend(loc="lower center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    save(fig, "rq1_grid")

    # fresh-fold optimism
    if "optimism_skill" in tuned:
        fig, ax = plt.subplots(figsize=(7.5, 2.8))
        _style(ax)
        w = 0.16
        for k, m in enumerate(models.MODELS):
            v = [float(tuned[(tuned.model == m) & (tuned.target == t)]["optimism_skill"].iloc[0]) for t in PRIMARY_TARGETS]
            ax.bar(np.arange(3) + (k - 2) * w, v, width=w - 0.02, color=COLORS[m], label=models.LABEL[m],
                   edgecolor="white", linewidth=1)
        ax.axhline(0, color="#8a8a85", lw=0.8)
        ax.set_xticks(range(3), [SHORT[t] for t in PRIMARY_TARGETS], fontsize=8)
        ax.set_ylabel("skill(seed 42) − skill(seed 7)", fontsize=8, color="#3a3a36")
        ax.legend(frameon=False, fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        save(fig, "fresh_fold_optimism")

    # parity: selected model and best other family per primary target
    path = d / "oof_tuned_grouped.csv.gz"
    if not path.exists():
        return
    oof = pd.read_csv(path)
    other = best_other(tuned)
    for t in PRIMARY_TARGETS:
        chosen = sel[t]["model"]
        show = [chosen] + [m for m in (other, "lgbm") if m != chosen][:1]
        fig, axes = plt.subplots(1, len(show), figsize=(3.4 * len(show), 3.3), squeeze=False)
        for i, m in enumerate(show):
            ax = axes[0][i]
            _style(ax)
            ax.grid(axis="x", color="#e4e3dd", linewidth=0.6)
            g = oof[(oof.model == m) & (oof.target == t)]
            lo, hi = float(min(g.y.min(), g.yhat.min())), float(max(g.y.max(), g.yhat.max()))
            ax.plot([lo, hi], [lo, hi], color="#8a8a85", lw=0.8)
            ax.scatter(g.y, g.yhat, s=5, color=COLORS[m], alpha=0.35, linewidths=0)
            r = tuned[(tuned.model == m) & (tuned.target == t)].iloc[0]
            ax.set_title(f"{models.LABEL[m]}{' (selected)' if m == chosen else ''}", fontsize=9, loc="left", color="#22221f")
            ax.text(0.03, 0.95, f"design skill {r.grouped_design_skill:.3f}\npooled R² {r.grouped_r2:.3f}",
                    transform=ax.transAxes, fontsize=7.5, va="top", color="#3a3a36")
            ax.set_xlabel(f"EnergyPlus {SHORT[t]}", fontsize=8, color="#3a3a36")
            if i == 0:
                ax.set_ylabel(f"Surrogate {SHORT[t]}", fontsize=8, color="#3a3a36")
        save(fig, f"parity_{t}")


# ── main ────────────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sub-task 3.3 model benchmark")
    ap.add_argument("--stage", required=True, choices=["probe", "untuned", "d5", "tuned", "secondary", "select"])
    ap.add_argument("--model", choices=list(CFG["d5"]["models"]), help="for --stage d5")
    ap.add_argument("--gpr-top-k", type=int, default=None,
                    help=f"E5 fallback: GPR on the top-{CFG['gpr_fallback']['top_k']} features (only if the probe says so)")
    ap.add_argument("--dev", action="store_true", help="25 %% of designs, small models; writes outputs/benchmark_dev")
    a = ap.parse_args(argv)
    if a.gpr_top_k not in (None, CFG["gpr_fallback"]["top_k"]):
        ap.error(f"--gpr-top-k must be {CFG['gpr_fallback']['top_k']} (frozen in config)")
    log = lambda s: print(s, flush=True)                 # noqa: E731
    t0 = time.time()
    print(f"Benchmark stage '{a.stage}'{'  [DEV]' if a.dev else ''}", flush=True)
    if a.stage == "probe":
        stage_probe(a.dev, log)
    elif a.stage == "untuned":
        stage_untuned(a.dev, log)
    elif a.stage == "d5":
        if not a.model:
            ap.error("--stage d5 needs --model")
        stage_d5(a.model, a.dev, log)
    elif a.stage == "tuned":
        stage_tuned(a.dev, a.gpr_top_k, log)
    elif a.stage == "secondary":
        stage_secondary(a.dev, a.gpr_top_k, log)
    else:
        stage_select(a.dev, log)
    print(f"{'DEV' if a.dev else 'OFFICIAL'} stage '{a.stage}' done in {time.time() - t0:.0f} s -> "
          f"{out_dir(a.dev).relative_to(OUT.parent.parent)}/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
