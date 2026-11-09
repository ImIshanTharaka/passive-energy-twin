#!/usr/bin/env python3
"""
climate_gen.py - Sub-task 3.4: climate generalisation (plan M6, M7, M10; claude/Phase3.4_Execution_Plan.md
F1-F6; settings frozen in config.yaml `climate_gen_3_4`, 8 Nov 2026).

Plain idea: hide one city, train on the other nine, predict the hidden one (LOCO), ten times. Do it
for every model family and every physics form (heating "per unit of cold", EUI built from parts)
on the 70 inputs (C-G + 7 EPW indices), then apply the pre-registered selection rule and the
"do we need more climates?" trigger.

Stages (run in this order):

    indices   index table of the 10 training climates + climate distances -> climate_indices.csv
    grid      every family x form, grouped CV (seed 42) + LOCO -> grid_table.csv, grid_per_climate.csv,
              oof_grid_grouped.csv.gz, oof_grid_loco.csv.gz, gpr_selected_features.json, skew_table.csv
    select    M10 over 3.3 tuned C-G direct candidates + grid (+ fix 4), F5 trigger check, M7,
              one-family record -> selection.json, climate_gen_report.md. If F5 fires and fix 4 has
              not been run, the selection is marked 'awaiting_fix4' and M7 is not concluded.
    fix4      only if `select` says so: the triggering selected candidate(s) without mean DBT,
              latitude and wind -> fix4_table.csv (+ OOF); then run `select` again.
    sensitivity  (config sensitivity_3_4, reported only) LightGBM ratio forms with the learned
              quantity rescaled to the original target's SD -> sensitivity_table.csv
    figures   thesis figures + climate_gen_summary.md from the saved tables (no model fits)

    ./venv/bin/python surrogate/climate_gen.py --stage indices
    caffeinate -i ./venv/bin/python surrogate/climate_gen.py --stage grid
    ./venv/bin/python surrogate/climate_gen.py --stage select
    ./venv/bin/python surrogate/climate_gen.py --stage grid --dev      # 25 % of designs, small models

Hyper-parameters: the 3.3 tuned settings (tuning/best_params.json), same family and original
target; GPR defaults with the top-15 fallback; MLP `mlp__all` (F2, no re-tuning). The decomposed
EUI of the single-target families is assembled from out-of-fold parts on the same folds:
48.87 + heating (ratio form) + cooling + OA term (OATermModel) - identical to models.DecomposedEUI
(tests/test_climate_gen.py), without refitting heating and cooling twice.

Writes surrogate/outputs/climate_gen/ (dev: climate_gen_dev/). Training split only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from surrogate import benchmark as B
from surrogate import climate_indices as CI
from surrogate import cv, metrics as M, models
from surrogate.config import ALL_TARGETS, CONFIG, PRIMARY_TARGETS, SECONDARY_TARGETS
from surrogate.paths import OUT

CFG = CONFIG["climate_gen_3_4"]
SEL = CONFIG["selection"]
M7 = CONFIG["m7_trigger"]
FS = CFG["feature_set"]                                   # C-G+I
FS_FIX4 = FS + "-r"                                      # C-G+I without mean DBT, latitude, wind
HEAT, COOL, EUI, PEAK = models.HEAT, models.COOL, models.EUI, models.PEAK_HEAT
SHORT = B.SHORT


def out_dir(dev: bool) -> Path:
    d = OUT / ("climate_gen_dev" if dev else "climate_gen")
    d.mkdir(parents=True, exist_ok=True)
    return d


def record_stage(d: Path, stage: str, info: dict, dev: bool):
    path = d / "climate_gen_result.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"stages": {}}
    data["stages"][stage] = {"official": not dev, "dev": dev, "created_utc": datetime.now(timezone.utc).isoformat(),
                             "dataset_sha256_training": CONFIG["dataset_sha256"]["training"],
                             "environment": B.environment(), **info}
    path.write_text(json.dumps(data, indent=2, default=float), encoding="utf-8")


# ── data ────────────────────────────────────────────────────────────────────────────────────

def load(dev: bool):
    """(df, X, heat): training rows (dev: B.load's fixed 25 % design sample) and the C-G+I matrix."""
    df, _, heat = B.load(dev)
    return df, CI.feature_matrix(df), heat


def climate_distances(table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per training climate: Euclidean distance (7 Phase 2 climate inputs, z-scored over the 10 training
    climates) to the nearest OTHER training climate - the x axis of the error-vs-distance figure."""
    from surrogate import features as FE
    df, _ = cv.load_training()
    c = df.groupby("climate_key")[FE.CLIMATE_INPUTS].first()
    z = (c - c.mean()) / c.std(ddof=0)
    d = np.sqrt(((z.to_numpy()[:, None, :] - z.to_numpy()[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    j = d.argmin(axis=1)
    return pd.DataFrame({"climate_key": c.index, "distance_to_nearest": d.min(axis=1),
                         "nearest_climate": c.index[j]}).set_index("climate_key")


def reduced_columns(X: pd.DataFrame) -> list[str]:
    drop = set(CFG["fix4"]["drop_climate_inputs"])
    return [c for c in X.columns if c not in drop]


# ── hyper-parameters (F2: no re-tuning) ─────────────────────────────────────────────────────

def params_for(model: str, targets: list[str], dev: bool, gpr_top_k: int | None) -> dict:
    """{target: params} ({None: params} for the MLP) - the 3.3 tuned settings, tuned on C-G."""
    return B.tuned_params_for(model, targets, dev, gpr_top_k, allow_missing=dev,
                              feature_set=CFG["hyperparameters"]["tuned_on_feature_set"])


def resolve_gpr_top_k(dev: bool) -> int | None:
    if dev:
        return None                                         # dev_params forces 5
    k = int(CFG["hyperparameters"]["gpr"]["top_k"])
    benchmark_k = B.resolve_gpr_top_k(False)                # must agree with the 3.3 timing probe (E5)
    if benchmark_k != k:
        raise SystemExit(f"config top_k {k} disagrees with the 3.3 timing probe ({benchmark_k})")
    return k


# ── out-of-fold predictions per candidate ───────────────────────────────────────────────────

class _Keep:
    """Factory wrapper that keeps every estimator it builds (to read GPR selected features)."""

    def __init__(self, factory):
        self.factory, self.built = factory, []

    def __call__(self, target):
        est = self.factory(target)
        self.built.append(est)
        return est


def _gpr_features(est) -> list | None:
    base = getattr(est, "base_", None) or getattr(est, "dev_", None) or est
    return getattr(base, "selected_features_", None)


# parts each candidate needs (fix 4 runs only these; the grid runs all)
NEEDS = {(HEAT, "direct"): [(HEAT, "direct")], (HEAT, "log1p"): [(HEAT, "log1p")], (HEAT, "ratio"): [(HEAT, "ratio")],
         (COOL, "direct"): [(COOL, "direct")], (EUI, "direct"): [(EUI, "direct")],
         (EUI, "decomposed"): [(HEAT, "ratio"), (COOL, "direct"), (EUI, "oa_term")]}


def single_family_oof(model: str, pbt: dict, df, X, folds, scheme: str, log=print,
                      only: list[tuple[str, str]] | None = None) -> tuple[pd.DataFrame, dict]:
    """Single-target candidates of one family on one fold set (all, or only the (target, form) pairs
    in `only` and the parts they need).

    Returns (oof with columns + form, gpr_features {candidate: [per-fold feature lists]})."""
    parts, feats = [], {}
    jobs = [(HEAT, "direct", models.make_form(model, pbt[HEAT], "direct")),
            (HEAT, "log1p", models.make_form(model, pbt[HEAT], "log1p")),
            (HEAT, "ratio", models.make_form(model, pbt[HEAT], "ratio")),
            (COOL, "direct", models.make_form(model, pbt[COOL], "direct")),
            (EUI, "direct", models.make_form(model, pbt[EUI], "direct")),
            (EUI, "oa_term", models.make_oa_term(model, pbt[EUI]))]
    if only is not None:
        need = {p for cand in only for p in NEEDS[cand]}
        jobs = [j for j in jobs if (j[0], j[1]) in need]
    for target, form, fac in jobs:
        keep = _Keep(fac)
        t0 = time.time()
        oof = cv.cross_val_predict(df, X, [target], keep, folds, scheme).assign(form=form)
        parts.append(oof)
        if model == "gpr":
            feats[f"{target}|{form}"] = [_gpr_features(e) for e in keep.built]
        log(f"    {model:5s} {scheme:7s} {SHORT[target]:8s} {form:8s} {time.time() - t0:7.1f} s")
    oof = pd.concat(parts, ignore_index=True)
    if (oof["form"] == "oa_term").any():
        oof = compose_decomposed(oof)
    return oof, feats


def compose_decomposed(oof: pd.DataFrame) -> pd.DataFrame:
    """Add EUI 'decomposed' = 48.87 + heating (ratio) + cooling (direct) + OA term, row by row (F3).

    All parts come from the same folds, so this equals models.DecomposedEUI fitted per fold."""
    key = ["run_id", "fold"]
    h = oof[(oof["target"] == HEAT) & (oof["form"] == "ratio")].set_index(key)["yhat"]
    c = oof[(oof["target"] == COOL) & (oof["form"] == "direct")].set_index(key)["yhat"]
    oa = oof[(oof["target"] == EUI) & (oof["form"] == "oa_term")]
    dec = oa.copy()
    idx = pd.MultiIndex.from_frame(oa[key])
    dec["yhat"] = models.ELEC + h.reindex(idx).to_numpy() + c.reindex(idx).to_numpy() + oa["yhat"].to_numpy()
    dec["fit_s"] = oa["fit_s"].to_numpy()                   # the extra fit; heating/cooling reused
    dec["form"] = "decomposed"
    if dec["yhat"].isna().any():
        raise AssertionError("decomposed EUI: missing heating or cooling prediction for some rows")
    return pd.concat([oof[oof["form"] != "oa_term"], dec], ignore_index=True)


def mlp_oof(params: dict, df, X, folds, scheme: str) -> pd.DataFrame:
    """MLPPhysics (F4): one 14-output network per fold; every form from predict_forms."""
    from eda import eda_lib as L
    L.require_training(df)
    Xr = X.reset_index(drop=True)
    design = df["design_id"].astype(str).to_numpy()
    Y = df[ALL_TARGETS].to_numpy(dtype=float, copy=True)
    parts = []
    for f in folds:
        net = models.MLPPhysics(params)
        kw = cv._fit_kwargs(net, df, f.train, design)
        t0 = time.perf_counter()
        net.fit(Xr.iloc[f.train], Y[f.train], **kw)
        fit_s = time.perf_counter() - t0
        forms = net.predict_forms(Xr.iloc[f.test])
        for (t, form) in forms.columns:
            if t not in (HEAT, COOL, EUI, PEAK):
                continue
            parts.append(pd.DataFrame({"run_id": df["run_id"].to_numpy()[f.test], "design_id": design[f.test],
                                       "climate_key": df["climate_key"].astype(str).to_numpy()[f.test],
                                       "target": t, "fold": f.name, "y": df[t].to_numpy()[f.test],
                                       "yhat": forms[(t, form)].to_numpy(), "fit_s": fit_s, "form": form}))
    out = pd.concat(parts, ignore_index=True)
    out["scheme"] = scheme
    return out


def lgbm_secondary_oof(pbt: dict, df, X, folds, scheme: str, targets_forms: list[tuple[str, str]]) -> pd.DataFrame:
    parts = [cv.cross_val_predict(df, X, [t], models.make_form("lgbm", pbt[t], form), folds, scheme).assign(form=form)
             for t, form in targets_forms]
    return pd.concat(parts, ignore_index=True)


# ── scoring ─────────────────────────────────────────────────────────────────────────────────

def score(oof_g: pd.DataFrame, oof_l: pd.DataFrame, heat, model: str, feature_set: str, n_features: int):
    """Table rows (one per target x form) + per-climate rows for one family."""
    rows, pcs = [], []
    for (t, form), g in oof_g.groupby(["target", "form"], sort=False):
        gr, pc = B.grouped_rows(g, heat)
        lo, tab = B.loco_rows(oof_l[(oof_l["target"] == t) & (oof_l["form"] == form)], heat)
        rows.append({"model": model, "feature_set": feature_set, "form": form, "target": t,
                     "n_features": n_features, "tuned": model in models.TUNABLE, **gr[t], **lo[t]})
        pcs.append(pc.assign(model=model, feature_set=feature_set, form=form, scheme="grouped"))
        pcs.append(tab.assign(model=model, feature_set=feature_set, form=form, scheme="loco"))
    return rows, pcs


def skew_targets(df: pd.DataFrame, heat) -> dict:
    out = {}
    for t in SECONDARY_TARGETS:
        g = df[df["climate_key"].isin(heat)] if t in CONFIG["targets"]["heating"] else df
        out[t] = float(g[t].skew())
    return out


# ── stages ──────────────────────────────────────────────────────────────────────────────────

def stage_indices(dev: bool, log=print):
    d = out_dir(dev)
    t = CI.training_indices().join(climate_distances())
    t.to_csv(d / "climate_indices.csv")
    record_stage(d, "indices", {"indices": CI.INDEX_COLUMNS}, dev)
    log(t.round(2).to_string())
    return t


def stage_grid(dev: bool, log=print, only: list[str] | None = None):
    df, X, heat = load(dev)
    k = resolve_gpr_top_k(dev)
    folds = {"grouped": cv.grouped_folds(df), "loco": cv.loco_folds(df)}
    d = out_dir(dev)
    rows, pcs, oofs, gpr_feats, params_used, timing = [], [], {"grouped": [], "loco": []}, {}, {}, {}
    for m in (only or CFG["models"]):
        t0 = time.time()
        log(f"  {models.LABEL[m]}")
        if m in models.MULTI_OUTPUT:
            p = params_for(m, PRIMARY_TARGETS, dev, k)[None]
            if dev:
                p = {**p, "max_epochs": 60, "patience": 10}
            params_used[m] = p
            res = {s: mlp_oof(p, df, X, folds[s], s) for s in folds}
        else:
            pbt = params_for(m, PRIMARY_TARGETS, dev, k)
            params_used[m] = pbt
            res = {}
            for s in folds:
                res[s], feats = single_family_oof(m, pbt, df, X, folds[s], s, log)
                if feats:
                    gpr_feats[s] = feats
            if m == "lgbm":                                   # peak heating forms (secondary, F2)
                pp = params_for(m, [PEAK], dev, k)
                for s in folds:
                    res[s] = pd.concat([res[s], lgbm_secondary_oof(pp, df, X, folds[s], s,
                                                                   [(PEAK, "direct"), (PEAK, "ratio")])],
                                       ignore_index=True)
        r, pc = score(res["grouped"], res["loco"], heat, m, FS, X.shape[1])
        rows += r
        pcs += pc
        for s in folds:
            oofs[s].append(res[s].assign(model=m, feature_set=FS))
        timing[m] = time.time() - t0
        for t in PRIMARY_TARGETS:
            sub = [x for x in r if x["target"] == t]
            log(f"    {SHORT[t]:8s} " + "  ".join(f"{x['form']}: skill {x['grouped_design_skill']:.3f} "
                                                  f"LOCO MAPE {x['loco_mape_pct']:.1f}%" for x in sub))
        log(f"    {m} done in {timing[m]:.0f} s")

    # skew check (reported only, F2 / config skew_check)
    sk = skew_targets(df, heat)
    skewed = [t for t, v in sk.items() if v > CFG["skew_check"]["threshold"]]
    skew_rows = []
    if skewed and (only is None or "lgbm" in only):
        pp = params_for("lgbm", skewed, dev, k)
        res = {s: lgbm_secondary_oof(pp, df, X, folds[s], s, [(t, f) for t in skewed for f in ("direct", "log1p")])
               for s in folds}
        r, _ = score(res["grouped"], res["loco"], heat, "lgbm", FS, X.shape[1])
        skew_rows = [{**x, "skew": sk[x["target"]], "selectable": False} for x in r]
    pd.DataFrame(skew_rows).to_csv(d / "skew_table.csv", index=False)

    table = pd.DataFrame(rows)
    table.to_csv(d / "grid_table.csv", index=False)
    pd.concat(pcs, ignore_index=True).to_csv(d / "grid_per_climate.csv", index=False)
    for s in folds:
        pd.concat(oofs[s], ignore_index=True).to_csv(d / f"oof_grid_{s}.csv.gz", index=False)
    (d / "gpr_selected_features.json").write_text(json.dumps(gpr_feats, indent=1), encoding="utf-8")
    record_stage(d, "grid", {"feature_set": FS, "n_features": X.shape[1], "gpr_top_k": k, "seconds": timing,
                             "params": {m: {str(t): v for t, v in p.items()} if m not in models.MULTI_OUTPUT else p
                                        for m, p in params_used.items()},
                             "skew": sk, "skewed_secondary": skewed, "n_rows": len(df)}, dev)
    return table


def stage_fix4(dev: bool, log=print):
    d = out_dir(dev)
    sel_path = d / "selection.json"
    if not sel_path.exists():
        raise SystemExit("run --stage select first: it decides whether fix 4 is needed (F5)")
    need = json.loads(sel_path.read_text(encoding="utf-8")).get("fix4", {}).get("needed_for", [])
    if not need:
        raise SystemExit("fix 4 is not triggered (F5): nothing to run")
    df, X, heat = load(dev)
    k = resolve_gpr_top_k(dev)
    cols = reduced_columns(X)
    Xr = X[cols]
    folds = {"grouped": cv.grouped_folds(df), "loco": cv.loco_folds(df)}
    rows, oofs = [], {"grouped": [], "loco": []}
    for item in need:
        m, t, form = item["model"], item["target"], item["form"]
        log(f"  fix 4: {models.LABEL[m]} {SHORT[t]} {form} on {len(cols)} inputs")
        if m in models.MULTI_OUTPUT:
            p = params_for(m, PRIMARY_TARGETS, dev, k)[None]
            if dev:
                p = {**p, "max_epochs": 60, "patience": 10}
            res = {s: mlp_oof(p, df, Xr, folds[s], s) for s in folds}
        else:
            pbt = params_for(m, PRIMARY_TARGETS, dev, k)
            res = {s: single_family_oof(m, pbt, df, Xr, folds[s], s, log, only=[(t, form)])[0] for s in folds}
        res = {s: v[(v["target"] == t) & (v["form"] == form)] for s, v in res.items()}
        r, _ = score(res["grouped"], res["loco"], heat, m, FS_FIX4, len(cols))
        rows += r
        for s in folds:
            oofs[s].append(res[s].assign(model=m, feature_set=FS_FIX4))
    pd.DataFrame(rows).to_csv(d / "fix4_table.csv", index=False)
    for s in folds:
        pd.concat(oofs[s], ignore_index=True).to_csv(d / f"oof_fix4_{s}.csv.gz", index=False)
    record_stage(d, "fix4", {"needed_for": need, "inputs": cols}, dev)


# ── sensitivity_3_4 (reported only) ────────────────────────────────────────────────────────

SENS = CONFIG.get("sensitivity_3_4", {})


def sensitivity_oof(pbt: dict, df, X, folds, scheme: str) -> pd.DataFrame:
    """LightGBM ratio forms with scale matching (TargetForm(scale_match=True)); same folds as the grid.
    Decomposed EUI = 48.87 + scaled heating ratio + cooling + OA term (cooling and OA term unscaled,
    exactly as in the grid)."""
    parts = []
    for target, form, fac in ((HEAT, "ratio", models.make_form("lgbm", pbt[HEAT], "ratio", scale_match=True)),
                              (PEAK, "ratio", models.make_form("lgbm", pbt[PEAK], "ratio", scale_match=True)),
                              (COOL, "direct", models.make_form("lgbm", pbt[COOL], "direct")),
                              (EUI, "oa_term", models.make_oa_term("lgbm", pbt[EUI]))):
        parts.append(cv.cross_val_predict(df, X, [target], fac, folds, scheme).assign(form=form))
    oof = compose_decomposed(pd.concat(parts, ignore_index=True))
    keep = {(HEAT, "ratio"), (PEAK, "ratio"), (EUI, "decomposed")}
    oof = oof[[(t, f) in keep for t, f in zip(oof["target"], oof["form"])]].copy()
    oof["form"] = oof["form"] + "_scaled"
    return oof


def stage_sensitivity(dev: bool, log=print):
    if not SENS:
        raise SystemExit("config block sensitivity_3_4 missing")
    df, X, heat = load(dev)
    k = resolve_gpr_top_k(dev)
    pbt = params_for("lgbm", PRIMARY_TARGETS + [PEAK], dev, k)
    folds = {"grouped": cv.grouped_folds(df), "loco": cv.loco_folds(df)}
    res = {s: sensitivity_oof(pbt, df, X, folds[s], s) for s in folds}
    rows, pcs = score(res["grouped"], res["loco"], heat, "lgbm", FS, X.shape[1])
    d = out_dir(dev)
    grid = pd.read_csv(d / "grid_table.csv")
    for r in rows:
        base_form = r["form"].replace("_scaled", "")
        g = grid[(grid["model"] == "lgbm") & (grid["target"] == r["target"]) & (grid["form"] == base_form)]
        r["unscaled_grouped_design_skill"] = float(g["grouped_design_skill"].iloc[0]) if len(g) else np.nan
        r["unscaled_loco_mape_pct"] = float(g["loco_mape_pct"].iloc[0]) if len(g) else np.nan
        r["selectable"] = False
    table = pd.DataFrame(rows)
    table.to_csv(d / "sensitivity_table.csv", index=False)
    pd.concat(pcs, ignore_index=True).to_csv(d / "sensitivity_per_climate.csv", index=False)
    for s in folds:
        res[s].assign(model="lgbm", feature_set=FS).to_csv(d / f"oof_sensitivity_{s}.csv.gz", index=False)
    record_stage(d, "sensitivity", {"block": "sensitivity_3_4", "selectable": False}, dev)
    for r in rows:
        log(f"  {SHORT[r['target']]:13s} {r['form']:18s} skill {r['unscaled_grouped_design_skill']:.3f} -> "
            f"{r['grouped_design_skill']:.3f}   LOCO MAPE {r['unscaled_loco_mape_pct']:.1f} -> {r['loco_mape_pct']:.1f} %")
    return table


# ── figures (step 7; reads saved tables only) ───────────────────────────────────────────────

FAMILY_ORDER = ["rf", "xgb", "lgbm", "gpr", "mlp"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3 (dataviz palette, validated)
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
PROVISIONAL_3_3 = {HEAT: "mlp", COOL: "mlp", EUI: "gpr"}   # 3.3 selection.json (provisional, E1)
CITY = {"atlanta": "Atlanta", "chennai": "Chennai", "chicago": "Chicago", "colombo": "Colombo", "denver": "Denver",
        "miami": "Miami", "minneapolis": "Minneapolis", "new_york": "New York", "phoenix": "Phoenix",
        "san_francisco": "San Francisco"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, d: Path, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(d / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")


def fig_before_after(d: Path, grid: pd.DataFrame):
    import matplotlib.pyplot as plt
    panels = [(HEAT, [("direct", "As is"), ("log1p", "Log"), ("ratio", "Per unit of cold")]),
              (EUI, [("direct", "As is"), ("decomposed", "Built from parts")])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (t, forms) in zip(axes, panels):
        _style(ax)
        h = 0.8 / len(forms)
        for j, (form, label) in enumerate(forms):
            colour = SERIES[2] if form in ("ratio", "decomposed") else SERIES[j]
            for i, m in enumerate(FAMILY_ORDER):
                r = grid[(grid["model"] == m) & (grid["target"] == t) & (grid["form"] == form)]
                if r.empty:
                    continue
                v = float(r["loco_mape_pct"].iloc[0])
                fails = float(r["grouped_design_skill"].iloc[0]) < SEL["primary_skill_min"]
                y = i - 0.4 + h * (j + 0.5)
                ax.barh(y, v, height=h - 0.04, color=colour, hatch="///" if fails else None,
                        edgecolor="white", linewidth=0.8)
                ax.text(v + 0.6, y, f"{v:.1f}", va="center", fontsize=7.5, color=INK2)
        ax.set_yticks(range(len(FAMILY_ORDER)), [models.LABEL[m] for m in FAMILY_ORDER])
        ax.invert_yaxis()
        ax.set_xlabel("New-city error, LOCO MAPE (%)", color=INK2, fontsize=9)
        ax.set_title(SHORT[t], color=INK, fontsize=11, loc="left", pad=22)
        ax.set_xlim(0, ax.get_xlim()[1] * 1.08)
        from matplotlib.patches import Patch
        handles = [Patch(color=SERIES[2] if f in ("ratio", "decomposed") else SERIES[j], label=lab)
                   for j, (f, lab) in enumerate(forms)]
        ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(forms),
                  handlelength=1.2, borderaxespad=0.2)
    fig.text(0.01, -0.04, "Hatched: grouped design skill below 0.90 (not eligible). LightGBM per unit of cold: "
             "re-used regularisation not scale-free (see sensitivity_table.csv).", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    _save(fig, d, "fig_before_after")
    plt.close(fig)


def _per_climate_loco(d: Path):
    pc = pd.read_csv(d / "grid_per_climate.csv")
    pc = pc[pc["scheme"] == "loco"]
    old = pd.read_csv(B.out_dir(False) / "tuned_per_climate.csv")
    old = old[old["scheme"] == "loco"].assign(feature_set="C-G", form="direct")
    return pd.concat([pc, old], ignore_index=True)


def _pick_rows(pc: pd.DataFrame, t: str, model: str, fs: str, form: str) -> pd.DataFrame:
    return pc[(pc["target"] == t) & (pc["model"] == model) & (pc["feature_set"] == fs) & (pc["form"] == form)]


def fig_bias_per_climate(d: Path, pc: pd.DataFrame, sel: dict):
    import matplotlib.pyplot as plt
    targets = [HEAT, COOL, EUI]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=False)
    for ax, t in zip(axes, targets):
        _style(ax)
        new = _pick_rows(pc, t, sel[t]["model"], sel[t]["feature_set"], sel[t]["form"]).set_index("climate_key")
        old = _pick_rows(pc, t, PROVISIONAL_3_3[t], "C-G", "direct").set_index("climate_key")
        order = new["abs_bias_pct"].sort_values().index.tolist()
        ys = np.arange(len(order))
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.scatter(old.loc[order, "bias_pct"], ys, s=22, color=SERIES[0], zorder=5,
                   edgecolor="white", linewidth=1.0, label=f"3.3: {models.LABEL[PROVISIONAL_3_3[t]]}, as is")
        ax.scatter(new.loc[order, "bias_pct"], ys, s=60, color=SERIES[2], zorder=4, marker="D",
                   edgecolor="white", linewidth=1.5,
                   label=f"3.4: {models.LABEL[sel[t]['model']]}, {sel[t]['form']}")
        for y, c in zip(ys, order):
            ax.plot([old.loc[c, "bias_pct"], new.loc[c, "bias_pct"]], [y, y], color=GRID, linewidth=1.5, zorder=2)
        ax.set_yticks(ys, [CITY[c] for c in order])
        ax.set_xlabel("City bias when hidden (%)  — + over-predicts, − under-predicts", color=INK2, fontsize=8.5)
        same = (sel[t]["model"], sel[t]["feature_set"], sel[t]["form"]) == (PROVISIONAL_3_3[t], "C-G", "direct")
        ax.set_title(SHORT[t] + ("  (3.3 pick kept)" if same else ""), color=INK, fontsize=11, loc="left", pad=30)
        ax.legend(frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1,
                  borderaxespad=0.2, handletextpad=0.3)
    fig.tight_layout()
    _save(fig, d, "fig_bias_per_climate")
    plt.close(fig)


def fig_error_vs_distance(d: Path, pc: pd.DataFrame, sel: dict, dist: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for ax, t in zip(axes, [HEAT, EUI]):
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        new = _pick_rows(pc, t, sel[t]["model"], sel[t]["feature_set"], sel[t]["form"]).set_index("climate_key")
        old = _pick_rows(pc, t, PROVISIONAL_3_3[t], "C-G", "direct").set_index("climate_key")
        x = dist.loc[new.index, "distance_to_nearest"].copy()
        # cities that are each other's nearest neighbour share a distance: nudge them apart so each
        # blue-green pair has its own column (display only; the table holds exact distances)
        left = set()
        for _, grp in x.groupby(x.round(6)):
            if len(grp) > 1:
                for j, c in enumerate(sorted(grp.index)):
                    x[c] += (j - (len(grp) - 1) / 2) * 0.06
                    if j == 0:
                        left.add(c)
        xo = x.reindex(old.index)
        for c in new.index:
            if c in old.index:
                ax.plot([x[c], x[c]], [old.loc[c, "abs_bias_pct"], new.loc[c, "abs_bias_pct"]], color=GRID,
                        linewidth=1.5, zorder=2)
        ax.scatter(xo, old["abs_bias_pct"], s=22, color=SERIES[0],          # on top, smaller: visible inside an
                   edgecolor="white", linewidth=1.0, zorder=5, label=f"3.3: {models.LABEL[PROVISIONAL_3_3[t]]}, as is")
        ax.scatter(x, new["abs_bias_pct"], s=60, color=SERIES[2], marker="D", edgecolor="white", linewidth=1.5,
                   zorder=4, label=f"3.4: {models.LABEL[sel[t]['model']]}, {sel[t]['form']}")
        for c in new.index:                                   # left member of a shared column: label on the left
            dx, ha = (-6, "right") if c in left else (6, "left")
            ax.annotate(CITY[c], (x[c], new.loc[c, "abs_bias_pct"]), xytext=(dx, -3), textcoords="offset points",
                        fontsize=7, color=INK2, ha=ha)
        ax.set_xlim(left=min(ax.get_xlim()[0], float(x.min()) - 0.35), right=max(ax.get_xlim()[1], float(x.max()) + 0.45))
        ax.set_xlabel("Distance to the nearest training climate (z-scored climate inputs)", color=INK2, fontsize=8.5)
        ax.set_ylabel("|City bias| when hidden (%)", color=INK2, fontsize=8.5)
        ax.set_title(SHORT[t], color=INK, fontsize=11, loc="left", pad=30)
        ax.legend(frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, borderaxespad=0.2)
    fig.tight_layout()
    _save(fig, d, "fig_error_vs_distance")
    plt.close(fig)


def fig_ranking_unseen(d: Path, pc: pd.DataFrame, sel: dict):
    import matplotlib.pyplot as plt
    targets = [HEAT, COOL, EUI]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.grid(axis="x", visible=False)
    for i, t in enumerate(targets):
        rows = _pick_rows(pc, t, sel[t]["model"], sel[t]["feature_set"], sel[t]["form"])
        jitter = np.linspace(-0.18, 0.18, len(rows))
        ax.scatter(i + jitter, rows["spearman"], s=40, color=SERIES[0], edgecolor="white", linewidth=1.5, zorder=3)
        med = float(rows["spearman"].median())
        ax.plot([i - 0.28, i + 0.28], [med, med], color=INK, linewidth=2)
        ax.text(i + 0.31, med, f"median {med:.3f}", va="center", fontsize=7.5, color=INK2)
        worst = rows.loc[rows["spearman"].idxmin()]
        ax.annotate(CITY[worst["climate_key"]], (i + jitter[list(rows.index).index(worst.name)], worst["spearman"]),
                    xytext=(-4, -10), textcoords="offset points", fontsize=7, color=MUTED, ha="right")
    ax.set_xticks(range(len(targets)), [f"{SHORT[t]}\n{models.LABEL[sel[t]['model']]}, {sel[t]['form']}"
                                        for t in targets], fontsize=8.5)
    ax.set_ylabel("Rank agreement in the hidden city\n(Spearman, 1 = same order)", color=INK2, fontsize=8.5)
    ax.set_ylim(min(0.9, ax.get_ylim()[0]), 1.0)
    fig.tight_layout()
    _save(fig, d, "fig_ranking_unseen")
    plt.close(fig)


def write_summary(d: Path, grid: pd.DataFrame, sel: dict, pc: pd.DataFrame, sens: pd.DataFrame | None):
    old = pd.concat([pd.read_csv(B.out_dir(False) / "tuned_table.csv").assign(feature_set="C-G", form="direct")])
    L = ["# Sub-task 3.4 — summary (thesis tables)", "",
         "All errors in original units (kWh/m²) after the inverse transform; LOCO = leave-one-climate-out.", "",
         "## Before (3.3 provisional) vs after (3.4 final)", "",
         "| Target | 3.3 pick | LOCO MAPE % | median \\|bias\\| % | 3.4 pick | grouped skill | LOCO MAPE % | median \\|bias\\| % | max \\|bias\\| % |",
         "|---|---|---|---|---|---|---|---|---|"]
    for t in PRIMARY_TARGETS:
        o = old[(old["target"] == t) & (old["model"] == PROVISIONAL_3_3[t])].iloc[0]
        r = sel[t]
        n = pd.concat([grid, old])
        n = n[(n["target"] == t) & (n["model"] == r["model"]) & (n["feature_set"] == r["feature_set"]) & (n["form"] == r["form"])].iloc[0]
        L.append(f"| {SHORT[t]} | {models.LABEL[PROVISIONAL_3_3[t]]} C-G direct | {o['loco_mape_pct']:.1f} | "
                 f"{o['loco_median_abs_bias_pct']:.1f} | {models.LABEL[r['model']]} {r['feature_set']} {r['form']} | "
                 f"{n['grouped_design_skill']:.3f} | {n['loco_mape_pct']:.1f} | {n['loco_median_abs_bias_pct']:.1f} | "
                 f"{n['loco_max_abs_bias_pct']:.1f} ({CITY.get(n['loco_worst_climate'], n['loco_worst_climate'])}) |")
    L += ["", "## LOCO MAPE % by family and form (C-G+I)", "",
          "| Family | Heating direct | Heating log1p | Heating ratio | Cooling direct | EUI direct | EUI decomposed |",
          "|---|---|---|---|---|---|---|"]
    for m in FAMILY_ORDER:
        cells = []
        for t, f in ((HEAT, "direct"), (HEAT, "log1p"), (HEAT, "ratio"), (COOL, "direct"), (EUI, "direct"), (EUI, "decomposed")):
            r = grid[(grid["model"] == m) & (grid["target"] == t) & (grid["form"] == f)]
            cells.append("—" if r.empty else f"{r['loco_mape_pct'].iloc[0]:.1f} ({r['grouped_design_skill'].iloc[0]:.3f})")
        L.append(f"| {models.LABEL[m]} | " + " | ".join(cells) + " |")
    L += ["", "Cells: LOCO MAPE % (grouped design skill).", ""]
    if sens is not None and not sens.empty:
        L += ["## Sensitivity (sensitivity_3_4, reported only): LightGBM ratio forms with scale matching", "",
              "| Target | Form | grouped skill (unscaled → scaled) | LOCO MAPE % (unscaled → scaled) | median \\|bias\\| % |",
              "|---|---|---|---|---|"]
        for r in sens.itertuples():
            L.append(f"| {SHORT[r.target]} | {r.form} | {r.unscaled_grouped_design_skill:.3f} → {r.grouped_design_skill:.3f} | "
                     f"{r.unscaled_loco_mape_pct:.1f} → {r.loco_mape_pct:.1f} | {r.loco_median_abs_bias_pct:.1f} |")
        L.append("")
    L += ["Figures: fig_before_after, fig_bias_per_climate, fig_error_vs_distance, fig_ranking_unseen (PNG + PDF)."]
    (d / "climate_gen_summary.md").write_text("\n".join(L), encoding="utf-8")


def stage_figures(dev: bool, log=print):
    import matplotlib
    matplotlib.use("Agg")
    d = out_dir(dev)
    grid = pd.read_csv(d / "grid_table.csv")
    result = json.loads((d / "selection.json").read_text(encoding="utf-8"))
    if result["status"] != "final":
        raise SystemExit("selection is not final: run fix4/select first")
    sel = result["selection"]
    pc = _per_climate_loco(d)
    dist = pd.read_csv(d / "climate_indices.csv", index_col="climate_key")
    sens = pd.read_csv(d / "sensitivity_table.csv") if (d / "sensitivity_table.csv").exists() else None
    fig_before_after(d, grid)
    fig_bias_per_climate(d, pc, sel)
    fig_error_vs_distance(d, pc, sel, dist)
    fig_ranking_unseen(d, pc, sel)
    write_summary(d, grid, sel, pc, sens)
    record_stage(d, "figures", {"figures": ["fig_before_after", "fig_bias_per_climate", "fig_error_vs_distance",
                                            "fig_ranking_unseen"], "summary": "climate_gen_summary.md"}, dev)
    log("  figures + climate_gen_summary.md written")


# ── selection (F6) ──────────────────────────────────────────────────────────────────────────

def baseline_pool(dev: bool, log=print) -> pd.DataFrame:
    """3.3 tuned C-G direct candidates (primary) and secondary rows, as candidates (form 'direct')."""
    src = B.out_dir(False)
    if dev:
        log("  [DEV] baseline pool read from the official 3.3 outputs (code check only)")
    parts = []
    for name in ("tuned_table.csv", "secondary_table.csv"):
        p = src / name
        if p.exists():
            parts.append(pd.read_csv(p).assign(form="direct"))
        elif not dev:
            raise SystemExit(f"missing 3.3 output {p}")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def select_target(cands: pd.DataFrame, target: str) -> dict:
    """M10 for one target over candidates (model, feature_set, form). Ties within the margin go to the
    LightGBM candidate with the best value (config climate_gen_3_4.selection.tie_break)."""
    primary = target in PRIMARY_TARGETS
    bar = SEL["primary_skill_min"] if primary else SEL["secondary_skill_min"]
    c = cands.reset_index(drop=True)
    passing = c[c["grouped_design_skill"] >= bar]
    rec = {"target": target, "kind": "primary" if primary else "secondary", "skill_bar": bar,
           "n_candidates": len(c), "n_passing": len(passing)}
    if passing.empty:
        best = c.loc[c["grouped_design_skill"].idxmax()]
        rec.update({k: best[k] for k in ("model", "feature_set", "form")},
                   rule="no candidate passes: best grouped skill", gate_fail=True)
        return rec
    if primary:
        key, best_i = "loco_mape_pct", passing["loco_mape_pct"].idxmin()
        tied = passing[passing[key] <= passing.loc[best_i, key] + SEL["tie_mape_points"] + 1e-12]
        rule = f"lowest LOCO MAPE (ties within {SEL['tie_mape_points']} points -> best LightGBM)"
        lg = tied[tied["model"] == "lgbm"]
        lg_best = lg[key].idxmin() if not lg.empty else None
    else:
        key, best_i = "grouped_design_skill", passing["grouped_design_skill"].idxmax()
        tied = passing[passing[key] >= passing.loc[best_i, key] - SEL["tie_skill"] - 1e-12]
        rule = f"highest grouped skill (ties within {SEL['tie_skill']} -> best LightGBM)"
        lg = tied[tied["model"] == "lgbm"]
        lg_best = lg[key].idxmax() if not lg.empty else None
    best = passing.loc[best_i]
    chosen = best
    if SEL["tie_break"] == "lightgbm" and lg_best is not None and best["model"] != "lgbm":
        chosen = passing.loc[lg_best]
        rule += "; LightGBM within the tie margin"
    rec.update({k: chosen[k] for k in ("model", "feature_set", "form")}, rule=rule, gate_fail=False,
               value=float(chosen[key]), best_by_value={k: best[k] for k in ("model", "feature_set", "form")},
               best_value=float(best[key]),
               tied=[f"{r.model}/{r.feature_set}/{r.form}" for r in tied.itertuples()])
    return rec


def _row(pool: pd.DataFrame, rec: dict) -> pd.Series:
    m = pool[(pool["target"] == rec["target"]) & (pool["model"] == rec["model"])
             & (pool["feature_set"] == rec["feature_set"]) & (pool["form"] == rec["form"])]
    if len(m) != 1:
        raise AssertionError(f"selected candidate not unique in pool: {rec}")
    return m.iloc[0]


def stage_select(dev: bool, log=print):
    d = out_dir(dev)
    grid = pd.read_csv(d / "grid_table.csv")
    fix4 = pd.read_csv(d / "fix4_table.csv") if (d / "fix4_table.csv").exists() else pd.DataFrame()
    base = baseline_pool(dev, log)
    pool = pd.concat([base, grid, fix4], ignore_index=True)
    sel = {t: select_target(pool[pool["target"] == t], t) for t in PRIMARY_TARGETS}
    for t in CFG["selection"]["secondary_rechecked"]:
        sel[t] = select_target(pool[pool["target"] == t], t)

    # F5: fix 4 for a selected primary candidate whose LOCO median |bias| is still > 5 %
    lim = float(CFG["fix4"]["trigger_value"])
    need = []
    for t in PRIMARY_TARGETS:
        r = _row(pool, sel[t])
        sel[t]["loco_median_abs_bias_pct"] = float(r["loco_median_abs_bias_pct"])
        sel[t]["grouped_design_skill"] = float(r["grouped_design_skill"])
        sel[t]["loco_mape_pct"] = float(r["loco_mape_pct"])
        if r["loco_median_abs_bias_pct"] > lim and sel[t]["feature_set"] != FS_FIX4:
            need.append({"target": t, "model": sel[t]["model"], "form": sel[t]["form"],
                         "loco_median_abs_bias_pct": float(r["loco_median_abs_bias_pct"])})
    fix4_done = not fix4.empty
    pending = [n for n in need if not fix4_done or fix4[(fix4["target"] == n["target"]) & (fix4["model"] == n["model"])
                                                         & (fix4["form"] == n["form"])].empty]
    status = "awaiting_fix4" if pending else "final"

    # M7 on the selected heating and EUI candidates
    m7 = None
    if status == "final":
        vals = {t: sel[t]["loco_median_abs_bias_pct"] for t in M7["targets"]}
        m7 = {"values": vals, "limit": M7["loco_median_abs_bias_pct"],
              "triggered": any(v > M7["loco_median_abs_bias_pct"] for v in vals.values())}
    families = sorted({sel[t]["model"] for t in PRIMARY_TARGETS} | {"mlp"})   # secondary picks stay MLP (3.3)
    result = {"official": not dev, "created_utc": datetime.now(timezone.utc).isoformat(),
              "dataset_sha256_training": CONFIG["dataset_sha256"]["training"],
              "rule": {"selection": SEL, "climate_gen_3_4": CFG["selection"]}, "status": status,
              "selection": sel, "fix4": {"trigger_pct": lim, "needed_for": need, "pending": pending, "run": fix4_done},
              "m7": m7, "families": families, "one_family": len(families) == 1,
              "pool_size": {t: int((pool["target"] == t).sum()) for t in sel}}
    (d / "selection.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    write_report(d, pool, result, dev)
    record_stage(d, "select", {"status": status, "selection": {t: f"{r['model']}/{r['feature_set']}/{r['form']}"
                                                                for t, r in sel.items()}}, dev)
    for t, r in sel.items():
        log(f"  {SHORT[t]:13s} -> {models.LABEL[r['model']]} / {r['feature_set']} / {r['form']}"
            + (f"   LOCO MAPE {r['loco_mape_pct']:.1f} %, median |bias| {r['loco_median_abs_bias_pct']:.1f} %"
               if t in PRIMARY_TARGETS else ""))
    if pending:
        log(f"  F5: fix 4 needed for {[(n['target'], n['model'], n['form']) for n in pending]} -> run --stage fix4, then select")
    elif m7:
        log(f"  M7: {'TRIGGERED - propose Sub-task 2.9' if m7['triggered'] else 'not triggered'} {m7['values']}")
    return result


# ── report ──────────────────────────────────────────────────────────────────────────────────

def _f(x, n=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


def write_report(d: Path, pool: pd.DataFrame, result: dict, dev: bool):
    L = [f"# Sub-task 3.4 — climate generalisation{' [DEV — not a result]' if dev else ''}", "",
         f"Created {result['created_utc']} · status **{result['status']}** · rule: M10 (skill ≥ "
         f"{SEL['primary_skill_min']} → lowest LOCO MAPE; ties ≤ {SEL['tie_mape_points']} points → best LightGBM)", ""]
    L += ["## Selection", "", "| Target | Model | Inputs | Form | Grouped skill | LOCO MAPE % | LOCO median \\|bias\\| % | Rule |",
          "|---|---|---|---|---|---|---|---|"]
    for t, r in result["selection"].items():
        row = _row(pool, r)
        L.append(f"| {SHORT[t]} | {models.LABEL[r['model']]} | {r['feature_set']} | {r['form']} | "
                 f"{_f(row['grouped_design_skill'])} | {_f(row.get('loco_mape_pct'), 1)} | "
                 f"{_f(row.get('loco_median_abs_bias_pct'), 1)} | {r['rule']} |")
    L += ["", f"Fix 4 (F5, trigger > {result['fix4']['trigger_pct']} % median |bias|): needed for "
          f"{result['fix4']['needed_for'] or 'none'}; run: {result['fix4']['run']}.", ""]
    if result["m7"]:
        L.append(f"**M7:** {'TRIGGERED — propose Sub-task 2.9' if result['m7']['triggered'] else 'not triggered'} "
                 f"(limit {result['m7']['limit']} %): " + ", ".join(f"{SHORT[t]} {v:.1f} %" for t, v in result['m7']['values'].items()))
    L += ["", f"Families of the final picks: {', '.join(result['families'])} (one family: {result['one_family']}).", ""]
    for t in PRIMARY_TARGETS + list(CFG["selection"]["secondary_rechecked"]):
        g = pool[pool["target"] == t].sort_values("loco_mape_pct" if t in PRIMARY_TARGETS else "grouped_design_skill",
                                                   ascending=t in PRIMARY_TARGETS)
        L += [f"## {SHORT[t]} — all candidates", "",
              "| Model | Inputs | Form | Grouped skill | Pooled R² | LOCO MAPE % | median \\|bias\\| % | max \\|bias\\| % (climate) "
              "| bias-removed R² | LOCO Spearman |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in g.itertuples():
            L.append(f"| {r.model} | {r.feature_set} | {r.form} | {_f(r.grouped_design_skill)} | {_f(r.grouped_r2)} | "
                     f"{_f(getattr(r, 'loco_mape_pct', None), 1)} | {_f(getattr(r, 'loco_median_abs_bias_pct', None), 1)} | "
                     f"{_f(getattr(r, 'loco_max_abs_bias_pct', None), 1)} ({getattr(r, 'loco_worst_climate', '')}) | "
                     f"{_f(getattr(r, 'loco_bias_removed_r2', None))} | {_f(getattr(r, 'loco_spearman_median', None))} |")
        L.append("")
    (d / "climate_gen_report.md").write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sub-task 3.4 climate generalisation")
    ap.add_argument("--stage", required=True, choices=["indices", "grid", "select", "fix4", "sensitivity", "figures"])
    ap.add_argument("--dev", action="store_true", help="25 %% of designs, small models; writes outputs/climate_gen_dev")
    ap.add_argument("--models", nargs="+", choices=list(CFG["models"]), help="dev only: run a subset of families")
    a = ap.parse_args(argv)
    if a.models and not a.dev:
        ap.error("--models is for dev runs only (the official grid runs every family, F2)")
    log = lambda s: print(s, flush=True)                 # noqa: E731
    t0 = time.time()
    print(f"Climate generalisation stage '{a.stage}'{'  [DEV]' if a.dev else ''}", flush=True)
    if a.stage == "indices":
        stage_indices(a.dev, log)
    elif a.stage == "grid":
        stage_grid(a.dev, log, a.models)
    elif a.stage == "select":
        stage_select(a.dev, log)
    elif a.stage == "sensitivity":
        stage_sensitivity(a.dev, log)
    elif a.stage == "figures":
        stage_figures(a.dev, log)
    else:
        stage_fix4(a.dev, log)
    print(f"{'DEV' if a.dev else 'OFFICIAL'} stage '{a.stage}' done in {time.time() - t0:.0f} s -> "
          f"{out_dir(a.dev).relative_to(OUT.parent.parent)}/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
