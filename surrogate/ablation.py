#!/usr/bin/env python3
"""
ablation.py - Sub-task 3.2 step 4: feature-set ablation (claude/Phase3.2_Execution_Plan.md D2-D4).

8 feature sets (ladder R -> G -> S -> U -> C and leave-one-group-out C-G, C-S, C-U)
x 2 untuned models (RF = the 2.8/S0 forest, LightGBM = library defaults; config.yaml ablation_3_2)
x 3 primary targets (direct form), each scored by grouped CV (5 folds, selection) and
leave-one-climate-out (10 folds, reported only). 720 temporary fits; only scores are kept.

Selection (pre-registered, D3/D4): score = mean grouped-CV design skill over the 3 primary
targets and both models; the best score wins; sets within 0.005 of it count as tied and the
one with the fewest features is chosen. LOCO is never used to choose.

    ./venv/bin/python surrogate/ablation.py                  # official run (Mac)
    ./venv/bin/python surrogate/ablation.py --dev            # quick code check: 30 RF trees, separate folder

Writes surrogate/outputs/ablation/ (dev: surrogate/outputs/ablation_dev/):
    ablation_table.csv         one row per set x model x target: grouped and LOCO metrics
    ablation_per_climate.csv   per held-out / scored climate (both protocols)
    ablation_result.json       D4 score per set, winner, settings, dataset sha256, versions
    ablation_report.md         tables in plain form
    ablation_ladder.png/.pdf   design skill and LOCO MAPE along the ladder
    ablation_leave_one_out.png/.pdf   loss of skill when each group is removed from the full set
Training split only.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np
import pandas as pd

from surrogate import cv, features as FE, metrics as M
from surrogate.config import CONFIG, PRIMARY_TARGETS
from surrogate.paths import OUT

CFG = CONFIG["ablation_3_2"]
MODEL_LABEL = {"rf": "Random Forest", "lgbm": "LightGBM"}
SHORT = {"heating_demand_kwh_m2": "Heating", "cooling_demand_kwh_m2": "Cooling", "eui_kwh_m2": "EUI"}
SET_LABEL = {"R": "R  raw 28", "G": "G  + geometry", "S": "S  + sun", "U": "U  + heat loss",
             "C": "C  + climate x building", "C-G": "C - geometry", "C-S": "C - sun", "C-U": "C - heat loss"}
GROUP_NAME = {"G": "geometry", "S": "sun/orientation", "U": "heat loss", "C": "climate x building"}
COLORS = {"rf": "#2a78d6", "lgbm": "#eb6834"}                 # validated categorical slots 1-2
MARKERS = {"rf": "o", "lgbm": "s"}


# ── models ──────────────────────────────────────────────────────────────────────────────────

def model_params(model: str, dev: bool = False) -> dict:
    if model == "rf":
        p = dict(CONFIG["reference_2_8"]["rf_params"])
        if dev:
            p["n_estimators"] = 30
        return p
    if model == "lgbm":
        return dict(CFG["lgbm_params"])
    raise KeyError(model)


def factory(model: str, params: dict):
    if model == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return lambda target: RandomForestRegressor(**params)
    from lightgbm import LGBMRegressor
    return lambda target: LGBMRegressor(**params)


# ── selection rule (D3/D4) ──────────────────────────────────────────────────────────────────

def choose_winner(table: pd.DataFrame, tie: float | None = None) -> tuple[pd.DataFrame, str]:
    """scores per set (mean grouped design skill over targets x models) and the winning set."""
    tie = CFG["selection"]["tie_skill"] if tie is None else tie
    s = (table.groupby("feature_set")
         .agg(score=("grouped_design_skill", "mean"), n_features=("n_features", "first"), n=("grouped_design_skill", "size"))
         .reset_index())
    best = s["score"].max()
    s["tied_with_best"] = s["score"] >= best - tie - 1e-12
    pool = s[s["tied_with_best"]].sort_values(["n_features", "score"], ascending=[True, False])
    winner = str(pool.iloc[0]["feature_set"])
    s["winner"] = s["feature_set"] == winner
    order = {k: i for i, k in enumerate(FE.FEATURE_SETS)}
    s = s.sort_values("feature_set", key=lambda c: c.map(order)).reset_index(drop=True)
    return s, winner


# ── run ─────────────────────────────────────────────────────────────────────────────────────

def run(sets: list[str], models: list[str], protocols: list[str], dev: bool = False, log=print):
    df, cols = cv.load_training()
    heat = M.heating_climates(df)
    F = FE.engineer(df)
    rows, per_climate = [], []
    n_combo = len(sets) * len(models)
    k = 0
    t_all = time.time()
    for fs in sets:
        X = F[FE.feature_set(fs)]
        for model in models:
            k += 1
            params = model_params(model, dev)
            t0 = time.time()
            res = {t: {"feature_set": fs, "model": model, "target": t, "n_features": X.shape[1]} for t in PRIMARY_TARGETS}
            if "grouped" in protocols:
                oof = cv.run_grouped(df, X, PRIMARY_TARGETS, factory(model, params))
                summ, pc = M.summarise(oof, heat)
                for _, r in summ.iterrows():
                    res[r["target"]].update({f"grouped_{c}": r[c] for c in
                                             ("design_skill", "r2", "mae", "rmse", "cv_rmse", "spearman_median", "top_hit_mean")})
                    res[r["target"]]["grouped_fit_s"] = float(oof.loc[oof.target == r["target"]].groupby("fold")["fit_s"].first().sum())
                pc = pc.assign(feature_set=fs, model=model, scheme="grouped")
                per_climate.append(pc)
            if "loco" in protocols:
                oof = cv.run_loco(df, X, PRIMARY_TARGETS, factory(model, params))
                summ, tab = M.loco_summary(oof, heat)
                for _, r in summ.iterrows():
                    res[r["target"]].update({f"loco_{c}": r[c] for c in
                                             ("median_abs_bias_pct", "max_abs_bias_pct", "worst_climate", "mape_pct",
                                              "bias_removed_r2", "spearman_median", "top_hit_mean")})
                per_climate.append(tab.assign(feature_set=fs, model=model, scheme="loco"))
            rows += list(res.values())
            msg = "  ".join(f"{SHORT[t]} {res[t].get('grouped_design_skill', float('nan')):.3f}" for t in PRIMARY_TARGETS)
            log(f"[{k:2d}/{n_combo}] {fs:4s} {model:5s} ({X.shape[1]:2d} features)  design skill: {msg}"
                f"   {time.time() - t0:5.1f} s  (total {time.time() - t_all:5.0f} s)")
    table = pd.DataFrame(rows)
    per_climate = pd.concat(per_climate, ignore_index=True) if per_climate else pd.DataFrame()
    return df, heat, table, per_climate


# ── report ──────────────────────────────────────────────────────────────────────────────────

def _f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def write_report(path, record, table, scores, winner):
    T = PRIMARY_TARGETS
    L = [f"# Feature ablation (Sub-task 3.2){' — DEV RUN, not official' if record['dev'] else ''}", "",
         f"Run {record['created_utc'][:19]} UTC, {record['runtime_s']:.0f} s, dataset sha256 "
         f"`{record['dataset_sha256_training'][:12]}…`, {record['n_fits']} temporary fits. "
         "Untuned models (RF = 2.8 forest, LightGBM = library defaults). Training split only.", "",
         "## 1. Selection score (D4) — mean grouped-CV design skill over 3 targets × 2 models", "",
         "Design skill = share of the differences between designs *within a climate* that the model explains "
         "(heating over the 7 heating climates).", "",
         "| set | features | score | tied with best (±0.005) | chosen |", "|---|---|---|---|---|"]
    for _, r in scores.iterrows():
        L.append(f"| {SET_LABEL[r.feature_set]} | {int(r.n_features)} | {r.score:.4f} | "
                 f"{'yes' if r.tied_with_best else ''} | {'**WINNER**' if r.winner else ''} |")
    L += ["", f"**Winner: {SET_LABEL[winner]}** ({int(scores.loc[scores.winner, 'n_features'].iloc[0])} features) — "
          "highest score; among sets within 0.005 of the best, the one with the fewest features.", ""]

    def block(title, col, d=3, sets=None):
        sets = sets or list(FE.FEATURE_SETS)
        out = [f"## {title}", "", "| set | " + " | ".join(f"{MODEL_LABEL[m]} {SHORT[t]}" for m in record["models"] for t in T) + " |",
               "|---|" + "---|" * (len(T) * len(record["models"]))]
        for fs in sets:
            vals = []
            for m in record["models"]:
                for t in T:
                    sel = table[(table.feature_set == fs) & (table.model == m) & (table.target == t)]
                    vals.append(_f(float(sel[col].iloc[0]), d) if len(sel) and col in sel else "—")
            out.append(f"| {SET_LABEL[fs]} | " + " | ".join(vals) + " |")
        return out + [""]

    L += block("2. Grouped CV — design skill (higher is better)", "grouped_design_skill")
    L += block("3. Grouped CV — pooled R²", "grouped_r2")
    L += block("4. Grouped CV — within-climate ranking (median Spearman ρ)", "grouped_spearman_median")
    # leave-one-out drops
    L += ["## 5. Leave one group out — loss of design skill when a group is removed from the full set C", "",
          "| removed group | " + " | ".join(f"{MODEL_LABEL[m]} {SHORT[t]}" for m in record["models"] for t in T) + " | mean |",
          "|---|" + "---|" * (len(T) * len(record["models"]) + 1)]
    full = table[table.feature_set == "C"].set_index(["model", "target"])["grouped_design_skill"]
    for g, fs in (("G", "C-G"), ("S", "C-S"), ("U", "C-U"), ("C", "U")):
        part = table[table.feature_set == fs].set_index(["model", "target"])["grouped_design_skill"]
        if part.empty or full.empty:
            continue
        d = (full - part)
        vals = [_f(float(d.loc[(m, t)]), 4) for m in record["models"] for t in T]
        L.append(f"| {GROUP_NAME[g]} ({fs if fs != 'U' else 'C − climate × building = U'}) | " + " | ".join(vals) + f" | {_f(float(d.mean()), 4)} |")
    L += ["", "Positive = skill is lost without that group (the group carries information no other group has); "
          "near zero or negative = the group adds nothing beyond the others.", ""]
    if "loco_mape_pct" in table:
        L += ["## 6. Unseen climate (leave-one-climate-out) — REPORTED ONLY, not used to choose", ""]
        L += block("6a. LOCO MAPE % (lower is better)", "loco_mape_pct", 1)
        L += block("6b. LOCO median |climate bias| %", "loco_median_abs_bias_pct", 1)
        L += block("6c. LOCO bias-removed R² (ranking quality in an unseen climate)", "loco_bias_removed_r2")
    L += ["## Settings", "", f"- RF: `{json.dumps(record['params']['rf'])}`" if "rf" in record["params"] else "",
          f"- LightGBM: `{json.dumps(record['params']['lgbm'])}`" if "lgbm" in record["params"] else "",
          f"- Environment: Python {record['environment']['python']}, "
          + ", ".join(f"{k} {v}" for k, v in record["environment"].items() if k not in ("python", "platform"))]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ── figures ─────────────────────────────────────────────────────────────────────────────────

def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8a8a85")
    ax.tick_params(colors="#55554f", labelsize=8)
    ax.grid(axis="y", color="#e4e3dd", linewidth=0.6)
    ax.set_axisbelow(True)


def figures(out_dir, table, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ladder = [s for s in FE.LADDER if s in set(table.feature_set)]
    has_loco = "loco_mape_pct" in table
    rows = 2 if has_loco else 1
    fig, axes = plt.subplots(rows, 3, figsize=(10, 3.3 * rows), squeeze=False)
    for j, t in enumerate(PRIMARY_TARGETS):
        for i, (col, ylab) in enumerate([("grouped_design_skill", "Design skill (grouped CV)"),
                                         ("loco_mape_pct", "MAPE % (unseen climate, LOCO)")][:rows]):
            ax = axes[i, j]
            _style(ax)
            for m in models:
                sub = table[(table.model == m) & (table.target == t)].set_index("feature_set").loc[ladder]
                ax.plot(range(len(ladder)), sub[col], color=COLORS[m], marker=MARKERS[m], markersize=6,
                        linewidth=2, label=MODEL_LABEL[m])
                ax.annotate(f"{sub[col].iloc[-1]:.{3 if i == 0 else 1}f}", (len(ladder) - 1, sub[col].iloc[-1]),
                            xytext=(6, 0), textcoords="offset points", va="center", fontsize=7.5, color="#3a3a36")
            ax.set_xticks(range(len(ladder)), ladder)
            ax.set_xlim(-0.3, len(ladder) - 0.4)
            if i == 0:
                ax.set_title(SHORT[t], fontsize=10, color="#22221f", loc="left")
            if j == 0:
                ax.set_ylabel(ylab, fontsize=8.5, color="#3a3a36")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.supxlabel("Feature set (cumulative): R raw · G +geometry · S +sun · U +heat loss · C +climate×building",
                  fontsize=8.5, color="#3a3a36")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"ablation_ladder.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    if not {"C", "C-G", "C-S", "C-U", "U"} <= set(table.feature_set):
        return
    full = table[table.feature_set == "C"].set_index(["model", "target"])["grouped_design_skill"]
    groups = [("G", "C-G"), ("S", "C-S"), ("U", "C-U"), ("C", "U")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), sharey=True)
    w = 0.38
    for j, t in enumerate(PRIMARY_TARGETS):
        ax = axes[j]
        _style(ax)
        ax.axhline(0, color="#8a8a85", linewidth=0.8)
        for k, m in enumerate(models):
            vals = [float(full.loc[(m, t)] - table[(table.feature_set == fs) & (table.model == m) & (table.target == t)]
                          ["grouped_design_skill"].iloc[0]) for _, fs in groups]
            x = np.arange(len(groups)) + (k - (len(models) - 1) / 2) * w
            ax.bar(x, vals, width=w - 0.04, color=COLORS[m], label=MODEL_LABEL[m], edgecolor="white", linewidth=1)
        ax.set_xticks(range(len(groups)), [GROUP_NAME[g].replace(" x ", "×\n").replace("/", "/\n") for g, _ in groups], fontsize=7.5)
        ax.set_title(SHORT[t], fontsize=10, color="#22221f", loc="left")
    axes[0].set_ylabel("Design skill lost when removed", fontsize=8.5, color="#3a3a36")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"ablation_leave_one_out.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sub-task 3.2 feature ablation")
    ap.add_argument("--dev", action="store_true", help="30 RF trees, writes outputs/ablation_dev (not official)")
    ap.add_argument("--sets", nargs="+", default=CFG["feature_sets"], choices=list(FE.FEATURE_SETS))
    ap.add_argument("--models", nargs="+", default=CFG["models"], choices=["rf", "lgbm"])
    ap.add_argument("--protocols", nargs="+", default=CFG["protocols"], choices=["grouped", "loco"])
    a = ap.parse_args(argv)
    official = (not a.dev and a.sets == CFG["feature_sets"] and a.models == CFG["models"]
                and a.protocols == CFG["protocols"])
    out_dir = OUT / ("ablation" if official else "ablation_dev")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_fits = len(a.sets) * len(a.models) * len(PRIMARY_TARGETS) * (5 * ("grouped" in a.protocols) + 10 * ("loco" in a.protocols))
    print(f"Feature ablation: {len(a.sets)} sets x {len(a.models)} models x {len(PRIMARY_TARGETS)} targets, "
          f"protocols {a.protocols} -> {n_fits} fits{'  [DEV]' if not official else ''}", flush=True)
    df, heat, table, per_climate = run(a.sets, a.models, a.protocols, dev=a.dev,
                                       log=lambda s: print(s, flush=True))
    runtime = time.time() - t0

    scores, winner = (choose_winner(table) if "grouped" in a.protocols else (pd.DataFrame(), None))
    params = {m: model_params(m, a.dev) for m in a.models}
    pkgs = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "matplotlib")
    record = {"step": "3.2 feature ablation", "official": official, "dev": a.dev,
              "created_utc": datetime.now(timezone.utc).isoformat(), "runtime_s": round(runtime, 1),
              "dataset_sha256_training": CONFIG["dataset_sha256"]["training"], "rows": len(df),
              "heating_climates": heat, "feature_sets": a.sets, "models": a.models, "protocols": a.protocols,
              "targets": PRIMARY_TARGETS, "n_fits": n_fits, "params": params,
              "selection_rule": CFG["selection"],
              "scores": scores.to_dict(orient="records"), "winner": winner,
              "winner_features": FE.feature_set(winner) if winner else None,
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              **{p: version(p) for p in pkgs}}}
    table.to_csv(out_dir / "ablation_table.csv", index=False)
    per_climate.to_csv(out_dir / "ablation_per_climate.csv", index=False)
    (out_dir / "ablation_result.json").write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")
    if winner:
        write_report(out_dir / "ablation_report.md", record, table, scores, winner)
        figures(out_dir, table, a.models)

    print()
    if winner:
        print(scores[["feature_set", "n_features", "score", "tied_with_best", "winner"]].to_string(index=False, float_format="%.4f"))
        print(f"\nWINNER: {winner} ({len(FE.feature_set(winner))} features)")
    print(f"{'OFFICIAL' if official else 'DEV'} run, {runtime:.0f} s - wrote {out_dir.relative_to(OUT.parent.parent)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
