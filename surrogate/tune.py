#!/usr/bin/env python3
"""
tune.py - Sub-task 3.3 run B: Optuna tuning under equal budgets (plan M5, claude/Phase3.3_Execution_Plan.md).

One study per (model, target) for RF, XGBoost and LightGBM (60 trials each); one study for the
multi-output MLP (30 trials, objective = mean RMSE/SD over the 3 primary targets, E6). GPR is not
tuned (E5). Every trial is a full 5-fold grouped CV on the 2.8 folds; objective = RMSE on the rows
the target is scored on (heating climates for heating targets). TPE sampler, seed 42, 10 random
start-up trials, no pruning, no time limit. All settings come from config.yaml `benchmark_3_3`.

    ./venv/bin/python surrogate/tune.py --model lgbm                      # 3 primary targets, 60 trials each
    ./venv/bin/python surrogate/tune.py --model lgbm --targets secondary  # run D (7 secondary targets)
    ./venv/bin/python surrogate/tune.py --model mlp                       # one 30-trial study
    ./venv/bin/python surrogate/tune.py --model xgb --feature-set R       # sensitivity_3_3: tune on raw inputs
    ./venv/bin/python surrogate/tune.py --report                          # rebuild report + figures only
    ./venv/bin/python surrogate/tune.py --model rf --dev                  # 3 small trials, tuning_dev/

Resumable: studies live in surrogate/outputs/tuning/optuna.db (SQLite, git-ignored); a rerun only
adds the missing trials. (TPE restarts its random state on resume, so an interrupted study is not
bit-identical to an uninterrupted one; the trial log records every trial either way.)

Writes surrogate/outputs/tuning/ (dev: tuning_dev/):
    trials/<study>.csv      every trial: value, settings, per-target scores, fit time
    best_params.json        best settings per study (full parameter dict, ready for models.make)
    tuning_report.md        best value per study, parameters at range edges, importance
    tuning_convergence.png/.pdf, param_importance.png/.pdf
Training split only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from surrogate import cv, features as FE, metrics as M, models
from surrogate.config import CONFIG, PRIMARY_TARGETS, SECONDARY_TARGETS, ALL_TARGETS
from surrogate.paths import OUT

CFG = CONFIG["benchmark_3_3"]
TUNE = CFG["tuning"]
DEV_TRIALS = 3
SHORT = {"heating_demand_kwh_m2": "Heating", "cooling_demand_kwh_m2": "Cooling", "eui_kwh_m2": "EUI",
         "peak_heating_w_m2": "Peak heating", "peak_cooling_w_m2": "Peak cooling",
         "zone_cooling_core_kwh_m2": "Zone core", "zone_cooling_south_kwh_m2": "Zone south",
         "zone_cooling_east_kwh_m2": "Zone east", "zone_cooling_north_kwh_m2": "Zone north",
         "zone_cooling_west_kwh_m2": "Zone west"}
MULTI_STUDY = "all"                                       # MLP study name suffix (one net, all targets)


def out_dir(dev: bool) -> Path:
    return OUT / ("tuning_dev" if dev else "tuning")


SENS = CONFIG.get("sensitivity_3_3", {})


def study_name(model: str, target: str, feature_set: str | None = None) -> str:
    """'<model>__<target>'; studies on another feature set (sensitivity_3_3, raw inputs R) get '__<set>'."""
    base = f"{model}__{target}"
    return base if feature_set in (None, CFG["feature_set"]) else f"{base}__{feature_set}"


def budget(model: str, dev: bool) -> int:
    n = int(TUNE["trials"].get(model, 0))
    if n <= 0:
        raise ValueError(f"{model} is not tuned by Optuna (plan E5)")
    return DEV_TRIALS if dev else n


# ── feature set per model (3.2 D5 / plan E3) ────────────────────────────────────────────────

def feature_set_for(model: str) -> str:
    """C-G unless benchmark.py's D5 check gave this model its own set (outputs/benchmark/feature_sets.json)."""
    path = OUT / "benchmark" / "feature_sets.json"
    if path.exists():
        sets = json.loads(path.read_text(encoding="utf-8")).get("feature_sets", {})
        if model in sets:
            return sets[model]
    return CFG["feature_set"]


# ── objective ───────────────────────────────────────────────────────────────────────────────

def scored_rmse(oof: pd.DataFrame, target: str, heat: list[str]) -> float:
    g = M.scored_rows(oof[oof["target"] == target], target, heat)
    return float(np.sqrt(np.mean((g["yhat"].to_numpy() - g["y"].to_numpy()) ** 2)))


def scored_sd(df: pd.DataFrame, target: str, heat: list[str]) -> float:
    g = df[df["climate_key"].isin(heat)] if target in M.HEATING_TARGETS else df
    return float(g[target].std(ddof=0))


def dev_shrink(model: str, params: dict) -> dict:
    """Dev runs only: small models so the code path is checked in seconds."""
    p = dict(params)
    if "n_estimators" in p:
        p["n_estimators"] = min(int(p["n_estimators"]), 40)
    if model == "mlp":
        p.update(max_epochs=15, patience=5)
    return p


def evaluate(model: str, params: dict, df, X, targets: list[str], heat, folds, scheme="grouped") -> pd.DataFrame:
    """Out-of-fold predictions for one setting (single-output per target, or one multi-output net)."""
    if model in models.MULTI_OUTPUT:
        return cv.cross_val_predict_multi(df, X, ALL_TARGETS, models.make_multi(model, params), folds, scheme)
    return cv.cross_val_predict(df, X, targets, models.make(model, params), folds, scheme)


def make_objective(model, target, df, X, heat, folds, dev, sds):
    """target = a target name, or MULTI_STUDY for the MLP."""
    def objective(trial):
        params = models.suggest(trial, model)
        if dev:
            params = dev_shrink(model, params)
        t0 = time.perf_counter()
        oof = evaluate(model, params, df, X, [target] if target != MULTI_STUDY else ALL_TARGETS, heat, folds)
        trial.set_user_attr("cv_seconds", round(time.perf_counter() - t0, 2))
        scored = [target] if target != MULTI_STUDY else ALL_TARGETS
        summ, _ = M.summarise(oof, heat)
        summ = summ.set_index("target")
        for t in scored:
            trial.set_user_attr(f"rmse__{t}", scored_rmse(oof, t, heat))
            trial.set_user_attr(f"skill__{t}", float(summ.loc[t, "design_skill"]))
        if target == MULTI_STUDY:                     # E6: mean RMSE / SD over the primary targets
            return float(np.mean([scored_rmse(oof, t, heat) / sds[t] for t in PRIMARY_TARGETS]))
        return scored_rmse(oof, target, heat)
    return objective


# ── running studies ─────────────────────────────────────────────────────────────────────────

def _storage(dev: bool) -> str:
    d = out_dir(dev)
    d.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(d / 'optuna.db').as_posix()}"


def run_study(model: str, target: str, df, X, heat, dev: bool, log=print, feature_set: str | None = None):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_total = budget(model, dev)
    sampler = optuna.samplers.TPESampler(seed=int(TUNE["seed"]), n_startup_trials=int(TUNE["n_startup_trials"]))
    fs = feature_set or (feature_set_for(model) if not dev else CFG["feature_set"])
    study = optuna.create_study(study_name=study_name(model, target, fs), storage=_storage(dev), direction="minimize",
                                sampler=sampler, load_if_exists=True)
    study.set_user_attr("model", model)
    study.set_user_attr("target", target)
    study.set_user_attr("feature_set", fs)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    todo = n_total - len(done)
    if todo <= 0:
        log(f"  {study.study_name}: complete ({len(done)}/{n_total} trials), best {study.best_value:.4f}")
        return study
    folds = cv.grouped_folds(df)
    sds = {t: scored_sd(df, t, heat) for t in PRIMARY_TARGETS}
    obj = make_objective(model, target, df, X, heat, folds, dev, sds)
    t_study = time.time()

    def cb(st, tr):
        k = len([t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE])
        log(f"  {st.study_name} trial {k:3d}/{n_total}  value {tr.value:.4f}  best {st.best_value:.4f}  "
            f"({tr.user_attrs.get('cv_seconds', 0):.1f} s; study {time.time() - t_study:6.0f} s)")
    study.optimize(obj, n_trials=todo, callbacks=[cb], gc_after_trial=True)
    return study


def trials_frame(study) -> pd.DataFrame:
    import optuna
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        r = {"trial": t.number, "value": t.value, "duration_s": t.duration.total_seconds() if t.duration else None}
        r.update({f"param__{k}": v for k, v in t.params.items()})
        r.update(t.user_attrs)
        rows.append(r)
    return pd.DataFrame(rows)


def best_record(study, dev: bool) -> dict:
    model, target = study.user_attrs["model"], study.user_attrs["target"]
    bt = study.best_trial
    params = models.tuned_params(model, dict(bt.params))
    if dev:
        params = dev_shrink(model, params)
    n = len(trials_frame(study))
    return {"model": model, "target": target, "feature_set": study.user_attrs.get("feature_set", CFG["feature_set"]),
            "objective": TUNE["mlp_objective"] if model == "mlp" else TUNE["objective"],
            "best_value": float(bt.value), "best_trial": bt.number, "n_trials": n,
            "params": params, "searched": dict(bt.params),
            "scores": {k: v for k, v in bt.user_attrs.items() if k.startswith(("rmse__", "skill__"))},
            "dev": dev}


def save_study(study, dev: bool):
    d = out_dir(dev)
    (d / "trials").mkdir(parents=True, exist_ok=True)
    trials_frame(study).to_csv(d / "trials" / f"{study.study_name}.csv", index=False)
    path = d / "best_params.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"studies": {}}
    data["studies"][study.study_name] = best_record(study, dev)
    data["updated_utc"] = datetime.now(timezone.utc).isoformat()
    data["dataset_sha256_training"] = CONFIG["dataset_sha256"]["training"]
    path.write_text(json.dumps(data, indent=2, default=float), encoding="utf-8")


def load_best(dev: bool = False) -> dict:
    """{study_name: record} from best_params.json (used by benchmark.py)."""
    path = out_dir(dev) / "best_params.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))["studies"]


# ── report ──────────────────────────────────────────────────────────────────────────────────

def edge_flags(model: str, searched: dict, tol: float = 0.02) -> list[str]:
    """Searched parameters whose best value lies within 2 % of a range end (log-aware)."""
    flags = []
    for name, s in models.search_space(model).items():
        if s["type"] not in ("int", "float") or name not in searched:
            continue
        lo, hi, v = float(s["low"]), float(s["high"]), float(searched[name])
        if s.get("log"):
            lo, hi, v = math.log(lo), math.log(hi), math.log(v)
        pos = (v - lo) / (hi - lo) if hi > lo else 0.5
        if pos <= tol:
            flags.append(f"{name} at low end ({searched[name]})")
        elif pos >= 1 - tol:
            flags.append(f"{name} at high end ({searched[name]})")
    return flags


def importances(study) -> dict:
    import optuna
    try:
        return optuna.importance.get_param_importances(
            study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=int(TUNE["seed"])))
    except Exception:                                  # too few trials / one parameter
        return {}


def report(dev: bool, log=print):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    d = out_dir(dev)
    db = d / "optuna.db"
    if not db.exists():
        log("no studies yet")
        return
    storage = _storage(dev)
    summaries = optuna.study.get_all_study_summaries(storage)
    studies = [optuna.load_study(study_name=s.study_name, storage=storage) for s in summaries]
    studies = [s for s in studies if len(trials_frame(s))]
    for s in studies:
        save_study(s, dev)
    L = [f"# Hyper-parameter tuning (Sub-task 3.3, run B){' — DEV RUN, not official' if dev else ''}", "",
         f"Updated {datetime.now(timezone.utc).isoformat()[:19]} UTC. Optuna TPE (seed {TUNE['seed']}, "
         f"{TUNE['n_startup_trials']} random start-up trials), no pruning, budgets {TUNE['trials']}. "
         "Objective: grouped-CV RMSE on scored rows (MLP: mean RMSE/SD over the 3 primary targets). "
         "Search spaces: config.yaml `benchmark_3_3.search_spaces`.", "",
         "| study | trials | best objective | best trial | design skill at best | trials to reach best + 1 % | parameters at a range edge |",
         "|---|---|---|---|---|---|---|"]
    imp_all = {}
    for s in sorted(studies, key=lambda s: s.study_name):
        tf = trials_frame(s)
        model, target = s.user_attrs["model"], s.user_attrs["target"]
        bt = s.best_trial
        skills = ", ".join(f"{SHORT.get(k[7:], k[7:])} {v:.3f}" for k, v in bt.user_attrs.items()
                           if k.startswith("skill__") and (target != MULTI_STUDY or k[7:] in PRIMARY_TARGETS))
        best_so_far = tf.sort_values("trial")["value"].cummin().to_numpy()
        reach = int(np.argmax(best_so_far <= bt.value * 1.01)) + 1
        flags = edge_flags(model, bt.params)
        L.append(f"| {s.study_name} | {len(tf)} | {bt.value:.4f} | {bt.number} | {skills} | {reach} | "
                 f"{'; '.join(flags) if flags else '—'} |")
        imp_all[s.study_name] = importances(s)
    L += ["", "A parameter at a range edge means the best value may lie outside the searched range; it is "
          "reported (sensitivity note), not acted on — the ranges are frozen.", "",
          "## Parameter importance (fANOVA share of objective variance)", ""]
    for name, imp in imp_all.items():
        if imp:
            L.append(f"- **{name}**: " + ", ".join(f"{k} {v:.2f}" for k, v in list(imp.items())[:5]))
    (d / "tuning_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    figures(d, studies, imp_all)
    log(f"report: {d / 'tuning_report.md'}")


def figures(d: Path, studies, imp_all):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from surrogate.benchmark import COLORS, _style
    if not studies:
        return
    names = sorted(s.study_name for s in studies)
    by = {s.study_name: s for s in studies}
    n = len(names)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.5 * rows), squeeze=False)
    for i, name in enumerate(names):
        ax = axes[i // cols][i % cols]
        _style(ax)
        tf = trials_frame(by[name]).sort_values("trial")
        model = by[name].user_attrs["model"]
        ax.plot(tf["trial"] + 1, tf["value"], "o", ms=3, color="#b5b4ad", label="trial")
        ax.plot(tf["trial"] + 1, tf["value"].cummin(), "-", lw=2, color=COLORS[model], label="best so far")
        tgt = by[name].user_attrs["target"]
        fs_tag = by[name].user_attrs.get("feature_set", CFG["feature_set"])
        ax.set_title(f"{models.LABEL[model]} · {SHORT.get(tgt, 'all targets')}"
                     + (f" · {fs_tag} inputs" if fs_tag != CFG["feature_set"] else ""),
                     fontsize=8.5, loc="left", color="#22221f")
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.set_xlabel("trial", fontsize=8, color="#3a3a36")
        lo, hi = tf["value"].min(), tf["value"].quantile(0.9)
        ax.set_ylim(lo - 0.05 * (hi - lo + 1e-9), hi + 0.05 * (hi - lo + 1e-9))
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    axes[0][0].set_ylabel("objective (lower = better)", fontsize=8, color="#3a3a36")
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(d / f"tuning_convergence.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    imp_names = [k for k in names if imp_all.get(k)]
    if not imp_names:
        return
    params = sorted({p for k in imp_names for p in imp_all[k]})
    fig, ax = plt.subplots(figsize=(1.2 + 0.55 * len(params), 0.5 + 0.32 * len(imp_names)))
    M_ = np.array([[imp_all[k].get(p, np.nan) for p in params] for k in imp_names])
    im = ax.imshow(M_, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(params)), params, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(imp_names)), [k.replace("__", " · ") for k in imp_names], fontsize=7)
    for (r, c), v in np.ndenumerate(M_):
        if np.isfinite(v) and v >= 0.1:
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.5 else "#22221f")
    fig.colorbar(im, ax=ax, fraction=0.03, label="importance share")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(d / f"param_importance.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────────────────────────

def resolve_targets(model: str, arg: list[str]) -> list[str]:
    if model in models.MULTI_OUTPUT:
        return [MULTI_STUDY]
    out = []
    for a in arg:
        if a == "primary":
            out += PRIMARY_TARGETS
        elif a == "secondary":
            out += SECONDARY_TARGETS
        elif a in ALL_TARGETS:
            out.append(a)
        else:
            raise SystemExit(f"unknown target {a!r}")
    return list(dict.fromkeys(out))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sub-task 3.3 Optuna tuning (equal budgets)")
    ap.add_argument("--model", choices=list(models.TUNABLE))
    ap.add_argument("--targets", nargs="+", default=["primary"], help="primary | secondary | target names")
    ap.add_argument("--dev", action="store_true", help=f"{DEV_TRIALS} small trials per study, writes outputs/tuning_dev")
    ap.add_argument("--report", action="store_true", help="only rebuild report, best_params.json and figures")
    ap.add_argument("--feature-set", choices=[SENS.get("feature_set", "R")], default=None,
                    help="sensitivity_3_3 only: tune on the raw inputs (studies named <model>__<target>__R)")
    a = ap.parse_args(argv)
    if a.report:
        report(a.dev)
        return 0
    if not a.model:
        ap.error("--model is required (or --report)")
    df, _ = cv.load_training()
    heat = M.heating_climates(df)
    if a.feature_set and a.model not in SENS.get("models", []):
        ap.error(f"--feature-set {a.feature_set} is only for {SENS.get('models')} (config sensitivity_3_3)")
    fs = a.feature_set or (CFG["feature_set"] if a.dev else feature_set_for(a.model))
    X = FE.engineer(df)[FE.feature_set(fs)]
    targets = resolve_targets(a.model, a.targets)
    n = budget(a.model, a.dev)
    print(f"Tuning {models.LABEL[a.model]} on {fs} ({X.shape[1]} features): {len(targets)} "
          f"stud{'y' if len(targets) == 1 else 'ies'} x {n} trials x 5 folds{'  [DEV]' if a.dev else ''}", flush=True)
    t0 = time.time()
    for t in targets:
        st = run_study(a.model, t, df, X, heat, a.dev, log=lambda s: print(s, flush=True), feature_set=fs)
        save_study(st, a.dev)
    print(f"done in {time.time() - t0:.0f} s", flush=True)
    report(a.dev, log=lambda s: print(s, flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
