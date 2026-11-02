"""
metrics.py - Phase 3 scores (plan M2): a thin layer over eda_lib plus the ranking and
unseen-climate metrics that Phase 4 depends on.

Reused from 2.8 unchanged (so numbers stay comparable):
    regression_metrics (pooled R2, MAE, RMSE, CV(RMSE)), design_skill, heating_climates,
    evaluate_predictions (per-target summary + per-climate table).

Added in Phase 3:
    ranking_by_climate  within-climate Spearman rho and top-10 % hit rate
    summarise           2.8 summary + median Spearman and hit rate per target
    loco_summary        per held-out climate: bias %, MAPE %, bias-removed R2, ranking
    m7_triggered        the pre-registered Sub-task 2.9 trigger

Heating targets (heating demand, peak heating) are scored over the heating climates only, as in 2.8.
"Best" design = lowest value (every target is an energy or load to minimise).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from eda import eda_lib as L
from surrogate.config import CONFIG

regression_metrics = L.regression_metrics
design_skill = L.design_skill
heating_climates = L.heating_climates
HEATING_TARGETS = tuple(CONFIG["targets"]["heating"])
TOP_FRACTION = CONFIG["metrics"]["top_fraction"]
MAPE_MIN_Y = CONFIG["metrics"]["mape_min_y"]


def scored_rows(g: pd.DataFrame, target: str, heat: list[str]) -> pd.DataFrame:
    """Rows a target is scored on: heating climates only for heating targets."""
    return g[g["climate_key"].isin(heat)] if target in HEATING_TARGETS else g


# ── ranking ───────────────────────────────────────────────────────────────────

def top_hit_rate(y, yhat, fraction: float = TOP_FRACTION) -> float:
    """Share of the truly best `fraction` of designs (lowest y) that the model also puts there."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    k = max(1, int(round(fraction * len(y))))
    true_best = set(np.argsort(y, kind="stable")[:k])
    pred_best = set(np.argsort(yhat, kind="stable")[:k])
    return len(true_best & pred_best) / k


def spearman(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    if len(y) < 3 or np.ptp(y) == 0 or np.ptp(yhat) == 0:
        return float("nan")
    return float(spearmanr(y, yhat).statistic)


def ranking_by_climate(oof: pd.DataFrame, heat: list[str], fraction: float = TOP_FRACTION) -> pd.DataFrame:
    """target, climate_key, n, spearman, top_hit - within each climate, scored climates only."""
    rows = []
    for t, g in oof.groupby("target", sort=False):
        for c, gc in scored_rows(g, t, heat).groupby("climate_key"):
            rows.append({"target": t, "climate_key": c, "n": len(gc), "spearman": spearman(gc["y"], gc["yhat"]),
                         "top_hit": top_hit_rate(gc["y"], gc["yhat"], fraction)})
    return pd.DataFrame(rows)


def summarise(oof: pd.DataFrame, heat: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(summary, per_climate): eda_lib.evaluate_predictions + spearman_median, top_hit_mean."""
    summary, per_climate = L.evaluate_predictions(oof, heat)
    rank = ranking_by_climate(oof, heat)
    agg = rank.groupby("target").agg(spearman_median=("spearman", "median"), top_hit_mean=("top_hit", "mean"))
    summary = summary.merge(agg, left_on="target", right_index=True, how="left")
    per_climate = per_climate.merge(rank[["target", "climate_key", "spearman", "top_hit"]],
                                    on=["target", "climate_key"], how="left")
    return summary, per_climate


# ── unseen climates (LOCO) ────────────────────────────────────────────────────

def bias_pct(y, yhat) -> float:
    """Climate-level bias: 100 * (mean prediction - mean truth) / mean truth."""
    my = float(np.mean(y))
    return 100.0 * (float(np.mean(yhat)) - my) / my if my != 0 else float("nan")


def mape_pct(y, yhat, min_y: float = MAPE_MIN_Y) -> float:
    """Mean absolute percentage error over rows with |y| >= min_y."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    keep = np.abs(y) >= min_y
    return float(100.0 * np.mean(np.abs(yhat[keep] - y[keep]) / np.abs(y[keep]))) if keep.any() else float("nan")


def loco_summary(oof: pd.DataFrame, heat: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(summary per target, table per target x held-out climate) for leave-one-climate-out predictions.

    per climate : n, bias_pct, abs_bias_pct, mape_pct, r2_bias_removed (within-climate R2 after removing
                  that climate's mean error = ranking/shape quality), spearman, top_hit
    summary     : median / max |bias| % and worst climate, pooled MAPE % (all scored rows),
                  bias_removed_r2 (design skill of the bias-corrected predictions, pooled over climates),
                  spearman_median, top_hit_mean
    """
    if (oof["scheme"] != "loco").any():
        raise ValueError("loco_summary expects predictions from cv.run_loco (scheme 'loco')")
    table, summary = [], []
    for t, g in oof.groupby("target", sort=False):
        g = scored_rows(g, t, heat)
        corrected = g["yhat"] - g.groupby("climate_key")["yhat"].transform("mean") \
            + g.groupby("climate_key")["y"].transform("mean")
        for c, gc in g.groupby("climate_key"):
            b = bias_pct(gc["y"], gc["yhat"])
            table.append({"target": t, "climate_key": c, "n": len(gc), "bias_pct": b, "abs_bias_pct": abs(b),
                          "mape_pct": mape_pct(gc["y"], gc["yhat"]),
                          "r2_bias_removed": L.r2(gc["y"], corrected.loc[gc.index]),
                          "spearman": spearman(gc["y"], gc["yhat"]), "top_hit": top_hit_rate(gc["y"], gc["yhat"])})
        tc = pd.DataFrame([r for r in table if r["target"] == t])
        worst = tc.loc[tc["abs_bias_pct"].idxmax()]
        summary.append({"target": t, "climates": len(tc),
                        "median_abs_bias_pct": float(tc["abs_bias_pct"].median()),
                        "max_abs_bias_pct": float(worst["abs_bias_pct"]), "worst_climate": worst["climate_key"],
                        "mape_pct": mape_pct(g["y"], g["yhat"]),
                        "bias_removed_r2": design_skill(g["y"], corrected, g["climate_key"]),
                        "spearman_median": float(tc["spearman"].median()), "top_hit_mean": float(tc["top_hit"].mean())})
    return pd.DataFrame(summary), pd.DataFrame(table)


def m7_triggered(loco_sum: pd.DataFrame) -> tuple[bool, dict]:
    """Pre-registered trigger (M7): LOCO median |bias| above the limit for heating OR EUI."""
    cfg = CONFIG["m7_trigger"]
    values = loco_sum.set_index("target")["median_abs_bias_pct"]
    missing = [t for t in cfg["targets"] if t not in values.index]
    if missing:
        raise ValueError(f"M7 needs LOCO results for {missing}")
    vals = {t: float(values[t]) for t in cfg["targets"]}
    return any(v > cfg["loco_median_abs_bias_pct"] for v in vals.values()), vals
