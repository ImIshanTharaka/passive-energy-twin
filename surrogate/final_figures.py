"""
final_figures.py - thesis figures for Sub-task 3.7 (reads the saved tables in outputs/final_test only).

Called by `final_test.py --stage figures`. Same style as the 3.4-3.6 figures (climate_gen palette:
dataviz categorical slots 1-2, validated). Identity is never colour-alone: the two holdout parts have
their own marker shapes, reference values are neutral lines, and the report holds every number as a table.
Labels use plain descriptions only (no internal step numbers, gate IDs or feature-set names).

Figures (PNG + PDF):
    fig_parity_primary     EnergyPlus vs surrogate, heating / cooling / EUI; top row: unseen designs in the
                           10 training climates; bottom row: London (unseen climate)
    fig_parity_secondary   the 7 secondary targets, both parts in each panel
    fig_cv_vs_holdout      design skill, cross-validation vs holdout, all 10 targets
    fig_coverage           90 % interval coverage: training climates (standard) and London (new-site)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from surrogate.climate_gen import AXIS, GRID, INK, INK2, MUTED, SERIES
from surrogate.config import ALL_TARGETS, CONFIG, PRIMARY_TARGETS, SECONDARY_TARGETS
from surrogate.explain_figures import _plt, _save, _style

HEATING_TARGETS = tuple(CONFIG["targets"]["heating"])
UNITS = {t: ("W/m²" if t.startswith("peak") else "kWh/m²") for t in ALL_TARGETS}
NAME = {"heating_demand_kwh_m2": "Heating demand", "cooling_demand_kwh_m2": "Cooling demand",
        "eui_kwh_m2": "Energy use intensity", "peak_heating_w_m2": "Peak heating load",
        "peak_cooling_w_m2": "Peak cooling load", "zone_cooling_core_kwh_m2": "Core zone cooling",
        "zone_cooling_south_kwh_m2": "South zone cooling", "zone_cooling_east_kwh_m2": "East zone cooling",
        "zone_cooling_north_kwh_m2": "North zone cooling", "zone_cooling_west_kwh_m2": "West zone cooling"}
PART = {"in_distribution": (SERIES[0], "o", "Unseen designs, 10 training climates"),
        "london": (SERIES[1], "^", "Unseen designs, unseen climate (London)")}


def _part_label(part: str, dev: bool, site: str) -> str:
    if part == "london" and dev:
        return f"DEV: {site} posing as the unseen climate"
    return PART[part][2]


def _scored(g: pd.DataFrame, t: str, heat: list[str]) -> pd.DataFrame:
    return g[g["climate_key"].isin(heat)] if t in HEATING_TARGETS and g["part"].iloc[0] == "in_distribution" else g


def _parity(ax, g: pd.DataFrame, colour, marker, size=16):
    lo = float(min(g["y"].min(), g["yhat"].min()))
    hi = float(max(g["y"].max(), g["yhat"].max()))
    pad = 0.04 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=AXIS, linewidth=1.0, zorder=1)
    ax.scatter(g["y"], g["yhat"], s=size, marker=marker, facecolor=colour, edgecolor="white", linewidth=0.4,
               alpha=0.9, zorder=3)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")


def _note(ax, text):
    ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", ha="left", fontsize=8, color=INK2,
            bbox={"facecolor": "white", "edgecolor": GRID, "boxstyle": "round,pad=0.25", "linewidth": 0.6})


def fig_parity_primary(d: Path, r: dict, dev: bool):
    plt = _plt()
    p = pd.read_csv(d / "predictions.csv.gz")
    p = p[p["variant"] == "primary"]
    hs = pd.read_csv(d / "holdout_summary.csv").query("variant == 'primary'").set_index("target")
    lm = pd.read_csv(d / "london_metrics.csv").query("variant == 'primary'").set_index("target")
    heat = r["heating_climates"]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2))
    for j, t in enumerate(PRIMARY_TARGETS):
        for i, part in enumerate(("in_distribution", "london")):
            ax = axes[i, j]
            _style(ax, "both")
            g = _scored(p[(p["target"] == t) & (p["part"] == part)], t, heat)
            col, mk, _ = PART[part]
            _parity(ax, g, col, mk)
            if part == "in_distribution":
                s = hs.loc[t]
                _note(ax, f"R² {s['r2']:.3f}\ndesign skill {s['design_skill']:.3f}\nRMSE {s['rmse']:.2f}")
            else:
                s = lm.loc[t]
                _note(ax, f"bias {s['bias_pct']:+.1f} %\nMAPE {s['mape_pct']:.1f} %\nSpearman {s['spearman']:.3f}")
            if i == 0:
                ax.set_title(NAME[t])
            ax.set_xlabel(f"EnergyPlus ({UNITS[t]})")
            if j == 0:
                ax.set_ylabel(f"{_part_label(part, dev, r['new_site'])}\n\nSurrogate ({UNITS[t]})", fontsize=8.5)
    fig.suptitle("Final holdout test: surrogate vs EnergyPlus" + (" (DEV - not a result)" if dev else ""),
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    out = _save(fig, d, "fig_parity_primary")
    plt.close(fig)
    return out


def fig_parity_secondary(d: Path, r: dict, dev: bool):
    plt = _plt()
    p = pd.read_csv(d / "predictions.csv.gz")
    p = p[p["variant"] == "primary"]
    heat = r["heating_climates"]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.6))
    axes = axes.ravel()
    for k, t in enumerate(SECONDARY_TARGETS):
        ax = axes[k]
        _style(ax, "both")
        g_all = p[p["target"] == t]
        lo, hi = float(min(g_all["y"].min(), g_all["yhat"].min())), float(max(g_all["y"].max(), g_all["yhat"].max()))
        pad = 0.04 * (hi - lo if hi > lo else 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=AXIS, linewidth=1.0, zorder=1)
        for part in ("in_distribution", "london"):
            col, mk, _ = PART[part]
            g = _scored(g_all[g_all["part"] == part], t, heat)
            ax.scatter(g["y"], g["yhat"], s=13, marker=mk, facecolor=col, edgecolor="white", linewidth=0.4,
                       alpha=0.85, zorder=3, label=_part_label(part, dev, r["new_site"]))
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(NAME[t], fontsize=9.5)
        ax.set_xlabel(f"EnergyPlus ({UNITS[t]})", fontsize=8)
        ax.set_ylabel(f"Surrogate ({UNITS[t]})", fontsize=8)
    axes[-1].axis("off")
    h, lab = axes[0].get_legend_handles_labels()
    axes[-1].legend(h, lab, loc="center", frameon=False, fontsize=8.5, markerscale=1.6)
    fig.suptitle("Final holdout test: secondary outputs" + (" (DEV - not a result)" if dev else ""),
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    out = _save(fig, d, "fig_parity_secondary")
    plt.close(fig)
    return out


def fig_cv_vs_holdout(d: Path, r: dict, dev: bool):
    plt = _plt()
    hs = pd.read_csv(d / "holdout_summary.csv").query("variant == 'primary'").set_index("target")
    cv = pd.read_csv(d / "cv_reference.csv").query("variant == 'primary'").set_index("target")
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    _style(ax, "x")
    y = np.arange(len(ALL_TARGETS))[::-1]
    for yi, t in zip(y, ALL_TARGETS):
        a, b = float(cv.loc[t, "design_skill"]), float(hs.loc[t, "design_skill"])
        ax.plot([a, b], [yi, yi], color=GRID, linewidth=2, zorder=1)
    ax.scatter(cv.loc[ALL_TARGETS, "design_skill"], y, s=60, marker="o", facecolor="white", edgecolor=SERIES[0],
               linewidth=1.6, zorder=3, label="Cross-validation (training designs)")
    ax.scatter(hs.loc[ALL_TARGETS, "design_skill"], y, s=60, marker="D", facecolor=SERIES[0], edgecolor="white",
               linewidth=0.6, zorder=4, label="Holdout (unseen designs, training climates)")
    thr = float(CONFIG["final_test_3_7"]["s3"]["design_skill_min"])
    ax.axvline(thr, color=MUTED, linewidth=1.0, linestyle="--", zorder=0)
    ax.text(thr, y[0] + 0.7, f" minimum for main outputs ({thr:.2f})", fontsize=7.5, color=INK2, va="bottom")
    ax.set_yticks(y, [NAME[t] for t in ALL_TARGETS])
    lo = min(float(cv["design_skill"].min()), float(hs["design_skill"].min()), thr)
    ax.set_xlim(lo - 0.01, 1.002)
    ax.set_ylim(-0.7, len(ALL_TARGETS) + 0.2)
    ax.set_xlabel("Design skill (share of the differences between designs explained, within each climate)")
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.set_title("Accuracy on unseen designs vs cross-validation" + (" (DEV)" if dev else ""), loc="left")
    fig.tight_layout()
    out = _save(fig, d, "fig_cv_vs_holdout")
    plt.close(fig)
    return out


def fig_coverage(d: Path, r: dict, dev: bool):
    plt = _plt()
    ci = pd.read_csv(d / "coverage_in_distribution.csv").set_index("target")
    cl = pd.read_csv(d / "coverage_london.csv").query("interval == 'new_site'").set_index("target")
    band = CONFIG["final_test_3_7"]["s7"]["coverage_band"]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    _style(ax, "x")
    y = np.arange(len(ALL_TARGETS))[::-1]
    ax.axvspan(band[0], band[1], color=GRID, alpha=0.55, zorder=0, linewidth=0)
    ax.axvline(0.90, color=MUTED, linewidth=1.0, linestyle="--", zorder=1)
    ax.scatter(ci.loc[ALL_TARGETS, "coverage_90"], y + 0.12, s=55, marker=PART["in_distribution"][1],
               facecolor=SERIES[0], edgecolor="white", linewidth=0.6, zorder=3,
               label="Unseen designs, training climates (standard interval)")
    ax.scatter(cl.loc[ALL_TARGETS, "coverage_90"], y - 0.12, s=60, marker=PART["london"][1],
               facecolor=SERIES[1], edgecolor="white", linewidth=0.6, zorder=3,
               label=("Unseen climate" if not dev else f"DEV: {r['new_site']}") + " (new-site interval)")
    ax.text(band[0], len(ALL_TARGETS) - 0.35, f" target band {band[0]:.2f}–{band[1]:.2f}", fontsize=7.5, color=INK2)
    ax.set_yticks(y, [NAME[t] for t in ALL_TARGETS])
    lo = min(0.70, float(np.nanmin(np.r_[ci["coverage_90"], cl["coverage_90"]])) - 0.02)
    ax.set_xlim(lo, 1.01)
    ax.set_ylim(-0.7, len(ALL_TARGETS) + 0.1)
    ax.set_xlabel("Share of true EnergyPlus values inside the 90 % prediction interval")
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.set_title("Do the 90 % intervals hold on unseen data?" + (" (DEV)" if dev else ""), loc="left")
    fig.tight_layout()
    out = _save(fig, d, "fig_coverage")
    plt.close(fig)
    return out


def make_all(d: Path, dev: bool, log=print):
    import json
    r = json.loads((d / "final_test_result.json").read_text(encoding="utf-8"))
    for f in (fig_parity_primary, fig_parity_secondary, fig_cv_vs_holdout, fig_coverage):
        out = f(d, r, dev)
        log(f"  {out.name} (+ .pdf)")
