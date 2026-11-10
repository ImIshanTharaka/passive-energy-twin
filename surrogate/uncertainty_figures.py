"""
uncertainty_figures.py - thesis figures for Sub-task 3.5 (reads the saved tables only).

Called by `uncertainty.py --stage figures`. Same style and validated palette slots 1-3 as the 3.4
figures (climate_gen.SERIES; dataviz palette, CVD-checked). Identity is never colour-alone: every
series also has its own marker and a legend; reference values (90 % target, 0.85-0.95 band, D_max)
are drawn as neutral lines/bands.

Figures (PNG + PDF): fig_coverage_per_climate, fig_width_per_climate, fig_reliability,
fig_gpr_vs_conformal, fig_loco_vs_distance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from surrogate.climate_gen import AXIS, CITY, GRID, INK, INK2, MUTED, SERIES, SURFACE
from surrogate.uncertainty import COVERAGE_BAND, SHORT_T

PRIMARY = ["heating_demand_kwh_m2", "cooling_demand_kwh_m2", "eui_kwh_m2"]
MODE_STYLE = {   # mode: (label, colour, marker, size)
    "standard_grouped": ("Normal range, known city (grouped CV)", SERIES[0], "o", 34),
    "new_site_loco": ("New-site range, hidden city (LOCO)", SERIES[2], "D", 34),
    "standard_in_loco": ("Normal range used for a hidden city", SERIES[1], "s", 30),
}
METHOD_LABEL = {"lgbm": "Conformal, scaled (deployed)", "none": "Conformal, same width",
                "given": "Conformal, GPR-scaled (hybrid)", "native": "GPR's own interval"}
UNIT = {"heating_demand_kwh_m2": "kWh/m²", "cooling_demand_kwh_m2": "kWh/m²", "eui_kwh_m2": "kWh/m²"}


def _style(ax, grid_axis="x"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, d: Path, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(d / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")


def _band(ax, vertical: bool):
    lo, hi = COVERAGE_BAND
    if vertical:
        ax.axvspan(lo, hi, color=GRID, alpha=0.45, linewidth=0, zorder=0)
        ax.axvline(0.90, color=AXIS, linewidth=1.2, zorder=1)
    else:
        ax.axhspan(lo, hi, color=GRID, alpha=0.45, linewidth=0, zorder=0)
        ax.axhline(0.90, color=AXIS, linewidth=1.2, zorder=1)


def _deployed(pc: pd.DataFrame, sel: dict, t: str) -> pd.DataFrame:
    return pc[(pc["target"] == t) & (pc["method"] == sel[t]["deployed_method"]) & pc["scored"]]


def fig_coverage_per_climate(d: Path, pc: pd.DataFrame, sel: dict):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))
    for ax, t in zip(axes, PRIMARY):
        _style(ax)
        _band(ax, vertical=True)
        g = _deployed(pc, sel, t)
        order = (g[g["mode"] == "new_site_loco"].set_index("climate_key")["coverage_90"].sort_values().index.tolist())
        ys = np.arange(len(order))
        for mode, (lab, col, mk, s) in MODE_STYLE.items():
            v = g[g["mode"] == mode].set_index("climate_key").reindex(order)["coverage_90"]
            ax.scatter(v, ys, s=s, color=col, marker=mk, edgecolor="white", linewidth=1.2, zorder=4, label=lab)
        ax.set_yticks(ys, [CITY[c] for c in order])
        ax.set_xlim(-0.02, 1.04)
        ax.set_xlabel("Share of designs inside the 90 % range", color=INK2, fontsize=8.5)
        ax.set_title(SHORT_T[t], color=INK, fontsize=11, loc="left")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.01, 1.07), ncol=3)
    fig.text(0.01, -0.03, "Grey band: 0.85–0.95 (gate S7); line: 0.90 target. Heating scored on the 7 heating "
             "climates.", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    _save(fig, d, "fig_coverage_per_climate")
    plt.close(fig)


def fig_width_per_climate(d: Path, pc: pd.DataFrame, sel: dict):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))
    show = ["standard_grouped", "new_site_loco"]
    for ax, t in zip(axes, PRIMARY):
        _style(ax)
        g = _deployed(pc, sel, t)
        order = g[g["mode"] == "standard_grouped"].set_index("climate_key")["mean_width_90"].sort_values().index.tolist()
        h = 0.38
        for j, mode in enumerate(show):
            lab, col, _, _ = MODE_STYLE[mode]
            v = g[g["mode"] == mode].set_index("climate_key").reindex(order)["mean_width_90"].to_numpy()
            ys = np.arange(len(order)) + 0.2 - j * 0.4      # first legend entry drawn on top
            ax.barh(ys, v, height=h, color=col, edgecolor="white", linewidth=1.0, label=lab,
                    hatch=None if j == 0 else "///")
        ax.set_yticks(np.arange(len(order)), [CITY[c] for c in order])
        ax.set_xlabel(f"Mean width of the 90 % range ({UNIT[t]})", color=INK2, fontsize=8.5)
        ax.set_title(SHORT_T[t], color=INK, fontsize=11, loc="left")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.01, 1.07), ncol=2)
    fig.tight_layout()
    _save(fig, d, "fig_width_per_climate")
    plt.close(fig)


def fig_reliability(d: Path, rel: pd.DataFrame, sel: dict):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, t in zip(axes, PRIMARY):
        _style(ax, "both")
        ax.plot([0.4, 1.0], [0.4, 1.0], color=AXIS, linewidth=1.2, zorder=1)
        g = rel[(rel["target"] == t) & (rel["method"] == sel[t]["deployed_method"])]
        for mode, (lab, col, mk, s) in MODE_STYLE.items():
            v = g[g["mode"] == mode].sort_values("level")
            ax.plot(v["level"], v["coverage"], color=col, linewidth=2, zorder=3)
            ax.scatter(v["level"], v["coverage"], color=col, marker=mk, s=s, edgecolor="white", linewidth=1.2,
                       zorder=4, label=lab)
        ax.set_xlim(0.45, 0.98)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Promised level", color=INK2, fontsize=8.5)
        ax.set_title(SHORT_T[t], color=INK, fontsize=11, loc="left")
    axes[0].set_ylabel("Share actually inside", color=INK2, fontsize=8.5)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.01, 1.08), ncol=3)
    fig.text(0.01, -0.03, "Diagonal = the range keeps its promise at every level (50, 80, 90, 95 %).",
             fontsize=7.5, color=MUTED)
    fig.tight_layout()
    _save(fig, d, "fig_reliability")
    plt.close(fig)


def fig_gpr_vs_conformal(d: Path, ov: pd.DataFrame):
    import matplotlib.pyplot as plt
    t = "heating_demand_kwh_m2"
    g = ov[ov["target"] == t]
    methods = [m for m in ("lgbm", "none", "given", "native") if m in set(g["method"])]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    for ax, mode in zip(axes, ["standard_grouped", "new_site_loco"]):
        _style(ax)
        _band(ax, vertical=True)
        lab, col, mk, _ = MODE_STYLE[mode]
        for i, m in enumerate(methods):
            r = g[(g["mode"] == mode) & (g["method"] == m)]
            if r.empty:
                continue
            cov, w = float(r["coverage_90"].iloc[0]), float(r["mean_width_90"].iloc[0])
            ax.barh(i, cov, height=0.6, color=col, edgecolor="white", linewidth=1.0,
                    hatch="///" if m == "native" else None)
            ax.text(min(cov, 1.0) + 0.01, i, f"{cov:.2f}  (width {w:.2f})", va="center", fontsize=7.5, color=INK2)
        ax.set_yticks(range(len(methods)), [METHOD_LABEL[m] for m in methods])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.35)
        ax.set_xlabel("Share inside the 90 % range", color=INK2, fontsize=8.5)
        ax.set_title(lab, color=INK, fontsize=10, loc="left")
    fig.text(0.01, -0.04, "Heating (GPR pick), 7 heating climates. Width in kWh/m². Hatched: no calibration "
             "(GPR's own predictive SD).", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    _save(fig, d, "fig_gpr_vs_conformal")
    plt.close(fig)


def fig_loco_vs_distance(d: Path, ev: pd.DataFrame, d_max: float):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4), sharey=True)
    for ax, t in zip(axes, PRIMARY):
        _style(ax, "both")
        _band(ax, vertical=False)
        ax.axvline(d_max, color=INK2, linewidth=1, linestyle="--", zorder=1)
        g = ev[ev["target"] == t]
        inside = g["n_out_of_range"] == 0
        ax.scatter(g.loc[inside, "distance_to_nearest"], g.loc[inside, "coverage_90_new_site"], s=40,
                   color=SERIES[2], marker="D", edgecolor="white", linewidth=1.2, zorder=4,
                   label="inside the training range")
        ax.scatter(g.loc[~inside, "distance_to_nearest"], g.loc[~inside, "coverage_90_new_site"], s=40,
                   facecolor="white", edgecolor=SERIES[2], linewidth=1.6, marker="D", zorder=4,
                   label="outside the range (≥ 1 climate input)")
        for _, r in g.iterrows():            # selective labels: only cities below the band, and the furthest
            if r["coverage_90_new_site"] >= COVERAGE_BAND[0] and r["distance_to_nearest"] < 0.95 * d_max:
                continue
            ax.annotate(CITY[r["climate_key"]], (r["distance_to_nearest"], r["coverage_90_new_site"]),
                        textcoords="offset points", xytext=(5, -3), fontsize=7, color=INK2)
        ax.set_xlabel("Distance to the nearest other training city", color=INK2, fontsize=8.5)
        ax.set_title(SHORT_T[t], color=INK, fontsize=11, loc="left")
        ax.set_ylim(-0.03, 1.05)
    axes[0].set_ylabel("New-site range: share inside (hidden city)", color=INK2, fontsize=8.5)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.01, 1.08), ncol=2)
    fig.text(0.01, -0.03, f"Dashed line: D_max = {d_max:.2f} (extrapolation limit). Range check against the "
             "other 9 cities. Labelled: cities below 0.85 and the furthest city; all values in domain_evidence.csv.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout()
    _save(fig, d, "fig_loco_vs_distance")
    plt.close(fig)


def make_all(d: Path, dev: bool) -> list[str]:
    pc = pd.read_csv(d / "coverage_per_climate.csv")
    ov = pd.read_csv(d / "coverage_table.csv")
    rel = pd.read_csv(d / "reliability.csv")
    sel = json.loads((d / "selection.json").read_text(encoding="utf-8"))["selection"]
    made = []
    if all(t in sel for t in PRIMARY):
        fig_coverage_per_climate(d, pc, sel)
        fig_width_per_climate(d, pc, sel)
        fig_reliability(d, rel, sel)
        made += ["fig_coverage_per_climate", "fig_width_per_climate", "fig_reliability"]
    if "heating_demand_kwh_m2" in sel:
        fig_gpr_vs_conformal(d, ov)
        made.append("fig_gpr_vs_conformal")
    if (d / "domain_evidence.csv").exists():
        res = json.loads((d / "uncertainty_result.json").read_text(encoding="utf-8"))
        fig_loco_vs_distance(d, pd.read_csv(d / "domain_evidence.csv"), res["stages"]["domain"]["d_max"])
        made.append("fig_loco_vs_distance")
    print(f"  figures: {', '.join(made)}")
    return made
