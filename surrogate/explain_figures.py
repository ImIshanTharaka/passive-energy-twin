"""
explain_figures.py - thesis figures for Sub-task 3.6 (reads the saved tables in outputs/explain only).

Called by `explain.py --stage figures`. Same style as the 3.4 / 3.5 figures (climate_gen palette).
Group colours = dataviz categorical slots 1-7 in fixed order (validated light mode: CVD worst adjacent
dE 9.1, normal-vision 19.6; three slots below 3:1 contrast -> every segment >= 8 % carries a printed
label and the report holds the same numbers as tables). Identity is never colour-alone: climates have
their own markers, strong/other deltas their own marker shapes, reference values neutral lines.

Figures (PNG + PDF): fig_oat_parity, fig_oat_signs, fig_sweeps, fig_shap_groups, fig_shap_inputs,
fig_shap_dependence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from surrogate.climate_gen import AXIS, CITY, GRID, INK, INK2, MUTED, SERIES, SURFACE
from surrogate.uncertainty import SHORT_T

GROUP_ORDER = ["climate", "geometry", "opaque_envelope", "glazing", "window_area", "shading", "orientation"]
GROUP_COLOUR = dict(zip(GROUP_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]))
GROUP_LABEL = {"climate": "Climate (whole site)", "geometry": "Geometry", "opaque_envelope": "Walls, roof, floor, air",
               "glazing": "Glass type (U, SHGC)", "window_area": "Window area (WWR)", "shading": "Shading",
               "orientation": "Orientation"}
CLIMATE_STYLE = {"colombo": (SERIES[1], "o", "Colombo"), "minneapolis": (SERIES[0], "s", "Minneapolis")}
PRIMARY = ["heating_demand_kwh_m2", "cooling_demand_kwh_m2", "eui_kwh_m2"]
NICE = {"glazing_shgc": "SHGC", "glazing_u": "Glazing U", "wwr_north": "WWR north", "wwr_east": "WWR east",
        "wwr_south": "WWR south", "wwr_west": "WWR west"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                         "axes.labelcolor": INK2, "text.color": INK})
    return plt


def _style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, d: Path, name: str) -> Path:
    for ext in ("png", "pdf"):
        fig.savefig(d / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    return d / f"{name}.png"


def fig_oat_parity(d: Path):
    plt = _plt()
    t = pd.read_csv(d / "oat_table.csv")
    c = t[t["counted"]].copy()
    c["x"] = 100 * c["d_ep"] / c["ep_base"].abs()
    c["y"] = 100 * c["d_sur"] / c["ep_base"].abs()
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    _style(ax, "both")
    lim = float(np.nanmax(np.abs(np.r_[c["x"], c["y"]]))) * 1.05 if len(c) else 1.0
    ax.plot([-lim, lim], [-lim, lim], color=AXIS, linewidth=1.2, zorder=1)
    ax.axhline(0, color=GRID, linewidth=0.8)
    ax.axvline(0, color=GRID, linewidth=0.8)
    for key, (col, mk, lab) in CLIMATE_STYLE.items():
        for strong, face, size in ((True, col, 46), (False, "white", 30)):
            g = c[(c["climate_key"] == key) & (c["strong"] == strong)]
            ax.scatter(g["x"], g["y"], s=size, marker=mk, facecolor=face, edgecolor=col, linewidth=1.2, zorder=3,
                       label=f"{lab}, {'strong set' if strong else 'other'}")
    wrong = c[~c["sign_ok"]]
    ax.scatter(wrong["x"], wrong["y"], s=120, marker="o", facecolor="none", edgecolor=INK, linewidth=1.0, zorder=4,
               label="Wrong sign")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("EnergyPlus change from base design (% of base)")
    ax.set_ylabel("Surrogate change from base design (% of base)")
    ax.set_title("One-at-a-time test: surrogate vs EnergyPlus (counted changes)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    return _save(fig, d, "fig_oat_parity")


def fig_oat_signs(d: Path):
    plt = _plt()
    t = pd.read_csv(d / "oat_table.csv")
    t = t[t["informative"]]
    t["col"] = t["target"].map(SHORT_T) + " · " + t["climate_key"].map(lambda k: CITY.get(k, k)[:4])
    t["row"] = t["parameter"] + " (" + t["bound"] + ")"
    state = np.where(~t["counted"], 0, np.where(t["sign_ok"], 1, 2))
    t["state"] = state
    m = t.pivot_table(index="row", columns="col", values="state", aggfunc="first")
    m = m.reindex(sorted(m.index))
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#f1f0ec", SERIES[0], SERIES[1]])
    fig, ax = plt.subplots(figsize=(0.42 * m.shape[1] + 2.5, 0.22 * m.shape[0] + 1.6))
    ax.imshow(m.to_numpy(dtype=float), cmap=cmap, vmin=0, vmax=2, aspect="auto", interpolation="nearest")
    for (i, j), v in np.ndenumerate(m.to_numpy(dtype=float)):
        if v == 2:
            ax.text(j, i, "✕", ha="center", va="center", color="white", fontsize=7, fontweight="bold")
    ax.set_xticks(range(m.shape[1]), m.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(m.shape[0]), m.index, fontsize=7)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    import matplotlib.patches as mp
    ax.legend(handles=[mp.Patch(color=SERIES[0], label="Same direction as EnergyPlus"),
                       mp.Patch(color=SERIES[1], label="Opposite direction (✕)"),
                       mp.Patch(color="#f1f0ec", label="Not counted (change ≤ 2 % or heating in Colombo)")],
              frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("One-at-a-time test: direction agreement per parameter, target and city")
    return _save(fig, d, "fig_oat_signs")


def fig_sweeps(d: Path):
    plt = _plt()
    s = pd.read_csv(d / "sweep_summary.csv")
    lab = [f"{NICE.get(r.input, r.input)} → {SHORT_T[r.target]} ({'↑' if r.direction == 'up' else '↓'})"
           for r in s.itertuples()]
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(s) + 1.2))
    _style(ax, "x")
    y = np.arange(len(s))[::-1]
    ax.barh(y, 100 * s["share_reversing"], height=0.55, color=SERIES[0])
    for yi, v in zip(y, s["share_reversing"]):
        ax.text(100 * v + 0.05, yi, f"{100 * v:.1f} %", va="center", fontsize=8, color=INK2)
    ax.axvline(1.0, color=INK2, linewidth=1.0, linestyle="--")
    ax.text(1.0, y.max() + 0.6, " 1 % limit (S6)", fontsize=7.5, color=INK2, va="bottom")
    ax.set_yticks(y, lab)
    ax.set_xlabel("Designs whose prediction moves the wrong way by more than the CV RMSE (%)")
    ax.set_title("Monotonic sweeps: share of designs with a reversal")
    ax.set_xlim(0, max(2.0, 100 * float(s["share_reversing"].max()) * 1.25))
    return _save(fig, d, "fig_sweeps")


def _stacked(ax, shares: pd.DataFrame, labels: list[str]):
    y = np.arange(len(shares))[::-1]
    left = np.zeros(len(shares))
    for g in GROUP_ORDER:
        if g not in shares.columns:
            continue
        v = shares[g].fillna(0).to_numpy()
        ax.barh(y, v * 100, left=left * 100, height=0.6, color=GROUP_COLOUR[g], edgecolor=SURFACE, linewidth=1.0,
                label=GROUP_LABEL[g])
        for yi, l, w in zip(y, left, v):
            if w >= 0.08:
                ax.text(100 * (l + w / 2), yi, f"{100 * w:.0f}", ha="center", va="center", fontsize=7, color=INK)
        left += v
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of mean |SHAP| (%)")


def fig_shap_groups(d: Path):
    plt = _plt()
    sh = pd.read_csv(d / "shap_group_share.csv")
    cmp_ = pd.read_csv(d / "shap_groups_vs_2p8.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1, 1.0], "wspace": 0.55})
    ov = sh[sh["view"] == "overall"].set_index("target").reindex(
        [t for t in SHORT_T if t in set(sh["target"])])
    _style(axes[0], "x")
    _stacked(axes[0], ov, [SHORT_T[t] for t in ov.index])
    axes[0].set_title("All climates: what drives each target")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4)
    rows = []
    for t in [t for t in PRIMARY if t in set(cmp_["target"])]:     # 2.8 has per-climate shares for heating, cooling
        g = cmp_[cmp_["target"] == t]
        for src, col in (("3.6", "share_3_6"), ("2.8", "share_2_8")):
            m = g.pivot_table(index="climate_key", columns="group", values=col).mean()
            rows.append({"label": f"{SHORT_T[t]} · {'final model (3.6)' if src == '3.6' else 'RF (2.8)'}", **m})
    within = pd.DataFrame(rows).set_index("label")
    _style(axes[1], "x")
    _stacked(axes[1], within, list(within.index))
    axes[1].set_title("Within one climate (mean over climates): 3.6 vs 2.8\n(2.8 reported per-climate shares for heating and cooling only)")
    return _save(fig, d, "fig_shap_groups")


def fig_shap_inputs(d: Path):
    plt = _plt()
    imp = pd.read_csv(d / "shap_importance.csv")
    imp = imp[imp["view"] == "overall"].set_index("target")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, t in zip(axes, PRIMARY):
        _style(ax, "x")
        if t not in imp.index:
            continue
        s = imp.loc[t].drop("view").dropna().astype(float).sort_values(ascending=False).head(10)[::-1]
        cols = [GROUP_COLOUR["climate"] if k == "climate" else SERIES[1] for k in s.index]
        ax.barh(range(len(s)), s.to_numpy(), color=cols, height=0.6)
        ax.set_yticks(range(len(s)), [k.replace("_", " ") for k in s.index], fontsize=8)
        ax.set_xlabel("Mean |SHAP| (kWh/m²)")
        ax.set_title(f"{SHORT_T[t]}: top 10 inputs")
    return _save(fig, d, "fig_shap_inputs")


def fig_shap_dependence(d: Path):
    plt = _plt()
    from surrogate import cv
    sv = pd.read_csv(d / "shap_by_climate.csv.gz")
    df, _ = cv.load_training()
    vals = df.set_index("run_id")
    panels = [("glazing_shgc", "cooling_demand_kwh_m2"), ("glazing_u", "heating_demand_kwh_m2"),
              ("wwr_south", "zone_cooling_south_kwh_m2")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax, (inp, t) in zip(axes, panels):
        _style(ax, "both")
        g = sv[sv["target"] == t]
        if g.empty:
            continue
        x = vals.loc[g["run_id"], inp].to_numpy()
        ax.scatter(x, g[inp], s=9, color=SERIES[0], alpha=0.55, linewidth=0)
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.set_xlabel(NICE.get(inp, inp))
        ax.set_ylabel(f"SHAP for {SHORT_T[t]} (kWh/m²)")
        ax.set_title(f"{NICE.get(inp, inp)} → {SHORT_T[t]} (within climate)")
    return _save(fig, d, "fig_shap_dependence")


def make_all(d: Path) -> list[Path]:
    out = []
    for fn, need in ((fig_oat_parity, "oat_table.csv"), (fig_oat_signs, "oat_table.csv"),
                     (fig_sweeps, "sweep_summary.csv"), (fig_shap_groups, "shap_group_share.csv"),
                     (fig_shap_inputs, "shap_importance.csv"), (fig_shap_dependence, "shap_by_climate.csv.gz")):
        if (d / need).exists():
            out.append(fn(d))
    return out
