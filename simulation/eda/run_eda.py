#!/usr/bin/env python3
"""
run_eda.py - Sub-task 2.8 EDA: figures, metrics, report and the Phase 3 go/no-go gate.

Reads the frozen 2.7 dataset through eda_lib.load_split() (sha256-verified, E7), runs
plan sections A-E, applies the gate (plan §5) and the pre-registered Option C decision
rule (§5.1), and writes everything to outputs/eda/. Never writes to outputs/dataset/.

    python eda/run_eda.py            # official run (RF 300 trees, SHAP 1,000 rows)
    python eda/run_eda.py --fast     # development only: 50 trees, small SHAP sample
    python eda/run_eda.py --out DIR  # write somewhere else

The holdout is only described (E1): its inputs and London targets appear in F01/F03/F04;
no model is fitted or scored on it.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt         # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from eda import eda_lib as L            # noqa: E402

OUT_DEFAULT = ROOT / "outputs" / "eda"

# ── Style (dataviz reference palette, light mode: static thesis figures) ──────
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CLIMATE_GROUP = {"chicago": "cold", "denver": "cold", "minneapolis": "cold", "new_york": "cold",
                 "atlanta": "mild", "san_francisco": "mild",
                 "chennai": "hot", "colombo": "hot", "miami": "hot", "phoenix": "hot", "london": "holdout"}
GROUP_COLOR = {"cold": SERIES[0], "hot": SERIES[1], "mild": SERIES[2]}       # first 3 slots: all-pairs safe
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_bluered", ["#1c5cab", "#86b6ef", "#f0efec", "#f0a3a2", "#b52f2f"])
LABEL = {"heating_demand_kwh_m2": "Heating demand", "cooling_demand_kwh_m2": "Cooling demand",
         "eui_kwh_m2": "EUI", "peak_heating_w_m2": "Peak heating", "peak_cooling_w_m2": "Peak cooling",
         "zone_cooling_core_kwh_m2": "Core cooling", "zone_cooling_south_kwh_m2": "South-zone cooling",
         "zone_cooling_east_kwh_m2": "East-zone cooling", "zone_cooling_north_kwh_m2": "North-zone cooling",
         "zone_cooling_west_kwh_m2": "West-zone cooling"}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
        "pdf.fonttype": 42, "savefig.dpi": 200, "savefig.bbox": "tight"})


def save(fig, figdir: Path, name: str, written: list):
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"{name}.{ext}")
    plt.close(fig)
    written.append(name)


def climate_order(df: pd.DataFrame) -> list[str]:
    """Climates ordered cold -> hot by HDD18 (descending)."""
    return list(df.groupby("climate_key")["hdd18"].first().sort_values(ascending=False).index.astype(str))


# ═════════════════════════════════════════════════════════════════════════════
# Figures
# ═════════════════════════════════════════════════════════════════════════════

def f01_inputs(df, hold, cols, bounds, figdir, w):
    ins = cols["design_inputs"]
    fig, axes = plt.subplots(4, 6, figsize=(13, 8))
    for ax, c in zip(axes.flat, ins):
        lo, hi = bounds[c]["min"], bounds[c]["max"]
        bins = np.arange(lo - 0.5, hi + 1.5) if bounds[c]["integer"] else np.linspace(lo, hi, 21)
        ax.hist(df[c], bins=bins, density=True, color=SERIES[0], alpha=0.85, rwidth=0.92, label="training (3,000)")
        ax.hist(hold[c], bins=bins, density=True, histtype="step", color=SERIES[1], linewidth=1.5, label="holdout (400)")
        ax.set_title(c, fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", visible=False)
    for ax in list(axes.flat)[len(ins):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("F01  Design-input coverage: training vs holdout (density)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, figdir, "F01_input_histograms", w)


def f02_correlation(df, cols, figdir, w):
    ins = cols["design_inputs"]
    mats = [g[ins].rank().corr().abs().to_numpy() for _, g in df.groupby("climate_key")]
    m = np.max(mats, axis=0)
    np.fill_diagonal(m, np.nan)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(m, cmap=SEQ, vmin=0, vmax=L.INPUT_CORR_MAX)
    ax.set_xticks(range(len(ins)), ins, rotation=90, fontsize=7)
    ax.set_yticks(range(len(ins)), ins, fontsize=7)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(f"max |Spearman ρ| over climates (scale ends at the C2 limit {L.INPUT_CORR_MAX})")
    ax.set_title(f"F02  Design-input correlation, worst climate per pair (max {np.nanmax(m):.3f})", loc="left")
    save(fig, figdir, "F02_input_correlation", w)


def f03_climate(df, hold, figdir, w):
    clim = pd.concat([df, hold]).groupby("climate_key")[["hdd18", "cdd10", "annual_ghi", "latitude"]].first()
    fig, ax = plt.subplots(figsize=(7, 5))
    for k, r in clim.iterrows():
        k = str(k)
        hold_pt = k == "london"
        ax.scatter(r["hdd18"], r["cdd10"], s=60, color=SERIES[1] if hold_pt else SERIES[0],
                   edgecolor=SURFACE, linewidth=2, zorder=3, marker="D" if hold_pt else "o")
        ax.annotate(k.replace("_", " ") + (" (holdout)" if hold_pt else ""), (r["hdd18"], r["cdd10"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xlabel("HDD18 (°C·day)")
    ax.set_ylabel("CDD10 (°C·day)")
    ax.set_title("F03  Climate space: 10 training climates and the London holdout", loc="left")
    save(fig, figdir, "F03_climate_space", w)


def f04_targets(df, hold, figdir, w):
    order = climate_order(df)
    london = hold[hold["climate_key"] == "london"]
    targets = ["heating_demand_kwh_m2", "cooling_demand_kwh_m2", "eui_kwh_m2", "peak_heating_w_m2", "peak_cooling_w_m2"]
    fig, axes = plt.subplots(len(targets), 1, figsize=(11, 13), sharex=True)
    labels = order + ["london"]
    for ax, t in zip(axes, targets):
        data = [df.loc[df["climate_key"] == c, t].to_numpy() for c in order] + [london[t].to_numpy()]
        parts = ax.violinplot(data, showmedians=True, showextrema=False, widths=0.8)
        for i, b in enumerate(parts["bodies"]):
            g = CLIMATE_GROUP.get(labels[i], "mild")
            b.set_facecolor(SERIES[1] if g == "holdout" else GROUP_COLOR[g])
            b.set_edgecolor(SURFACE)
            b.set_alpha(0.85)
            if g == "holdout":
                b.set_hatch("///")
        parts["cmedians"].set_color(INK)
        parts["cmedians"].set_linewidth(1.2)
        ax.set_ylabel(f"{LABEL[t]}\n({'W/m²' if t.startswith('peak') else 'kWh/m²·yr'})")
        ax.grid(axis="x", visible=False)
    axes[-1].set_xticks(range(1, len(labels) + 1), [c.replace("_", "\n") for c in labels], fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g]) for g in ("cold", "mild", "hot")]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=SERIES[1], hatch="///", edgecolor=SURFACE))
    axes[0].legend(handles, ["cold", "mild", "hot", "London holdout (anchors)"], ncol=4, loc="upper right", fontsize=8)
    fig.suptitle("F04  Targets by climate (ordered by HDD18, cold → hot); line = median", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, figdir, "F04_targets_by_climate", w)


def f05_transforms(df, figdir, w):
    ts = ["heating_demand_kwh_m2", "peak_cooling_w_m2", "zone_cooling_west_kwh_m2"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5))
    for j, t in enumerate(ts):
        for i, (lab, x) in enumerate((("raw", df[t]), ("log1p", np.log1p(df[t])))):
            ax = axes[i, j]
            ax.hist(x, bins=40, color=SERIES[0], rwidth=0.92)
            ax.set_title(f"{LABEL[t]} - {lab} (skew {pd.Series(x).skew():.2f})", fontsize=9, loc="left")
            ax.grid(axis="x", visible=False)
    fig.suptitle("F05  Target skew: raw vs log1p (input to the Phase 3 transform decision)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, figdir, "F05_target_transforms", w)


def f06_target_corr(df, cols, figdir, w):
    t = cols["targets"]
    m = df[t].rank().corr().to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(m, cmap=DIV, vmin=-1, vmax=1)
    for i in range(len(t)):
        for j in range(len(t)):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color=SURFACE if abs(m[i, j]) > 0.7 else INK)
    names = [LABEL[x] for x in t]
    ax.set_xticks(range(len(t)), names, rotation=60, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(t)), names, fontsize=7.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("Spearman ρ (pooled)")
    ax.set_title("F06  Target–target correlation", loc="left")
    save(fig, figdir, "F06_target_correlation", w)


def f07_eui(df, diag, figdir, w):
    """EUI = lighting/equipment + SYSTEM heating + SYSTEM cooling (ideal loads incl. outdoor air).

    Panel b: net outdoor-air effect = (system - zone) heating + (system - zone) cooling.
    Positive: OA must be conditioned (e.g. humid tropics). Negative: OA gives free cooling (e.g. SF).
    """
    d = df[["run_id", "climate_key", "heating_demand_kwh_m2", "cooling_demand_kwh_m2", "eui_kwh_m2"]].merge(
        diag[["run_id", "elec_kwh_m2", "sys_heat_kwh_m2", "sys_cool_kwh_m2"]], on="run_id")
    d["oa_net"] = (d["sys_heat_kwh_m2"] - d["heating_demand_kwh_m2"]) + (d["sys_cool_kwh_m2"] - d["cooling_demand_kwh_m2"])
    d["closure"] = d["eui_kwh_m2"] - d["elec_kwh_m2"] - d["sys_heat_kwh_m2"] - d["sys_cool_kwh_m2"]
    order = climate_order(df)
    cols_ = ["elec_kwh_m2", "sys_heat_kwh_m2", "sys_cool_kwh_m2", "heating_demand_kwh_m2", "cooling_demand_kwh_m2",
             "oa_net", "eui_kwh_m2", "closure"]
    med = d.groupby("climate_key")[cols_].mean().loc[order]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})
    parts = [("elec_kwh_m2", "lighting + equipment"), ("sys_heat_kwh_m2", "system heating (incl. outdoor air)"),
             ("sys_cool_kwh_m2", "system cooling (incl. outdoor air)")]
    x = np.arange(len(order))
    bottom = np.zeros(len(order))
    for k, (col, lab) in enumerate(parts):
        ax.bar(x, med[col], bottom=bottom, color=SERIES[k], edgecolor=SURFACE, linewidth=2, width=0.7, label=lab)
        bottom += med[col].to_numpy()
    for i, v in enumerate(med["eui_kwh_m2"]):
        ax.text(i, v + 3, f"{v:.0f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, [c.replace("_", "\n") for c in order], fontsize=8)
    ax.set_ylabel("mean kWh/m²·yr")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("a) EUI components (sum = EUI)", loc="left", fontsize=9.5)
    v = med["oa_net"].to_numpy()
    bx.axhline(0, color=AXIS, linewidth=1)
    bx.bar(x, v, color=[DIV(0.85) if t > 0 else DIV(0.15) for t in v], width=0.7)
    for i, t in enumerate(v):
        bx.text(i, t + (3 if t >= 0 else -3), f"{t:+.0f}", ha="center", va="bottom" if t >= 0 else "top",
                fontsize=8, color=INK2)
    bx.set_xticks(x, [c.replace("_", " ") for c in order], fontsize=8, rotation=40, ha="right")
    bx.set_ylabel("system − zone demand (kWh/m²·yr)")
    bx.grid(axis="x", visible=False)
    bx.set_title("b) Net outdoor-air effect: + conditioning load, − free cooling", loc="left", fontsize=9.5)
    fig.suptitle("F07  EUI decomposition by climate (ideal loads, site energy)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, figdir, "F07_eui_decomposition", w)
    return med.round(2)


def f08_physics(phys, order, figdir, w):
    rules = list(dict.fromkeys(phys["rule"]))
    m = phys.pivot(index="rule", columns="climate_key", values="rho").reindex(index=rules, columns=order)
    verdict = phys.pivot(index="rule", columns="climate_key", values="verdict").reindex(index=rules, columns=order)
    exp = phys.groupby("rule")["expected"].first().reindex(rules)
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(m.to_numpy(float), cmap=DIV, vmin=-1, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iat[i, j]
            if pd.isna(v):
                continue
            mark = {"confirmed": "", "weak": "°", "opposite": "✕"}[verdict.iat[i, j]]
            ax.text(j, i, f"{v:+.2f}{mark}", ha="center", va="center", fontsize=7,
                    color=SURFACE if abs(v) > 0.6 else INK)
    ax.set_yticks(range(len(rules)), [f"{r}  (expect {'+' if exp[r] > 0 else '−'})" for r in rules], fontsize=8)
    ax.set_xticks(range(len(order)), [c.replace("_", "\n") for c in order], fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("Spearman ρ within climate")
    ax.set_title(f"F08  Physics directions (° = not significant at p < {L.PHYSICS_ALPHA}; ✕ = opposite sign; "
                 "blank = rule not applied)", loc="left", fontsize=9)
    save(fig, figdir, "F08_physics_directions", w)


def f09_facades(fac_table, order, figdir, w):
    means = fac_table.groupby(["climate_key", "bin"])["residual"].mean().unstack().reindex(columns=L.AZIMUTH_BINS)
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.2), sharey=True)
    x = np.arange(len(L.AZIMUTH_BINS))
    for ax, c in zip(axes.flat, order):
        g = CLIMATE_GROUP[c]
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.bar(x, means.loc[c], color=GROUP_COLOR[g], width=0.75, edgecolor=SURFACE, linewidth=2)
        ax.set_xticks(x, L.AZIMUTH_BINS, fontsize=7)
        ax.set_title(c.replace("_", " "), fontsize=9, loc="left")
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("zone cooling residual\n(kWh/m²·yr)")
    fig.suptitle("F09  Perimeter-zone cooling by the direction the facade truly faces "
                 "(after removing WWR, WWR×SHGC and overhang)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, figdir, "F09_facade_azimuth", w)


def f10_core(df, order, figdir, w):
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.2), sharey=True)
    for ax, c in zip(axes.flat, order):
        g = df[df["climate_key"] == c]
        data = [g.loc[g["n_floors"] == k, "zone_cooling_core_kwh_m2"] for k in range(1, 6)]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False)
        for b in bp["boxes"]:
            b.set_facecolor(GROUP_COLOR[CLIMATE_GROUP[c]])
            b.set_edgecolor(SURFACE)
        for k in ("medians", "whiskers", "caps"):
            for line in bp[k]:
                line.set_color(INK2 if k != "medians" else INK)
                line.set_linewidth(1)
        ax.set_xticks(range(1, 6), [str(k) for k in range(1, 6)])
        ax.set_title(c.replace("_", " "), fontsize=9, loc="left")
        ax.grid(axis="x", visible=False)
    for ax in axes[1, :]:
        ax.set_xlabel("n_floors")
    for ax in axes[:, 0]:
        ax.set_ylabel("core cooling (kWh/m²·yr)")
    fig.suptitle("F10  Core-zone cooling vs number of floors (single storey: core under the roof)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, figdir, "F10_core_nfloors", w)


def f11_parity(oof, summ, figdir, w):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    s = summ.set_index("target")
    for ax, t in zip(axes, L.PRIMARY_TARGETS):
        g = oof[oof["target"] == t]
        for grp in ("cold", "mild", "hot"):
            h = g[g["climate_key"].map(CLIMATE_GROUP) == grp]
            ax.scatter(h["y"], h["yhat"], s=8, alpha=0.55, color=GROUP_COLOR[grp], label=grp, linewidths=0)
        lo, hi = float(g[["y", "yhat"]].min().min()), float(g[["y", "yhat"]].max().max())
        ax.plot([lo, hi], [lo, hi], color=INK2, linewidth=1)
        r = s.loc[t]
        ax.set_title(f"{LABEL[t]}: R² {r['r2']:.3f} · design skill {r['design_skill']:.3f}", fontsize=9, loc="left")
        ax.set_xlabel("simulated (kWh/m²·yr)")
        ax.set_ylabel("RF out-of-fold prediction")
    axes[0].legend(fontsize=8, markerscale=2, loc="upper left")
    fig.suptitle("F11  Baseline RF, grouped 5-fold CV on design_id (out-of-fold predictions)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, figdir, "F11_rf_parity", w)


def f12_shap_features(shap_imp, figdir, w):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, t in zip(axes, L.PRIMARY_TARGETS):
        imp = shap_imp[t].head(15)[::-1]
        ax.barh(range(len(imp)), imp.to_numpy(), color=SERIES[0], height=0.7)
        ax.set_yticks(range(len(imp)), imp.index, fontsize=8)
        ax.set_xlabel("mean |SHAP| (target units)")
        ax.set_title(LABEL[t], fontsize=10, loc="left")
        ax.grid(axis="y", visible=False)
    fig.suptitle("F12  Feature importance (mean |SHAP|, top 15). Individual climate features are not "
                 "separately identifiable (F2): read climate as a group (F13)", x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout()
    save(fig, figdir, "F12_shap_features", w)


def f13_shap_groups(shares, figdir, w):
    groups = list(L.FEATURE_GROUPS)
    targets = list(shares.index)
    fig, ax = plt.subplots(figsize=(11, 5))
    left = np.zeros(len(targets))
    for k, g in enumerate(groups):
        v = shares[g].to_numpy()
        ax.barh(range(len(targets)), v, left=left, color=SERIES[k], edgecolor=SURFACE, linewidth=2, height=0.7, label=g)
        for i, (l0, vi) in enumerate(zip(left, v)):
            if vi >= 0.08:
                ax.text(l0 + vi / 2, i, f"{vi:.0%}", ha="center", va="center", fontsize=7, color=SURFACE)
        left += v
    ax.set_yticks(range(len(targets)), [LABEL[t] for t in targets])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=7, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title("F13  Share of mean |SHAP| by input group, per target", loc="left", pad=28)
    save(fig, figdir, "F13_shap_groups", w)


def f14_shap_climate(per_clim, order, figdir, w):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True, gridspec_kw={"wspace": 0.08})
    groups = [g for g in L.FEATURE_GROUPS if g != "climate"]
    for ax, t in zip(axes, ("cooling_demand_kwh_m2", "heating_demand_kwh_m2")):
        m = per_clim[t].reindex(index=order, columns=groups)
        im = ax.imshow(m.to_numpy(float), cmap=SEQ, vmin=0, vmax=max(0.5, float(np.nanmax(m.to_numpy()))), aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.iat[i, j]
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7, color=SURFACE if v > 0.3 else INK)
        ax.set_xticks(range(len(groups)), groups, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(order)), [c.replace("_", " ") for c in order], fontsize=8)
        ax.grid(False)
        ax.set_title(f"{LABEL[t]}: share of design-driven |SHAP|", fontsize=9.5, loc="left")
    fig.colorbar(im, ax=axes, shrink=0.8).set_label("share (design groups only)")
    fig.suptitle("F14  Which passive lever matters where", x=0.01, ha="left")
    save(fig, figdir, "F14_shap_by_climate", w)


def f15_learning(lc, figdir, w):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for k, t in enumerate(L.PRIMARY_TARGETS):
        g = lc[lc["target"] == t].sort_values("fraction")
        ax.plot(g["n_designs"], g["design_skill"], color=SERIES[k], marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=LABEL[t])
        ax.annotate(f"{LABEL[t]} {g['design_skill'].iloc[-1]:.3f}", (g["n_designs"].iloc[-1], g["design_skill"].iloc[-1]),
                    xytext=(8, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax.axhline(L.DESIGN_SKILL_MIN, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(ax.get_xlim()[0], L.DESIGN_SKILL_MIN, f" gate {L.DESIGN_SKILL_MIN}", va="bottom", fontsize=8, color=INK2)
    ax.set_xlabel("training designs (grouped 5-fold CV)")
    ax.set_ylabel("design skill")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("F15  Learning curve: does skill still rise with more designs?", loc="left")
    save(fig, figdir, "F15_learning_curve", w)


# ═════════════════════════════════════════════════════════════════════════════
# Report
# ═════════════════════════════════════════════════════════════════════════════

def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if pd.isna(v) else floatfmt.format(v)
        if isinstance(v, (list, tuple)):
            return ", ".join(map(str, v))
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def write_report(path: Path, M: dict, T: dict, figs: list, fast: bool):
    d = M["decision"]
    lines = [
        "# Phase 2.8 - EDA report (Phase 3 go/no-go)", "",
        f"Generated {M['created_utc']} by `eda/run_eda.py`{' **--fast (development only, not the official gate)**' if fast else ''}.",
        f"Dataset: `dataset_training.csv` sha256 `{M['dataset_sha256']['training'][:12]}…`, "
        f"`dataset_holdout.csv` `{M['dataset_sha256']['holdout'][:12]}…`. RF: {M['rf_params']}.", "",
        "## Gate and decision", "",
        f"- **Gate: {d['gate']}**",
        f"- **Decision: {d['decision']}**: {d['reason']}",
        f"- Rule: {d['rule']}. Failed checks: {', '.join(d['failed_checks']) or 'none'}", "",
        "## Checks", "", md_table(T["checks"]), "",
        "## Signal: grouped 5-fold CV on design_id (E2-E4)", "",
        "`design_skill` = 1 − SSE(model)/SSE(climate-mean predictor). Heating targets: skill over the heating "
        f"climates only ({', '.join(M['heating_climates'])}).", "",
        md_table(T["summary"]), "",
        "### Within-climate R² (primary targets)", "", md_table(T["within"]), "",
        "### Random 80/20 split vs grouped CV (anchor leakage gap)", "", md_table(T["leak"]), "",
        "### Learning curve (diagnostic for the decision rule)", "", md_table(T["learning"]), "",
        "## Physics directions (Spearman ρ within climate)", "", md_table(T["physics"]), "",
        "### Roof insulation and core cooling, by storey count (P8, hot climates)", "", md_table(T["core"]), "",
        "## Facade azimuth (O1)", "", md_table(T["facade"]), "",
        "## SHAP: share of mean |SHAP| by group", "", md_table(T["shares"], "{:.1%}"), "",
        "## EUI decomposition (mean kWh/m²·yr)", "",
        "EUI = elec + system heating + system cooling (`closure` ≈ 0). `oa_net` = (system − zone) heating + "
        "(system − zone) cooling: positive where outdoor air must be conditioned, negative where it gives free cooling.", "",
        md_table(T["eui"].reset_index(), "{:.1f}"), "",
        "## Largest out-of-fold residuals (R4, top 5 per primary target; all 20 in top_residuals.csv)", "",
        md_table(T["resid"]), "",
        "Each must be physically explainable. Review and record the explanation in the 2.8 plan (Rev 2).", "",
        "## Figures", "", *[f"- `figures/{f}.png` / `.pdf`" for f in figs], "",
        "## Environment", "", "```", json.dumps(M["environment"], indent=2), "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def environment() -> dict:
    import scipy, sklearn, shap   # noqa: E401
    return {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
            "shap": shap.__version__, "matplotlib": matplotlib.__version__}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sub-task 2.8 EDA and Phase 3 gate")
    ap.add_argument("--fast", action="store_true", help="development only: 50 trees, small SHAP sample")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    a = ap.parse_args(argv)
    t0 = time.time()
    rf = L.RF_PARAMS_FAST if a.fast else L.RF_PARAMS
    shap_n_primary = L.SHAP_PER_CLIMATE_FAST if a.fast else L.SHAP_PER_CLIMATE
    shap_n_secondary = 10 if a.fast else 30
    out, figdir = a.out, a.out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    style()

    def step(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    step("load (sha256-verified)")
    df, cols, sha_tr = L.load_split("training")
    hold, _, sha_ho = L.load_split("holdout")
    diag, _, _ = L.load_split("diagnostics")
    heat = L.heating_climates(df)
    order = climate_order(df)
    bounds = L.design_bounds()
    checks = [L.check_rows(df, hold)]

    step("A coverage")
    cov = L.coverage_uniformity(df, cols["design_inputs"], bounds)
    cor = L.input_correlation(df, cols["design_inputs"])
    checks += [L.check_coverage(cov), L.check_correlation(cor)]

    step("C physics, D facades")
    phys = L.evaluate_physics(df, heat=heat)
    pstat = L.physics_rule_status(phys)
    checks += L.physics_checks(pstat)
    ftab = L.facade_table(df)
    fac = L.facade_azimuth_check(ftab)
    checks.append(L.check_facades(fac))
    core = L.core_roof_interaction(df)

    step(f"E grouped CV ({rf['n_estimators']} trees x {L.N_FOLDS} folds x {len(cols['targets'])} targets)")
    oof = L.cv_grouped(df, cols["inputs"], cols["targets"], params=rf)
    summ, per = L.evaluate_predictions(oof, heat)
    checks += L.signal_checks(summ)
    step("E random 80/20 split (leakage gap)")
    oof_r = L.cv_random(df, cols["inputs"], cols["targets"], params=rf)
    summ_r, _ = L.evaluate_predictions(oof_r, heat)
    step("E learning curve (primary targets)")
    lc = L.learning_curve_skill(df, cols["inputs"], L.PRIMARY_TARGETS, heat, params=rf)
    decision = L.decide(checks, lc)

    step("E SHAP")
    shap_imp, shares, per_clim = {}, {}, {}
    for t in cols["targets"]:
        n = shap_n_primary if t in L.PRIMARY_TARGETS else shap_n_secondary
        sample = L.stratified_sample(df, n)
        model = L.fit_full(df, cols["inputs"], t, params=rf)
        sv = L.shap_values(model, sample[cols["inputs"]])
        gs = L.shap_grouped(sv)
        shap_imp[t] = L.shap_importance(sv)
        shares[t] = L.shap_share(gs)
        if t in ("cooling_demand_kwh_m2", "heating_demand_kwh_m2"):
            design_groups = [g for g in L.FEATURE_GROUPS if g != "climate"]
            a_ = gs[design_groups].abs().groupby(sample["climate_key"].astype(str).to_numpy()).mean()
            per_clim[t] = a_.div(a_.sum(axis=1), axis=0)
        step(f"   SHAP {t} ({len(sample)} rows)")
    shares_df = pd.DataFrame(shares).T.reindex(columns=list(L.FEATURE_GROUPS))

    step("figures")
    figs: list[str] = []
    f01_inputs(df, hold, cols, bounds, figdir, figs)
    f02_correlation(df, cols, figdir, figs)
    f03_climate(df, hold, figdir, figs)
    f04_targets(df, hold, figdir, figs)
    f05_transforms(df, figdir, figs)
    f06_target_corr(df, cols, figdir, figs)
    eui = f07_eui(df, diag, figdir, figs)
    f08_physics(phys, order, figdir, figs)
    f09_facades(ftab, order, figdir, figs)
    f10_core(df, order, figdir, figs)
    f11_parity(oof, summ, figdir, figs)
    f12_shap_features(shap_imp, figdir, figs)
    f13_shap_groups(shares_df, figdir, figs)
    f14_shap_climate(per_clim, order, figdir, figs)
    f15_learning(lc, figdir, figs)

    step("write outputs")
    resid = L.top_residuals(oof, diag, L.PRIMARY_TARGETS, n=20)
    resid.to_csv(out / "top_residuals.csv", index=False)
    pd.concat([oof, oof_r], ignore_index=True).to_csv(out / "cv_predictions.csv", index=False)
    leak = summ[["target", "r2", "design_skill"]].merge(
        summ_r[["target", "r2", "design_skill"]], on="target", suffixes=("_grouped", "_random"))
    leak["r2_gap"] = leak["r2_random"] - leak["r2_grouped"]
    leak["skill_gap"] = leak["design_skill_random"] - leak["design_skill_grouped"]
    within = per[per["target"].isin(L.PRIMARY_TARGETS)].pivot(index="climate_key", columns="target", values="r2")
    within = within.reindex(index=order, columns=L.PRIMARY_TARGETS).reset_index()
    checks_df = pd.DataFrame([{"id": c.id, "check": c.name, "severity": c.severity, "status": c.status,
                               "detail": c.detail} for c in checks])
    M = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "fast": a.fast,
         "dataset_sha256": {"training": sha_tr, "holdout": sha_ho}, "rf_params": rf,
         "shap_rows_per_climate": {"primary": shap_n_primary, "secondary": shap_n_secondary},
         "thresholds": {"r2_pooled_min": L.R2_POOLED_MIN, "design_skill_min": L.DESIGN_SKILL_MIN,
                        "ks_alpha": L.KS_ALPHA, "input_corr_max": L.INPUT_CORR_MAX, "physics_alpha": L.PHYSICS_ALPHA},
         "heating_climates": heat, "decision": decision, "gate": decision["gate"],
         "checks": checks_df.to_dict("records"),
         "signal_grouped": summ.to_dict("records"), "signal_random": summ_r.to_dict("records"),
         "leakage_gap": leak.to_dict("records"), "per_climate": per.to_dict("records"),
         "learning_curve": lc.to_dict("records"), "physics": phys.to_dict("records"),
         "physics_status": pstat.to_dict("records"), "facade": fac.to_dict("records"),
         "core_roof": core.to_dict("records"), "coverage_min_p": float(cov["p"].min()),
         "input_corr": cor.to_dict("records"),
         "shap_importance": {t: s.to_dict() for t, s in shap_imp.items()},
         "shap_group_share": shares_df.to_dict("index"),
         "shap_group_share_by_climate": {t: m.to_dict("index") for t, m in per_clim.items()},
         "eui_decomposition": eui.to_dict("index"), "figures": figs, "environment": environment(),
         "runtime_s": round(time.time() - t0, 1)}
    (out / "eda_metrics.json").write_text(json.dumps(M, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                                          encoding="utf-8")
    T = {"checks": checks_df,
         "summary": summ[["target", "r2", "climate_only_r2", "design_skill", "skill_climates", "within_r2_mean",
                          "mae", "rmse", "cv_rmse"]],
         "within": within, "leak": leak, "learning": lc,
         "physics": pstat[["rule", "severity", "status", "rho_min", "rho_max", "opposite", "weak"]],
         "core": core, "facade": fac[["climate_key", "latitude", "applies", "mean_resid_W_SW", "mean_resid_N", "pass"]],
         "shares": shares_df.reset_index().rename(columns={"index": "target"}), "eui": eui,
         "resid": resid.groupby("target", sort=False).head(5)[
             [c for c in ("target", "run_id", "y", "yhat", "abs_err", "outlier_targets") if c in resid.columns]]}
    write_report(out / "eda_report.md", M, T, figs, a.fast)
    step(f"done -> {out}")
    print(f"\nGATE: {decision['gate']}   DECISION: {decision['decision']}\n  {decision['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
