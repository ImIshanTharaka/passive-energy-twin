#!/usr/bin/env python3
"""
pilot_report.py - Validation report for a simulated batch (Sub-task 2.5.6).

Reads the sample file + the batch folder (run_log.csv, err_*.csv, results.csv;
the parsers are run first if their CSVs are missing or older than run_log.csv),
checks the acceptance criteria of the 2.5 plan, and writes:

    <batch>/pilot_report.md          PASS / WARN / FAIL per check, tables, findings
    <batch>/report_figures/*.png     4 figures

Checks
------
  R  reliability      all runs ok; 0 fatal; 0 severe; non-benign warnings by design
  P  physics          climate ordering (heating, cooling, EUI); EUI range
  D  directions       Spearman rho of inputs vs targets, per climate (screen for STRONG
                      effects only - with n ~ 42 and 21 parameters varying together,
                      |rho| < ~0.3 is not distinguishable from noise)
  E  edge cases       the 8 extreme designs all clean
  V  VT pair          (optional --vtpair) S2: targets identical when only VT changes
  K  peaks            hour-of-day of steady peaks (recovery spill-over check)
  M  meter gap        zone sensible vs EnergyTransfer meter
  G  ground coupling  Colombo heating vs floor insulation (ground-temperature diagnostic)
  T  runtime & disk   projection for the full campaign

USAGE (from the repository root)
-----
  python runner/pilot_report.py --samples sampling/samples_pilot.csv --batch outputs/pilot --vtpair outputs/vtpair
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runner import err_parser, output_parser          # noqa: E402

CLIMATE_ORDER = ["colombo", "phoenix", "san_francisco", "minneapolis"]
# Categorical palette (validated: scripts/validate_palette.js, light surface #fcfcfb)
CLIMATE_COLORS = {"colombo": "#eb6834", "phoenix": "#eda100",
                  "san_francisco": "#1baf7a", "minneapolis": "#2a78d6"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#1f1f1e", "#5c5b55", "#e4e3de"
DIV_NEG, DIV_MID, DIV_POS = "#2a78d6", "#f0efec", "#e34948"

INPUTS = ["floor_area", "aspect_ratio", "n_floors", "floor_height", "wall_r_value", "roof_r_value",
          "floor_r_value", "wall_thermal_mass", "infiltration_ach", "wwr_north", "wwr_east",
          "wwr_south", "wwr_west", "glazing_u", "glazing_shgc", "overhang_depth_n",
          "overhang_depth_e", "overhang_depth_s", "overhang_depth_w", "sidefin_depth", "orientation"]
TARGETS = {"heat_sens_kwh_m2": "heating", "cool_sens_kwh_m2": "cooling", "eui_site_kwh_m2": "EUI",
           "peak_heat_steady_w_m2": "peak heat", "peak_cool_steady_w_m2": "peak cool"}
# (target, derived/raw input, expected sign, climates where the effect must be strong)
DIRECTION_TESTS = [
    ("heat_sens_kwh_m2", "glazing_u", +1, ["minneapolis", "san_francisco", "phoenix"]),
    ("cool_sens_kwh_m2", "shgc_x_wwr", +1, CLIMATE_ORDER),
    ("peak_cool_steady_w_m2", "shgc_x_wwr", +1, CLIMATE_ORDER),
    ("peak_heat_steady_w_m2", "glazing_u", +1, ["minneapolis", "san_francisco", "phoenix"]),
    ("heat_sens_kwh_m2", "wall_r_value", -1, []),
    ("heat_sens_kwh_m2", "roof_r_value", -1, []),
    ("heat_sens_kwh_m2", "infiltration_ach", +1, []),
    ("cool_sens_kwh_m2", "mean_overhang", -1, []),
]
STRONG_RHO = 0.3
EUI_BAND = (50.0, 300.0)
CAMPAIGN_RUNS = 3400


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _stale(out: Path, src: Path) -> bool:
    return not out.exists() or out.stat().st_mtime < src.stat().st_mtime


def load(samples: Path, batch: Path) -> dict:
    log_path = batch / "run_log.csv"
    if not log_path.exists():
        raise SystemExit(f"{log_path} not found - run the batch first")
    if _stale(batch / "err_summary.csv", log_path):
        err_parser.aggregate(batch)
    if _stale(batch / "results.csv", log_path):
        output_parser.parse_batch(batch)
    log = pd.read_csv(log_path).drop_duplicates("run_id", keep="last")
    s = pd.read_csv(samples)
    res = pd.read_csv(batch / "results.csv").drop(columns=["n_floors"])
    d = s.merge(res, on="run_id", how="inner")
    d["shgc_x_wwr"] = d.glazing_shgc * d[["wwr_north", "wwr_east", "wwr_south", "wwr_west"]].mean(axis=1)
    d["mean_overhang"] = d[["overhang_depth_n", "overhang_depth_e", "overhang_depth_s",
                            "overhang_depth_w"]].mean(axis=1)
    return {"samples": s, "log": log, "data": d,
            "err_sum": pd.read_csv(batch / "err_summary.csv"),
            "err_msg": pd.read_csv(batch / "err_messages.csv"),
            "res_err": pd.read_csv(batch / "results_errors.csv")}


def spearman(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank()))


# ---------------------------------------------------------------------------
# Checks -> (status, title, markdown body)
# ---------------------------------------------------------------------------

def check_reliability(D) -> tuple:
    log, es, em, s = D["log"], D["err_sum"], D["err_msg"], D["samples"]
    n, ok = len(s), int((log.status == "ok").sum())
    fatal = int(es.terminated_fatal.sum())
    severe_runs = es[pd.to_numeric(es.reported_severe, errors="coerce").fillna(0) > 0].run_id.tolist()
    nb = em[em.benign == 0]
    lines = [f"- Runs: **{ok}/{n} ok**, failed {int((log.status == 'failed').sum())}, "
             f"build errors {int((log.status == 'build_error').sum())}; parser problems: {len(D['res_err'])}",
             f"- Fatal: {fatal}; runs with severe errors: **{len(severe_runs)}**"
             + (f" ({', '.join(severe_runs)})" if severe_runs else ""),
             f"- Clean runs (completed, no severe, only benign warnings): {int(es.clean.sum())}/{len(es)}"]
    if len(nb):
        lines += ["", "| category | level | runs | designs |", "|---|---|---|---|"]
        for (cat, lvl), g in nb.groupby(["category", "level"]):
            designs = sorted({r.split("_")[1] for r in g.run_id})
            lines.append(f"| {cat} | {lvl} | {g.run_id.nunique()} | {', '.join(designs[:14])}"
                         f"{' …' if len(designs) > 14 else ''} |")
        lines += ["", "Signatures:"] + [f"- `{sig}` ({k}×)" for sig, k in
                                        nb.signature.value_counts().items()]
    status = "PASS" if ok == n and fatal == 0 and not severe_runs and len(D["res_err"]) == 0 else "FAIL"
    if status == "PASS" and len(nb):
        status = "WARN"
    return status, "R. Reliability", "\n".join(lines)


def check_physics(D) -> tuple:
    d = D["data"]
    piv = lambda col: d.pivot(index="design_id", columns="climate_key", values=col)     # noqa: E731
    h, c, e = piv("heat_sens_kwh_m2"), piv("cool_sens_kwh_m2"), piv("eui_site_kwh_m2")
    col = d[d.climate_key == "colombo"].heat_sens_kwh_m2
    rows = [
        ("Minneapolis has the highest heating", (h.idxmax(axis=1) == "minneapolis").mean(), 1.0),
        ("San Francisco has the lowest EUI", (e.idxmin(axis=1) == "san_francisco").mean(), 1.0),
        ("Colombo heating < 2 kWh/m²", (col < 2).mean(), 1.0),
        ("Colombo cooling ≥ Phoenix (per design)", (c.colombo >= c.phoenix).mean(), None),
    ]
    in_band = d.eui_site_kwh_m2.between(*EUI_BAND)
    lines = ["| check | share of designs | required |", "|---|---|---|"]
    status = "PASS"
    for name, share, req in rows:
        lines.append(f"| {name} | {share:.0%} | {'100%' if req else 'info'} |")
        if req and share < req:
            status = "WARN"
    lines.append(f"| EUI within {EUI_BAND[0]:.0f}–{EUI_BAND[1]:.0f} kWh/m² | {in_band.mean():.0%} | info |")
    lines += ["", f"Colombo heating: median {col.median():.2f}, max {col.max():.2f} kWh/m². "
              f"Median cooling Colombo {c.colombo.median():.1f} vs Phoenix {c.phoenix.median():.1f} kWh/m².", "",
              "Per-climate distribution (min / median / max):", "",
              "| target | " + " | ".join(CLIMATE_ORDER) + " |", "|---|" + "---|" * len(CLIMATE_ORDER)]
    for t, name in TARGETS.items():
        cells = []
        for k in CLIMATE_ORDER:
            v = d[d.climate_key == k][t]
            cells.append(f"{v.min():.1f} / {v.median():.1f} / {v.max():.1f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    out = d[~in_band]
    if len(out):
        lines += ["", "EUI outside band: " + ", ".join(f"{r.run_id} ({r.eui_site_kwh_m2:.0f})"
                                                     for r in out.itertuples())]
    return status, "P. Physics by climate", "\n".join(lines)


def check_directions(D) -> tuple:
    L = D["data"][D["data"].design_set == "pilot_lhs"]
    n = L.groupby("climate_key").size().min()
    lines = [f"Spearman ρ on the LHS designs (n = {n} per climate). Bold = required strong effect "
             f"(expected sign and |ρ| ≥ {STRONG_RHO}).", "",
             "| target | input | expected | " + " | ".join(CLIMATE_ORDER) + " |",
             "|---|---|---|" + "---|" * len(CLIMATE_ORDER)]
    status = "PASS"
    for t, x, sign, strong in DIRECTION_TESTS:
        cells = []
        for k in CLIMATE_ORDER:
            g = L[L.climate_key == k]
            rho = spearman(g[t], g[x])
            req = k in strong
            ok = (np.sign(rho) == sign) and abs(rho) >= STRONG_RHO
            if req and not ok:
                status = "FAIL"
            cells.append(f"**{rho:+.2f}**" if req else f"{rho:+.2f}")
        lines.append(f"| {TARGETS[t]} | {x} | {'+' if sign > 0 else '−'} | " + " | ".join(cells) + " |")
    lines += ["", "Rows without bold entries are weak effects that a 42-design screen cannot resolve "
              "(SE of ρ ≈ 0.16); they need a one-at-a-time response test."]
    return status, "D. Direction of effects", "\n".join(lines)


def check_edge(D) -> tuple:
    d, es = D["data"], D["err_sum"].set_index("run_id")
    E = d[d.design_set == "edge"]
    if E.empty:
        return "INFO", "E. Edge cases", "No edge-case designs in this batch."
    bad = [r for r in E.run_id if not es.loc[r, "clean"]]
    piv = E.pivot(index="design_id", columns="climate_key", values="eui_site_kwh_m2")[CLIMATE_ORDER]
    lines = [f"Edge-case runs: {len(E)}; not clean: {len(bad)}" + (f" ({', '.join(bad)})" if bad else ""),
             "", "EUI (kWh/m²):", "", "| design | " + " | ".join(CLIMATE_ORDER) + " |",
             "|---|" + "---|" * len(CLIMATE_ORDER)]
    for dsg, row in piv.iterrows():
        lines.append(f"| {dsg} | " + " | ".join(f"{v:.1f}" for v in row) + " |")
    return ("PASS" if not bad else "WARN"), "E. Edge cases", "\n".join(lines)


def check_vtpair(vt_dir: Path | None) -> tuple:
    if vt_dir is None or not (vt_dir / "run_log.csv").exists():
        return "INFO", "V. VT-only pair (Decision S2)", "Not run."
    if _stale(vt_dir / "results.csv", vt_dir / "run_log.csv"):
        output_parser.parse_batch(vt_dir)
    r = pd.read_csv(vt_dir / "results.csv").set_index("run_id")
    num = r.select_dtypes("number")
    lines, status = [], "PASS"
    for a in [i for i in r.index if "vtA" in i]:
        b = a.replace("vtA", "vtB")
        diff = (num.loc[a] - num.loc[b]).abs() / num.loc[a].abs().clip(lower=1.0)
        n_diff = int((diff.fillna(0) > 1e-6).sum())
        if n_diff:
            status = "FAIL"
        lines.append(f"- {a} vs {b}: {n_diff} of {num.shape[1]} numeric columns differ (rel. > 1e-6)")
    return status, "V. VT-only pair (Decision S2)", "\n".join(lines)


def check_peaks(D) -> tuple:
    d = D["data"]
    lines = ["Hour-of-day (hour ending) of the steady-state peaks. Excluded transition hours: "
             "07, 08, 23 (Mon–Fri); 07, 08, 19 (Sat). A cluster at hour 09 would mean recovery lasts > 2 h.",
             "", "| peak | " + " | ".join(f"{h:02d}" for h in range(1, 25)) + " |", "|---|" + "---|" * 24]
    status = "PASS"
    for m in ("heat", "cool"):
        hrs = d[f"peak_{m}_steady_at"].dropna().str[-2:].astype(int)
        cnt = hrs.value_counts()
        lines.append(f"| {m} | " + " | ".join(str(cnt.get(h, "")) for h in range(1, 25)) + " |")
        if m == "heat":
            share9 = (hrs == 9).mean()
            if share9 > 0.10:
                status = "WARN"
            h9 = d[d.peak_heat_steady_at.str[-2:] == "09"]
            lines.append("")
            lines.append(f"Heating steady peaks at hour 09: {share9:.0%} of runs; ratio steady/hourly peak "
                         f"there median {(h9.peak_heat_steady_w_m2 / h9.peak_heat_hourly_w_m2).median():.2f}.")
    return status, "K. Peak timing (Decision T4)", "\n".join(lines)


def check_meter_gap(D) -> tuple:
    d = D["data"].copy()
    d["gap_heat_pct"] = 100 * (d.heat_sens_kwh_m2 - d.heat_meter_kwh_m2) / d.heat_sens_kwh_m2
    d["gap_cool_pct"] = 100 * (d.cool_sens_kwh_m2 - d.cool_meter_kwh_m2) / d.cool_sens_kwh_m2
    lines = ["Zone sensible minus EnergyTransfer meter, % of zone sensible (annual):", "",
             "| climate | heating mean | heating max | cooling mean | cooling max |", "|---|---|---|---|---|"]
    for k in CLIMATE_ORDER:
        g = d[d.climate_key == k]
        lines.append(f"| {k} | {g.gap_heat_pct.mean():.1f}% | {g.gap_heat_pct.max():.1f}% | "
                     f"{g.gap_cool_pct.mean():.2f}% | {g.gap_cool_pct.max():.2f}% |")
    lines += ["", f"Max unmet hours: heating {d.unmet_heat_h.max():.2f}, cooling {d.unmet_cool_h.max():.2f}.",
              f"Latent share of cooling (median): " + ", ".join(
                  f"{k} {d[d.climate_key == k].cool_latent_frac.median():.0%}" for k in CLIMATE_ORDER)]
    return "INFO", "M. Meter vs zone sensible; unmet hours; latent share", "\n".join(lines)


def check_ground(D) -> tuple:
    c = D["data"][D["data"].climate_key == "colombo"]
    if c.empty:
        return "INFO", "G. Ground coupling", "No Colombo runs."
    rho = spearman(c.heat_sens_kwh_m2, c.floor_r_value)
    status = "FAIL" if rho < -0.5 and c.heat_sens_kwh_m2.max() > 2 else "PASS"
    lines = [f"Colombo heating vs floor insulation: Spearman ρ = **{rho:+.2f}**; "
             f"max Colombo heating {c.heat_sens_kwh_m2.max():.2f} kWh/m².",
             "Colombo's ground is ~27–29 °C; heating that falls with floor insulation there means the slab "
             "is losing heat to a ground colder than the zone - i.e. the ground temperature is wrong for "
             "this climate (FAIL when ρ < −0.5 and heating > 2 kWh/m²)."]
    return status, "G. Ground coupling (ground-temperature diagnostic)", "\n".join(lines)


def check_runtime(D, batch: Path) -> tuple:
    log = D["log"]
    meta_workers = None
    meta = batch / "run_meta.json"
    if meta.exists():
        import json
        hist = json.loads(meta.read_text())
        if hist:
            meta_workers = hist[-1]["workers"]
            t0 = datetime.fromisoformat(hist[-1]["started"])
            t1 = datetime.fromisoformat(hist[-1]["finished"])
            wall = (t1 - t0).total_seconds()
            per_run = wall / max(hist[-1]["n_submitted"], 1)
    run_dirs = [p for p in batch.iterdir() if p.is_dir() and p.name != "report_figures"]
    size = sum(f.stat().st_size for p in run_dirs for f in p.iterdir())
    lines = [f"- EnergyPlus time per run: median {log.sim_s.median():.1f} s, "
             f"p90 {log.sim_s.quantile(.9):.1f} s, max {log.sim_s.max():.1f} s",
             f"- IDF build time per run: median {log.build_s.median():.1f} s, max {log.build_s.max():.1f} s",
             f"- Disk: {size / 1e6:.0f} MB for {len(run_dirs)} runs = {size / max(len(run_dirs), 1) / 1e6:.2f} MB/run"]
    if meta_workers:
        lines.append(f"- Throughput: {per_run:.1f} s/run on {meta_workers} workers "
                     f"→ {CAMPAIGN_RUNS} runs ≈ **{CAMPAIGN_RUNS * per_run / 3600:.1f} h**, "
                     f"≈ {CAMPAIGN_RUNS * size / max(len(run_dirs), 1) / 1e9:.1f} GB")
    return "INFO", "T. Runtime and disk", "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def figures(D, out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    out.mkdir(exist_ok=True)
    d = D["data"]
    files = []
    rng = np.random.default_rng(0)

    # 1. Targets by climate (small multiples, strip + median tick)
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.4), facecolor=SURFACE)
    for ax, (t, name) in zip(axes, TARGETS.items()):
        _style(ax)
        for i, k in enumerate(CLIMATE_ORDER):
            v = d[d.climate_key == k][t].values
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=10, color=CLIMATE_COLORS[k],
                       edgecolor=SURFACE, linewidth=0.5, zorder=3)
            ax.hlines(np.median(v), i - 0.3, i + 0.3, color=INK, linewidth=2, zorder=4)
        ax.set_xticks(range(4), ["Colombo", "Phoenix", "San Fran.", "Minneap."], rotation=30)
        unit = "W/m²" if "peak" in t else "kWh/m²·yr"
        ax.set_title(f"{name} ({unit})", fontsize=9, color=INK, loc="left")
    fig.suptitle("Pilot targets by climate (dots = runs, bar = median)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    files.append("fig1_targets_by_climate.png")
    fig.savefig(out / files[-1], dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # 2. Spearman heatmap inputs x (target, climate)
    L = d[d.design_set == "pilot_lhs"]
    cols = [(t, k) for t in ("heat_sens_kwh_m2", "cool_sens_kwh_m2", "eui_site_kwh_m2") for k in CLIMATE_ORDER]
    M = np.array([[spearman(L[L.climate_key == k][x], L[L.climate_key == k][t]) for (t, k) in cols]
                  for x in INPUTS])
    cmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])
    fig, ax = plt.subplots(figsize=(9.5, 8), facecolor=SURFACE)
    im = ax.imshow(M, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(INPUTS)), INPUTS, fontsize=8, color=INK)
    ax.set_xticks(range(len(cols)), [k.replace("san_francisco", "san_fran")[:9] for _, k in cols],
                  rotation=60, fontsize=7, color=INK2)
    for j, (t, _) in enumerate(cols):
        if j % 4 == 0:
            ax.text(j + 1.5, -0.8, TARGETS[t], ha="center", va="bottom", fontsize=9, color=INK)
            if j:
                ax.axvline(j - 0.5, color=SURFACE, linewidth=3)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if abs(M[i, j]) >= STRONG_RHO:
                ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=6.5,
                        color="#ffffff" if abs(M[i, j]) > 0.6 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Spearman ρ (labels shown where |ρ| ≥ 0.3)", fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    fig.suptitle("Input–target rank correlation, LHS designs, per climate", fontsize=10, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout()
    files.append("fig2_spearman_heatmap.png")
    fig.savefig(out / files[-1], dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # 3. Peak hour histograms with excluded hours shaded
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), facecolor=SURFACE, sharey=True)
    for ax, m, label in zip(axes, ("heat", "cool"), ("Heating", "Cooling")):
        _style(ax)
        for h in (7, 8, 19, 23):
            ax.axvspan(h - 0.5, h + 0.5, color=GRID, zorder=0)
        hrs = d[f"peak_{m}_steady_at"].dropna().str[-2:].astype(int).value_counts()
        ax.bar(hrs.index, hrs.values, width=0.7, color=CLIMATE_COLORS["minneapolis"] if m == "heat"
               else CLIMATE_COLORS["colombo"], zorder=3)
        ax.set_xticks(range(1, 25, 2))
        ax.set_xlim(0.5, 24.5)
        ax.set_xlabel("hour ending (shaded = excluded transition hours)", fontsize=8)
        ax.set_title(f"{label}: hour of steady-state peak (runs)", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    files.append("fig3_peak_hours.png")
    fig.savefig(out / files[-1], dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # 4. Ground diagnostic: heating vs floor R, per climate
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), facecolor=SURFACE)
    for ax, k in zip(axes, CLIMATE_ORDER):
        _style(ax)
        g = d[d.climate_key == k]
        ax.scatter(g.floor_r_value, g.heat_sens_kwh_m2, s=14, color=CLIMATE_COLORS[k],
                   edgecolor=SURFACE, linewidth=0.5, zorder=3)
        ax.set_xlabel("floor R-value (m²K/W)", fontsize=8)
        ax.set_title(f"{k}: heating (kWh/m²)  ρ={spearman(g.floor_r_value, g.heat_sens_kwh_m2):+.2f}",
                     fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    files.append("fig4_heating_vs_floor_insulation.png")
    fig.savefig(out / files[-1], dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return files


# ---------------------------------------------------------------------------

def build_report(samples: Path, batch: Path, vt_dir: Path | None) -> tuple[str, list]:
    D = load(samples, batch)
    checks = [check_reliability(D), check_physics(D), check_directions(D), check_edge(D),
              check_vtpair(vt_dir), check_peaks(D), check_meter_gap(D), check_ground(D),
              check_runtime(D, batch)]
    figs = figures(D, batch / "report_figures")
    head = [f"# Pilot validation report — `{batch.name}`", "",
            f"Generated {datetime.now():%Y-%m-%d %H:%M} by `runner/pilot_report.py` from `{samples}` "
            f"({len(D['samples'])} rows).", "", "| check | status |", "|---|---|"]
    head += [f"| {title} | **{status}** |" for status, title, _ in checks]
    body = []
    for status, title, text in checks:
        body += ["", f"## {title} — {status}", "", text]
    body += ["", "## Figures", ""] + [f"![{f}](report_figures/{f})" for f in figs]
    return "\n".join(head + body) + "\n", checks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--samples", type=Path, default=ROOT / "sampling" / "samples_pilot.csv")
    ap.add_argument("--batch", type=Path, default=ROOT / "outputs" / "pilot")
    ap.add_argument("--vtpair", type=Path)
    a = ap.parse_args(argv)
    text, checks = build_report(a.samples, a.batch, a.vtpair)
    (a.batch / "pilot_report.md").write_text(text)
    print(f"wrote {a.batch / 'pilot_report.md'}")
    for status, title, _ in checks:
        print(f"  {status:5} {title}")
    return 0 if all(s != "FAIL" for s, _, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
