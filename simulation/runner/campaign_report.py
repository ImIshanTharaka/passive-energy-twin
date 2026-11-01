#!/usr/bin/env python3
"""
campaign_report.py - QA report for the full simulation campaign (Sub-task 2.6).

Generalises the 2.5 pilot report (runner/pilot_report.py, which is fixed to the
four pilot climates) to any set of climates and design sets. Climates, their
order and the climate-dependent rules all come from climate/climate_metadata.csv,
not from constants, so the same code checks the R1 smoke, the main campaign and
the holdout.

CHECKS -> (status, title, markdown)
  C  completeness      every sample row logged ok and parsed; the 10 TARGETS finite and >= 0
  A  anchor integrity  every anchor design present in all of its climates (GroupKFold, 2.7/3)
  R  reliability       fatal / severe / non-benign messages; categories new vs the pilot, per climate
  P  physics           anchor response to HDD18 / CDD10; zero-HDD heating; EUI band; distributions
  D  directions        Spearman sign/strength per training climate (main batch, n = 300)
  H  holdout coverage  holdout targets inside training ranges; London features vs training range
  K  peak timing       hour-of-day of steady peaks (Decision T4)
  G  ground coupling   heating vs floor insulation in zero-HDD climates (Decision G1)
  M  diagnostics       zone sensible vs EnergyTransfer meter; unmet hours; latent share
  T  runtime & disk    actual throughput and storage (thesis methodology)

Gate (claude/Phase2.6_Execution_Plan.md section 4): GO if no FAIL and no WARN,
REVIEW if WARN only, NO-GO if any FAIL. Exit code 1 on any FAIL.

USAGE (from the repository root)
-----
  python runner/campaign_report.py                  # outputs/main + outputs/holdout
  python runner/campaign_report.py --partial        # smoke / mid-run: only rows already attempted
  python runner/campaign_report.py --set pilot sampling/samples_pilot.csv outputs/pilot --out outputs/pilot
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runner import err_parser, output_parser                         # noqa: E402
from runner import pilot_report as PR                                # noqa: E402
from climate.climate_features import CONTINUOUS_FEATURES             # noqa: E402

LOG = "run_log.csv"
METADATA = ROOT / "climate" / "climate_metadata.csv"
DEFAULT_SETS = [("main", ROOT / "sampling" / "samples_main.csv", ROOT / "outputs" / "main"),
                ("holdout", ROOT / "sampling" / "samples_holdout.csv", ROOT / "outputs" / "holdout")]
PILOT_MESSAGES = ROOT / "outputs" / "pilot" / "err_messages.csv"
# Runs whose severe/non-benign messages were investigated and shown not to affect the
# targets (run_id, category, evidence). They are listed in R but do not FAIL/WARN it.
ACCEPTED = ROOT / "outputs" / "accepted_exceptions.csv"

# target column -> short name (the 10 ML targets, Decision T1/T4)
TARGETS = {v: k for k, v in output_parser.TARGETS.items()}
MAIN5 = {"heat_sens_kwh_m2": "heating", "cool_sens_kwh_m2": "cooling", "eui_site_kwh_m2": "EUI",
         "peak_heat_steady_w_m2": "peak heat", "peak_cool_steady_w_m2": "peak cool"}
INPUTS = PR.INPUTS
EUI_BAND = PR.EUI_BAND
STRONG_RHO = PR.STRONG_RHO
MIN_N_DIRECTIONS = 30          # below this Spearman per climate is not interpretable
COMPLETE_WARN = 0.99           # share parsed for WARN instead of FAIL (gate section 4)
HEAT_RHO_MIN, COOL_RHO_MIN = 0.9, 0.8   # anchor response plausibility screen (WARN only)
ZERO_HDD = 1.0                 # hdd18 below this = no heating season (Colombo, Chennai)

# (target, input, expected sign, (climate feature, threshold) or None = every climate)
# Tier 1 = strong effect required (sign and |rho| >= STRONG_RHO) -> FAIL if missed.
STRONG_TESTS = [
    ("cool_sens_kwh_m2", "shgc_x_wwr", +1, None),
    ("peak_cool_steady_w_m2", "shgc_x_wwr", +1, None),
    ("heat_sens_kwh_m2", "glazing_u", +1, ("hdd18", 400.0)),
    ("peak_heat_steady_w_m2", "glazing_u", +1, ("hdd18", 400.0)),
]
# Tier 2 = weak effects (unresolvable in the 42-design pilot). Their direction was established by
# the one-at-a-time test (2.5, runner/oat.py); here they WARN only when the sign is significantly
# WRONG (|rho| >= 2 SE with the opposite sign) - a small but correct effect is not a defect.
SIGN_TESTS = [
    ("heat_sens_kwh_m2", "wall_r_value", -1, ("hdd18", 2000.0)),
    ("heat_sens_kwh_m2", "roof_r_value", -1, ("hdd18", 2000.0)),
    ("heat_sens_kwh_m2", "infiltration_ach", +1, ("hdd18", 2000.0)),
    ("cool_sens_kwh_m2", "mean_overhang", -1, ("cdd10", 5000.0)),
]

SURFACE, INK, INK2, GRID = PR.SURFACE, PR.INK, PR.INK2, PR.GRID
DOT, ACCENT = "#2a78d6", "#eb6834"          # training runs / holdout climate (pilot palette)

spearman = PR.spearman


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _refresh(batch: Path) -> None:
    log_path = batch / LOG
    if PR._stale(batch / "err_summary.csv", log_path):
        err_parser.aggregate(batch)
    if PR._stale(batch / "results.csv", log_path):
        output_parser.parse_batch(batch)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _applies(rule, m: pd.Series) -> bool:
    return rule is None or float(m[rule[0]]) >= rule[1]


def load(sets, partial: bool = False) -> dict:
    """sets = [(name, samples_csv, batch_dir)]. Returns the frames every check uses."""
    meta = pd.read_csv(METADATA)
    S, logs, res, es, em, rerr, batches = [], [], [], [], [], [], []
    for name, samples, batch in sets:
        samples, batch = Path(samples), Path(batch)
        s = pd.read_csv(samples)
        s.insert(0, "set", name)
        if not (batch / LOG).exists():
            if not partial:
                S.append(s)
            batches.append((name, samples, batch, False))
            continue
        _refresh(batch)
        log = pd.read_csv(batch / LOG).drop_duplicates("run_id", keep="last")
        if partial:
            s = s[s.run_id.isin(log.run_id)]
        S.append(s)
        logs.append(log)
        res.append(_read(batch / "results.csv"))
        es.append(_read(batch / "err_summary.csv"))
        em.append(_read(batch / "err_messages.csv"))
        rerr.append(_read(batch / "results_errors.csv"))
        batches.append((name, samples, batch, True))
    if not S or not logs:
        raise SystemExit("no batch has been run yet (no run_log.csv found)")
    cat = lambda xs: pd.concat([x for x in xs if len(x)], ignore_index=True) if any(len(x) for x in xs) \
        else pd.DataFrame()                                                          # noqa: E731
    S, log, R = cat(S), cat(logs), cat(res)
    feats = meta.rename(columns={"key": "climate_key"})[["climate_key", "role", *CONTINUOUS_FEATURES]]
    d = pd.DataFrame()
    if len(R):
        d = S.merge(R.drop(columns=["n_floors"], errors="ignore"), on="run_id", how="inner")
        d = d.merge(feats, on="climate_key", how="left")
        d["shgc_x_wwr"] = d.glazing_shgc * d[["wwr_north", "wwr_east", "wwr_south", "wwr_west"]].mean(axis=1)
        d["mean_overhang"] = d[["overhang_depth_n", "overhang_depth_e", "overhang_depth_s",
                                "overhang_depth_w"]].mean(axis=1)
    m = meta.set_index("key")
    order = m.loc[sorted(set(S.climate_key))].sort_values("hdd18", ascending=False).index.tolist()
    return {"samples": S, "log": log, "data": d, "err_sum": cat(es), "err_msg": cat(em),
            "res_err": cat(rerr), "meta": m, "order": order,
            "training": [k for k in order if m.loc[k, "role"] == "training"],
            "holdout_climates": [k for k in order if m.loc[k, "role"] == "holdout"],
            "batches": batches, "partial": partial}


def _fmt(x: float, nd: int = 2) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{nd}f}"


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_completeness(D) -> tuple:
    S, log, d = D["samples"], D["log"], D["data"]
    status_of = log.set_index("run_id").status
    S = S.assign(status=S.run_id.map(status_of).fillna("not run"),
                 parsed=S.run_id.isin(d.run_id) if len(d) else False)
    T = list(TARGETS)
    bad = pd.DataFrame()
    if len(d):
        v = d[T].astype(float)
        mask = ~np.isfinite(v) | (v < 0)
        bad = d.loc[mask.any(axis=1), ["run_id", *[t for t in T if mask[t].any()]]]
    lines = ["| batch | climate | expected | ok | parsed | failed | build error | not run |",
             "|---|---|---|---|---|---|---|---|"]
    order = {k: i for i, k in enumerate(D["order"])}
    g = S.groupby(["batch", "climate_key"])
    for (b, k), x in sorted(g, key=lambda kv: (kv[0][0], order[kv[0][1]])):
        lines.append(f"| {b} | {k} | {len(x)} | {(x.status == 'ok').sum()} | {x.parsed.sum()} | "
                     f"{(x.status == 'failed').sum()} | {(x.status == 'build_error').sum()} | "
                     f"{(x.status == 'not run').sum()} |")
    n, n_parsed = len(S), int(S.parsed.sum())
    tot = S.groupby("batch").agg(expected=("run_id", "size"), parsed=("parsed", "sum"))
    lines += ["", "Totals: " + "; ".join(f"{b} {int(r.parsed)}/{int(r.expected)}" for b, r in tot.iterrows())
              + f"; all {n_parsed}/{n} ({n_parsed / max(n, 1):.1%})."]
    if "holdout_ood" in tot.index:
        lines.append(f"Handoff floors: main ≥ 2,500, holdout ≥ 300 incl. ≥ 80 London — "
                     f"London parsed {int(tot.loc['holdout_ood', 'parsed'])}.")
    nok = S[~S.status.isin(["ok", "not run"])]
    if len(nok):
        msg = log.set_index("run_id").message
        lines += ["", "Runs not ok:"] + [f"- `{r}` ({st}): {str(msg.get(r, ''))[:120]}"
                                         for r, st in zip(nok.run_id[:25], nok.status[:25])]
        if len(nok) > 25:
            lines.append(f"- … {len(nok) - 25} more")
    if len(D["res_err"]):
        lines += ["", f"Parser problems: {len(D['res_err'])} — " + ", ".join(D["res_err"].run_id.astype(str)[:10])]
    if len(bad):
        lines += ["", f"**Runs with NaN/inf/negative targets: {len(bad)}** — "
                  + ", ".join(bad.run_id.astype(str)[:15])]
    for _, _, batch, ran in D["batches"]:
        if not ran:
            lines.append(f"\n`{batch}` has not been run.")
    if len(bad) or len(D["res_err"]):
        status = "FAIL"
    elif D["partial"]:
        status = "INFO"
    elif n_parsed == n:
        status = "PASS"
    elif n_parsed / max(n, 1) >= COMPLETE_WARN:
        status = "WARN"
    else:
        status = "FAIL"
    return status, "C. Completeness and target validity", "\n".join(lines)


def check_anchors(D) -> tuple:
    S, d = D["samples"], D["data"]
    A = S[S.design_set == "anchor"]
    if A.empty:
        return "INFO", "A. Anchor integrity", "No anchor designs in these batches."
    expected = A.groupby("design_id").climate_key.agg(set)
    got = d[d.design_set == "anchor"].groupby("design_id").climate_key.agg(set) if len(d) else pd.Series(dtype=object)
    missing = {dsg: sorted(exp - got.get(dsg, set())) for dsg, exp in expected.items()
               if exp - got.get(dsg, set())}
    n_cl = expected.map(len)
    lines = [f"- Anchor designs: {len(expected)}; climates per anchor: {n_cl.min()}–{n_cl.max()}",
             f"- Complete in every climate: **{len(expected) - len(missing)}/{len(expected)}**"]
    if missing:
        lines += ["", "| anchor | missing climates |", "|---|---|"]
        lines += [f"| {k} | {', '.join(v)} |" for k, v in list(missing.items())[:30]]
        lines.append("\nIncomplete anchors stay in the data; GroupKFold by `design_id` handles partial groups.")
    return ("PASS" if not missing else "WARN"), "A. Anchor integrity", "\n".join(lines)


def _baseline_categories() -> set:
    em = _read(PILOT_MESSAGES)
    return set(em.category) if len(em) else {"unused_construction"}


def _accepted() -> set:
    a = _read(ACCEPTED)
    return set(zip(a.run_id, a.category)) if len(a) else set()


def check_reliability(D, accepted: set | None = None) -> tuple:
    es, em, S, log = D["err_sum"], D["err_msg"], D["samples"], D["log"]
    base = _baseline_categories()
    accepted = _accepted() if accepted is None else accepted
    fatal = int(es.terminated_fatal.sum()) if len(es) else 0
    sev = es[pd.to_numeric(es.reported_severe, errors="coerce").fillna(0) > 0].run_id.tolist() if len(es) else []
    retried = int((log.attempts > 1).sum()) if "attempts" in log else 0
    lines = [f"- .err files parsed: {len(es)}; clean (no severe, only benign warnings): "
             f"{int(es.clean.sum()) if len(es) else 0}",
             f"- Fatal: **{fatal}**; runs with severe errors: **{len(sev)}**"
             + (f" ({', '.join(sev[:15])}{' …' if len(sev) > 15 else ''})" if sev else ""),
             f"- Runs that needed a retry: {retried}",
             f"- Baseline (pilot) message categories: {', '.join(sorted(base))}"]
    acc_rows = em[[(r, c) in accepted for r, c in zip(em.run_id, em.category)]] if len(em) else em
    acc_runs = set(acc_rows.run_id)
    open_sev = [r for r in sev if r not in acc_runs or
                not (em[(em.run_id == r) & (em.level == "Severe")].category.map(
                    lambda c, r=r: (r, c) in accepted).all())]
    if acc_runs:
        lines.append(f"- Reviewed and accepted (see `{ACCEPTED.name}`): {', '.join(sorted(acc_runs))}")
    status = "FAIL" if fatal or open_sev else "PASS"
    if len(em):
        em = em[[(r, c) not in accepted for r, c in zip(em.run_id, em.category)]]
        clim = S.set_index("run_id").climate_key
        em = em.assign(climate_key=em.run_id.map(clim))
        em["new"] = ~em.category.isin(base)
        # New-but-benign categories were reviewed and classified in err_parser.CATEGORY_RULES
        # (with the evidence in a comment there): listed for the record, they do not WARN.
        for c, g in em[(em.benign == 1) & em.new].groupby("category"):
            per = g.groupby("climate_key").run_id.nunique()
            lines.append(f"- New benign category `{c}` (reviewed, see err_parser): "
                         + ", ".join(f"{k} {n}" for k, n in per.items()) + " runs")
        flagged = em[em.benign == 0]
        if len(flagged):
            status = "WARN" if status == "PASS" else status
            cols = [k for k in D["order"] if k in set(flagged.climate_key)]
            piv = flagged.groupby(["category", "level", "climate_key"]).run_id.nunique().unstack(fill_value=0)
            lines += ["", "Non-benign categories (runs per climate):", "",
                      "| category | level | new | " + " | ".join(cols) + " |",
                      "|---|---|---|" + "---|" * len(cols)]
            for (c, lvl), row in piv.iterrows():
                lines.append(f"| {c} | {lvl} | {'yes' if c not in base else ''} | "
                             + " | ".join(str(int(row.get(k, 0)) or "") for k in cols) + " |")
            lines += ["", "Signatures:"] + [f"- `{s}` ({n}×)" for s, n in
                                            flagged.signature.value_counts().head(15).items()]
    return status, "R. Reliability (EnergyPlus messages)", "\n".join(lines)


def check_physics(D) -> tuple:
    d, m, order = D["data"], D["meta"], D["order"]
    if d.empty:
        return "INFO", "P. Physics by climate", "No parsed runs."
    status, lines = "PASS", []
    # 1. anchor response: per anchor, rank correlation across climates
    tr = d[d.climate_key.isin(D["training"]) & (d.design_set == "anchor")]
    groups = [g for _, g in tr.groupby("design_id") if g.climate_key.nunique() >= 4]
    if groups:
        rh = pd.Series([spearman(g.heat_sens_kwh_m2, g.hdd18) for g in groups]).dropna()
        rc = pd.Series([spearman(g.cool_sens_kwh_m2, g.cdd10) for g in groups]).dropna()
        coldest = max(D["training"], key=lambda k: m.loc[k, "hdd18"])
        top = np.mean([g.loc[g.heat_sens_kwh_m2.idxmax(), "climate_key"] == coldest for g in groups])
        lines += ["Anchor designs, across training climates (Spearman per design, then summarised):", "",
                  "| response | median ρ | min ρ | screen |", "|---|---|---|---|",
                  f"| heating vs HDD18 | {rh.median():+.2f} | {rh.min():+.2f} | median ≥ {HEAT_RHO_MIN} |",
                  f"| cooling vs CDD10 | {rc.median():+.2f} | {rc.min():+.2f} | median ≥ {COOL_RHO_MIN} |",
                  "", f"{coldest} has the highest heating for {top:.0%} of {len(groups)} anchors.", ""]
        if rh.median() < HEAT_RHO_MIN or rc.median() < COOL_RHO_MIN:
            status = "WARN"
    # 2. zero-HDD climates
    for k in [k for k in order if m.loc[k, "hdd18"] < ZERO_HDD]:
        h = d[d.climate_key == k].heat_sens_kwh_m2
        if len(h):
            share = (h < 2).mean()
            lines.append(f"- {k} (HDD18 = 0): heating < 2 kWh/m² in {share:.0%} of runs "
                         f"(median {h.median():.2f}, max {h.max():.2f})")
            if share < 1:
                status = "WARN"
    band = d.eui_site_kwh_m2.between(*EUI_BAND)
    lines += [f"- EUI within {EUI_BAND[0]:.0f}–{EUI_BAND[1]:.0f} kWh/m²: {band.mean():.1%} of runs "
              f"(range {d.eui_site_kwh_m2.min():.0f}–{d.eui_site_kwh_m2.max():.0f})", "",
              "Median [p5–p95] per climate (climates ordered by HDD18; kWh/m²·yr, peaks W/m²):", "",
              "| climate | n | " + " | ".join(MAIN5.values()) + " |", "|---|---|" + "---|" * len(MAIN5)]
    for k in order:
        g = d[d.climate_key == k]
        if g.empty:
            continue
        cells = [f"{g[t].median():.1f} [{g[t].quantile(.05):.1f}–{g[t].quantile(.95):.1f}]" for t in MAIN5]
        lines.append(f"| {k} | {len(g)} | " + " | ".join(cells) + " |")
    return status, "P. Physics by climate", "\n".join(lines)


def _direction_frame(D) -> pd.DataFrame:
    d = D["data"]
    tr = d[d.climate_key.isin(D["training"]) & (d.design_set != "edge")]   # edge = pilot extremes
    main = tr[tr.batch == "main"]
    return main if len(main) else tr


def check_directions(D) -> tuple:
    L, m = _direction_frame(D), D["meta"]
    if L.empty:
        return "INFO", "D. Direction of effects", "No training-climate runs."
    n = L.groupby("climate_key").size()
    cl = [k for k in D["training"] if n.get(k, 0) >= MIN_N_DIRECTIONS]
    if not cl:
        return "INFO", "D. Direction of effects", (f"Too few runs per climate (max {n.max()}, "
                                                    f"need {MIN_N_DIRECTIONS}).")
    se2 = 2 / np.sqrt(max(n[cl].min() - 1, 1))
    lines = [f"Spearman ρ per climate (n = {n[cl].min()}–{n[cl].max()}). **Bold** = tier 1, required: "
             f"expected sign and |ρ| ≥ {STRONG_RHO} (FAIL). *Italic* = tier 2, weak effect: WARN only if the "
             f"sign is significantly wrong (|ρ| ≥ 2 SE = {se2:.2f}, marked ✗); ✓ = expected sign and significant. "
             "Plain = not checked in that climate.", "",
             "| target | input | exp. | " + " | ".join(cl) + " |", "|---|---|---|" + "---|" * len(cl)]
    status = "PASS"
    for tier, tests in ((1, STRONG_TESTS), (2, SIGN_TESTS)):
        for t, x, sign, rule in tests:
            cells = []
            for k in cl:
                g = L[L.climate_key == k]
                rho = spearman(g[t], g[x])
                req = _applies(rule, m.loc[k])
                thr = STRONG_RHO if tier == 1 else se2
                ok = not np.isnan(rho) and np.sign(rho) == sign and abs(rho) >= thr
                wrong = not np.isnan(rho) and np.sign(rho) == -sign and abs(rho) >= thr
                txt = _fmt(rho)
                if tier == 1:
                    if req and not ok:
                        status = "FAIL"
                    cells.append(f"**{txt}**" if req else txt)
                else:
                    if req and wrong and status == "PASS":
                        status = "WARN"
                    mark = " ✗" if wrong else (" ✓" if ok else "")
                    cells.append(f"*{txt}*{mark}" if req else txt)
            lines.append(f"| {MAIN5.get(t, t)} | {x} | {'+' if sign > 0 else '−'} | " + " | ".join(cells) + " |")
    return status, "D. Direction of effects", "\n".join(lines)


def check_holdout(D) -> tuple:
    d, m = D["data"], D["meta"]
    lines = []
    main = d[(d.batch == "main")] if len(d) else d
    ho = d[d.design_set == "holdout"] if len(d) else d
    if len(ho) and len(main):
        lines += ["In-distribution holdout: share of runs inside the main batch's range for the same climate:", "",
                  "| target | inside |", "|---|---|"]
        rng = main.groupby("climate_key")[list(TARGETS)].agg(["min", "max"])
        for t, name in TARGETS.items():
            lo = ho.climate_key.map(rng[(t, "min")])
            hi = ho.climate_key.map(rng[(t, "max")])
            lines.append(f"| {name} | {ho[t].between(lo, hi).mean():.1%} |")
        lines.append("")
    tr_keys = D["training"]
    for k in D["holdout_climates"]:
        lines += [f"Held-out climate **{k}** — climate features vs the training range:", "",
                  "| feature | training min | training max | " + k + " | position |", "|---|---|---|---|---|"]
        for f in CONTINUOUS_FEATURES:
            lo, hi, v = m.loc[tr_keys, f].min(), m.loc[tr_keys, f].max(), m.loc[k, f]
            pos = "**below**" if v < lo else ("**above**" if v > hi else "inside")
            lines.append(f"| {f} | {lo:g} | {hi:g} | {v:g} | {pos} |")
        lines += ["", "A feature outside the training range makes this an extrapolation test for that "
                  "feature; Phase 3 should report holdout accuracy with this caveat (RQ1)."]
        H = d[d.climate_key == k] if len(d) else d
        A = d[d.climate_key.isin(tr_keys) & (d.design_set == "anchor")] if len(d) else d
        if len(H) and len(A):
            rng = A.groupby("design_id")[list(TARGETS)].agg(["min", "max"])
            lines += ["", f"{k} anchors: share of runs inside the same design's range over the training "
                      "climates (output-space interpolation):", "", "| target | inside | median |", "|---|---|---|"]
            for t, name in TARGETS.items():
                lo = H.design_id.map(rng[(t, "min")])
                hi = H.design_id.map(rng[(t, "max")])
                lines.append(f"| {name} | {H[t].between(lo, hi).mean():.0%} | {H[t].median():.1f} |")
        lines.append("")
    if not lines:
        lines = ["No holdout runs in these batches."]
    return "INFO", "H. Holdout coverage", "\n".join(lines)


def check_ground(D) -> tuple:
    d, m = D["data"], D["meta"]
    lines, status = [], "PASS"
    for k in [k for k in D["order"] if m.loc[k, "hdd18"] < ZERO_HDD]:
        c = d[d.climate_key == k] if len(d) else d
        if len(c) < 5:
            continue
        rho = spearman(c.heat_sens_kwh_m2, c.floor_r_value)
        if rho < -0.5 and c.heat_sens_kwh_m2.max() > 2:
            status = "FAIL"
        lines.append(f"- {k}: heating vs floor insulation ρ = **{_fmt(rho)}**, "
                     f"max heating {c.heat_sens_kwh_m2.max():.2f} kWh/m²")
    if not lines:
        return "INFO", "G. Ground coupling", "No zero-HDD climate with ≥ 5 runs."
    lines.append("\nFAIL when ρ < −0.5 and heating > 2 kWh/m² (slab losing heat to a ground that is too cold; "
                 "the pre-G1 symptom).")
    return status, "G. Ground coupling (Decision G1)", "\n".join(lines)


def check_peaks(D) -> tuple:
    if D["data"].empty:
        return "INFO", "K. Peak timing (Decision T4)", "No parsed runs."
    return PR.check_peaks({"data": D["data"]})


def check_diagnostics(D) -> tuple:
    d = D["data"]
    if d.empty:
        return "INFO", "M. Meter gap, unmet hours, latent share", "No parsed runs."
    d = d.copy()
    hs, cs = d.heat_sens_kwh_m2.where(d.heat_sens_kwh_m2 > 0.5), d.cool_sens_kwh_m2.where(d.cool_sens_kwh_m2 > 0.5)
    d["gh"] = 100 * (d.heat_sens_kwh_m2 - d.heat_meter_kwh_m2) / hs
    d["gc"] = 100 * (d.cool_sens_kwh_m2 - d.cool_meter_kwh_m2) / cs
    lines = ["Zone sensible minus EnergyTransfer meter, % of zone sensible (runs with > 0.5 kWh/m²):", "",
             "| climate | heating mean | cooling mean | latent share of cooling (median) | max unmet h (heat / cool) |",
             "|---|---|---|---|---|"]
    for k in D["order"]:
        g = d[d.climate_key == k]
        if len(g):
            lines.append(f"| {k} | {g.gh.mean():.1f}% | {g.gc.mean():.2f}% | {g.cool_latent_frac.median():.0%} | "
                         f"{g.unmet_heat_h.max():.1f} / {g.unmet_cool_h.max():.1f} |")
    return "INFO", "M. Meter gap, unmet hours, latent share", "\n".join(lines)


def check_runtime(D) -> tuple:
    lines = ["| batch | runs on disk | GB | MB/run | EnergyPlus s/run (median / p90 / max) | wall h | s/run throughput |",
             "|---|---|---|---|---|---|---|"]
    for name, _, batch, ran in D["batches"]:
        if not ran:
            continue
        log = pd.read_csv(batch / LOG).drop_duplicates("run_id", keep="last")
        ids = set(log.run_id)
        size, nd = 0, 0
        for p in batch.iterdir():
            if p.is_dir() and p.name in ids:
                nd += 1
                size += sum(f.stat().st_size for f in p.iterdir() if f.is_file())
        wall = sub = 0.0
        meta = batch / "run_meta.json"
        if meta.exists():
            for h in json.loads(meta.read_text()):
                if h.get("finished"):
                    wall += (datetime.fromisoformat(h["finished"]) - datetime.fromisoformat(h["started"])).total_seconds()
                    sub += h.get("n_submitted", 0)
        ok = log[log.status == "ok"]
        lines.append(f"| {name} | {nd} | {size / 1e9:.2f} | {size / max(nd, 1) / 1e6:.2f} | "
                     f"{ok.sim_s.median():.1f} / {ok.sim_s.quantile(.9):.1f} / {ok.sim_s.max():.1f} | "
                     f"{wall / 3600:.2f} | {wall / sub if sub else float('nan'):.1f} |")
    return "INFO", "T. Runtime and disk", "\n".join(lines)


CHECKS = [check_completeness, check_anchors, check_reliability, check_physics, check_directions,
          check_holdout, check_peaks, check_ground, check_diagnostics, check_runtime]


def gate(checks) -> str:
    st = [c[0] for c in checks]
    return "NO-GO" if "FAIL" in st else ("REVIEW" if "WARN" in st else "GO")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _short(k: str) -> str:
    return {"san_francisco": "San Fran.", "minneapolis": "Minneap.", "new_york": "New York"}.get(k, k.title())


def figures(D, out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    d, order, m = D["data"], D["order"], D["meta"]
    if d.empty:
        return []
    out.mkdir(parents=True, exist_ok=True)
    order = [k for k in order if k in set(d.climate_key)]
    hold = set(D["holdout_climates"])
    files, rng = [], np.random.default_rng(0)
    col = lambda k: ACCENT if k in hold else DOT                                         # noqa: E731

    # 1. main targets by climate
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8), facecolor=SURFACE)
    for ax, (t, name) in zip(axes, MAIN5.items()):
        PR._style(ax)
        for i, k in enumerate(order):
            v = d[d.climate_key == k][t].values
            if len(v) > 150:
                v = rng.choice(v, 150, replace=False)
            ax.scatter(i + rng.uniform(-0.22, 0.22, len(v)), v, s=8, color=col(k), alpha=0.55,
                       edgecolor="none", zorder=3)
            ax.hlines(np.median(d[d.climate_key == k][t]), i - 0.32, i + 0.32, color=INK, linewidth=2, zorder=4)
        ax.set_xticks(range(len(order)), [_short(k) for k in order], rotation=60, fontsize=7)
        ax.set_title(f"{name} ({'W/m²' if 'peak' in t else 'kWh/m²·yr'})", fontsize=9, color=INK, loc="left")
    fig.suptitle("Targets by climate, ordered by HDD18 (dots = runs, ≤150 shown; bar = median; "
                 "orange = held-out climate)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    files.append("fig1_targets_by_climate.png")
    fig.savefig(out / files[-1], dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # 2. anchor response to the climate drivers
    A = d[d.design_set == "anchor"]
    if len(A):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE)
        for ax, (t, f, lab) in zip(axes, [("heat_sens_kwh_m2", "hdd18", "heating"),
                                           ("cool_sens_kwh_m2", "cdd10", "cooling")]):
            PR._style(ax)
            for k in order:
                v = A[A.climate_key == k][t]
                if v.empty:
                    continue
                x = m.loc[k, f]
                ax.vlines(x, v.quantile(.25), v.quantile(.75), color=col(k), linewidth=2, zorder=3)
                ax.scatter([x], [v.median()], s=40, color=col(k), edgecolor=SURFACE, linewidth=1.5, zorder=4)
                ax.annotate(_short(k), (x, v.median()), xytext=(5, 3), textcoords="offset points",
                            fontsize=7, color=INK2)
            ax.set_xlabel(f"{f.upper()} (°C·day)", fontsize=8)
            ax.set_title(f"Anchor designs: {lab} (kWh/m²·yr), median and IQR", fontsize=9, color=INK, loc="left")
        fig.tight_layout()
        files.append("fig2_anchor_climate_response.png")
        fig.savefig(out / files[-1], dpi=150, facecolor=SURFACE)
        plt.close(fig)

    # 3. Spearman heatmap inputs x training climates, three targets
    L = _direction_frame(D)
    cl = [k for k in D["training"] if (L.climate_key == k).sum() >= MIN_N_DIRECTIONS]
    if cl:
        cmap = LinearSegmentedColormap.from_list("div", [PR.DIV_NEG, PR.DIV_MID, PR.DIV_POS])
        fig, axes = plt.subplots(1, 3, figsize=(15, 7.5), facecolor=SURFACE, sharey=True)
        for ax, t in zip(axes, ("heat_sens_kwh_m2", "cool_sens_kwh_m2", "eui_site_kwh_m2")):
            M = np.array([[spearman(L[L.climate_key == k][x], L[L.climate_key == k][t]) for k in cl]
                          for x in INPUTS])
            im = ax.imshow(M, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
            ax.set_yticks(range(len(INPUTS)), INPUTS, fontsize=7.5, color=INK)
            ax.set_xticks(range(len(cl)), [_short(k) for k in cl], rotation=60, fontsize=7, color=INK2)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if abs(M[i, j]) >= STRONG_RHO:
                        ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=6,
                                color="#ffffff" if abs(M[i, j]) > 0.6 else INK)
            ax.set_title(MAIN5[t], fontsize=10, color=INK, loc="left")
            for s_ in ax.spines.values():
                s_.set_visible(False)
        cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
        cb.set_label("Spearman ρ (labelled where |ρ| ≥ 0.3)", fontsize=8, color=INK2)
        cb.ax.tick_params(labelsize=7, colors=INK2)
        fig.suptitle("Input–target rank correlation per training climate", fontsize=10, color=INK, x=0.01, ha="left")
        files.append("fig3_spearman_heatmap.png")
        fig.savefig(out / files[-1], dpi=150, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)

    # 4. hour of steady-state peaks
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), facecolor=SURFACE, sharey=True)
    for ax, kind, label in zip(axes, ("heat", "cool"), ("Heating", "Cooling")):
        PR._style(ax)
        for h in (7, 8, 19, 23):
            ax.axvspan(h - 0.5, h + 0.5, color=GRID, zorder=0)
        hrs = d[f"peak_{kind}_steady_at"].dropna().astype(str).str[-2:].astype(int).value_counts()
        ax.bar(hrs.index, hrs.values, width=0.7, color=DOT, zorder=3)
        ax.set_xticks(range(1, 25, 2))
        ax.set_xlim(0.5, 24.5)
        ax.set_xlabel("hour ending (shaded = excluded transition hours)", fontsize=8)
        ax.set_title(f"{label}: hour of steady-state peak (runs)", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    files.append("fig4_peak_hours.png")
    fig.savefig(out / files[-1], dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return files


# ---------------------------------------------------------------------------

def build_report(sets, out_dir: Path, partial: bool = False, with_figures: bool = True) -> tuple[str, list]:
    D = load(sets, partial)
    checks = [c(D) for c in CHECKS]
    figs = figures(D, out_dir / "campaign_figures") if with_figures else []
    g = gate(checks)
    src = ", ".join(f"`{Path(s).name}` → `{Path(b).name}/`" for _, s, b, ran in D["batches"] if ran)
    head = [f"# Campaign QA report (Sub-task 2.6){' — PARTIAL' if partial else ''}", "",
            f"Generated {datetime.now():%Y-%m-%d %H:%M} by `runner/campaign_report.py` from {src}. "
            f"Climates (by HDD18): {', '.join(D['order'])}.", "",
            f"**Gate 2.6 → 2.7: {g}**" + (" (partial run — gate not applicable)" if partial else ""), "",
            "| check | status |", "|---|---|"]
    head += [f"| {title} | **{status}** |" for status, title, _ in checks]
    body = []
    for status, title, text in checks:
        body += ["", f"## {title} — {status}", "", text]
    if figs:
        body += ["", "## Figures", ""] + [f"![{f}](campaign_figures/{f})" for f in figs]
    return "\n".join(head + body) + "\n", checks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", nargs=3, action="append", metavar=("NAME", "SAMPLES", "BATCH"),
                    help="a (name, samples csv, batch dir) triple; repeatable. Default: main + holdout")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs", help="folder for the report and figures")
    ap.add_argument("--partial", action="store_true", help="only rows already attempted (smoke / mid-run)")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    sets = [(n, Path(s), Path(b)) for n, s, b in a.set] if a.set else DEFAULT_SETS
    text, checks = build_report(sets, a.out, a.partial, not a.no_figures)
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "campaign_report.md"
    path.write_text(text)
    print(f"wrote {path}")
    for status, title, _ in checks:
        print(f"  {status:5} {title}")
    print(f"Gate: {gate(checks)}")
    return 0 if all(s != "FAIL" for s, _, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
