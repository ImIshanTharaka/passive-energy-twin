#!/usr/bin/env python3
"""
oat.py - One-at-a-time (OAT) parameter response test (Sub-task 2.5, step 7.4).

PURPOSE
-------
The pilot's random-design screen (pilot_report check D) can only confirm STRONG
effects: with 42 designs and 21 parameters varying together, |rho| < ~0.3 is noise.
This test proves that EVERY sampled parameter reaches EnergyPlus and moves the
results in the physically expected direction, by changing one parameter at a time.

DESIGN
------
  baseline   every sampled parameter at the middle of its range (n_floors = 3)
  P_lo/P_hi  parameter P at its minimum / maximum, all others at baseline
             (orientation: 0 vs 90 deg - 0 vs 180 gives a near-mirror building)
  climates   colombo (cooling-dominated), minneapolis (heating-dominated)
  -> (1 + 2 x 21) designs x 2 climates = 86 runs

CHECKS (report)
---------------
  R  responds   (hi - lo) changes at least one target by > RESPOND_PCT % of baseline
  E  expected   sign of (hi - lo) matches physics for the listed (target, climate) pairs

USAGE (from the repository root)
-----
  python runner/oat.py --build                         -> sampling/samples_oat.csv
  python runner/batch_runner.py --samples sampling/samples_oat.csv --workers 8
  python runner/oat.py --report [--batch outputs/oat]  -> outputs/oat/oat_report.md
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sampling.transform import load_design_space        # noqa: E402  (no scipy needed)

ID_COLUMNS = ("run_id", "batch", "design_set", "design_id", "climate_key")   # = sampler.ID_COLUMNS
CLIMATES = ("colombo", "minneapolis")
BATCH = "oat"
SAMPLES = ROOT / "sampling" / "samples_oat.csv"
LEVEL_OVERRIDE = {"orientation": (0.0, 90.0)}
RESPOND_PCT = 0.1

# Expected sign of (value at max - value at min), per (target column, climate).
_C, _M = "colombo", "minneapolis"
EXPECT: dict[str, list[tuple[str, str, int]]] = {
    "wall_r_value":     [("heat_sens_kwh_m2", _M, -1)],
    "roof_r_value":     [("heat_sens_kwh_m2", _M, -1)],
    "floor_r_value":    [("heat_sens_kwh_m2", _M, -1),     # less edge loss to frozen ground
                         ("cool_sens_kwh_m2", _C, -1)],    # less heat from 27-29 C ground
    "infiltration_ach": [("heat_sens_kwh_m2", _M, +1)],
    "glazing_u":        [("heat_sens_kwh_m2", _M, +1)],
    "glazing_shgc":     [("cool_sens_kwh_m2", _C, +1), ("cool_sens_kwh_m2", _M, +1),
                         ("heat_sens_kwh_m2", _M, -1)],
    "wwr_north":        [("zone_cool_sens_north_kwh_m2", _C, +1), ("zone_heat_sens_north_kwh_m2", _M, +1)],
    "wwr_east":         [("zone_cool_sens_east_kwh_m2", _C, +1)],
    "wwr_south":        [("zone_cool_sens_south_kwh_m2", _C, +1)],
    "wwr_west":         [("zone_cool_sens_west_kwh_m2", _C, +1)],
    "overhang_depth_n": [("zone_cool_sens_north_kwh_m2", _C, -1)],
    "overhang_depth_e": [("zone_cool_sens_east_kwh_m2", _C, -1)],
    "overhang_depth_s": [("zone_cool_sens_south_kwh_m2", _C, -1)],
    "overhang_depth_w": [("zone_cool_sens_west_kwh_m2", _C, -1)],
    "sidefin_depth":    [("cool_sens_kwh_m2", _C, -1)],
}
REPORT_TARGETS = ["heat_sens_kwh_m2", "cool_sens_kwh_m2", "eui_site_kwh_m2", "peak_heat_steady_w_m2",
                  "peak_cool_steady_w_m2"] + [f"zone_{m}_sens_{g}_kwh_m2" for m in ("cool", "heat")
                                               for g in ("core", "south", "east", "north", "west")]


def designs(space=None) -> list[tuple[str, str, dict]]:
    """[(design_id, varied parameter or '', full idf parameter dict)]."""
    space = space or load_design_space()
    base = space.midpoint()
    out = [("BASE", "", space.from_values(base))]
    for p in space.params:
        lo, hi = LEVEL_OVERRIDE.get(p.name, (p.lo, p.hi))
        for tag, v in (("lo", lo), ("hi", hi)):
            vals = dict(base)
            vals[p.name] = v
            out.append((f"{p.name}__{tag}", p.name, space.from_values(vals)))
    return out


def _fmt(space, name, value) -> str:
    src = dict(space.derived).get(name, name)
    p = space.spec(src)
    return str(int(value)) if p.is_integer else f"{float(value):.{p.decimals}f}"


def build(path: Path = SAMPLES) -> int:
    space = load_design_space()
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(list(ID_COLUMNS) + space.idf_names)
    n = 0
    for climate in CLIMATES:
        for did, _, params in designs(space):
            w.writerow([f"{BATCH}_{did}_{climate}", BATCH, "oat", did, climate]
                       + [_fmt(space, k, params[k]) for k in space.idf_names])
            n += 1
    path.write_text(buf.getvalue())
    return n


def analyse(batch: Path) -> tuple[list[dict], dict]:
    """Per parameter: % change (hi - lo) / baseline for each target and climate, and checks."""
    from runner import output_parser
    log = batch / "run_log.csv"
    res = batch / "results.csv"
    if not res.exists() or res.stat().st_mtime < log.stat().st_mtime:
        output_parser.parse_batch(batch)
    with res.open(newline="") as fh:
        r = {row["run_id"]: row for row in csv.DictReader(fh)}

    def val(did, climate, col):
        return float(r[f"{BATCH}_{did}_{climate}"][col])

    space = load_design_space()
    rows, summary = [], {"n_params": 0, "responds": 0, "expect_total": 0, "expect_ok": 0, "missing": []}
    for p in space.params:
        summary["n_params"] += 1
        try:
            row = {"param": p.name}
            biggest = 0.0
            for climate in CLIMATES:
                for col in REPORT_TARGETS:
                    base = val("BASE", climate, col)
                    d = val(f"{p.name}__hi", climate, col) - val(f"{p.name}__lo", climate, col)
                    pct = 100.0 * d / max(abs(base), 0.05)
                    row[(col, climate)] = pct
                    if col in ("heat_sens_kwh_m2", "cool_sens_kwh_m2", "eui_site_kwh_m2") \
                            or col.startswith("zone_cool"):
                        biggest = max(biggest, abs(pct))
            row["max_abs_pct"] = biggest
            row["responds"] = biggest > RESPOND_PCT
            summary["responds"] += row["responds"]
            row["expect"] = []
            for col, climate, sign in EXPECT.get(p.name, []):
                pct = row[(col, climate)]
                ok = pct * sign > 0
                row["expect"].append((col, climate, sign, pct, ok))
                summary["expect_total"] += 1
                summary["expect_ok"] += ok
            rows.append(row)
        except KeyError as e:
            summary["missing"].append(f"{p.name}: {e}")
    return rows, summary


def report(batch: Path) -> str:
    rows, s = analyse(batch)
    status = "PASS" if (s["responds"] == s["n_params"] and s["expect_ok"] == s["expect_total"]
                        and not s["missing"]) else "FAIL"
    short = {"heat_sens_kwh_m2": "heat", "cool_sens_kwh_m2": "cool", "eui_site_kwh_m2": "EUI"}
    lines = [f"# One-at-a-time parameter response test — {status}", "",
             f"Baseline = every parameter at mid-range (n_floors 3); each row changes ONE parameter "
             f"min → max (orientation 0 → 90°). Values: (max − min) / baseline, %.", "",
             f"- Parameters that respond (> {RESPOND_PCT}% on heating, cooling, EUI or a zone cooling): "
             f"**{s['responds']}/{s['n_params']}**",
             f"- Expected directions confirmed: **{s['expect_ok']}/{s['expect_total']}**"]
    if s["missing"]:
        lines.append(f"- Missing runs: {s['missing']}")
    lines += ["", "| parameter | heat Minneapolis | cool Colombo | cool Minneapolis | EUI Colombo | "
              "EUI Minneapolis | max | expected direction |", "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        exp = "; ".join(f"{'✅' if ok else '❌'} {short.get(c, c.replace('_kwh_m2', ''))} {cl[:4]} "
                        f"{'↓' if sg < 0 else '↑'} ({pct:+.1f}%)" for c, cl, sg, pct, ok in row["expect"])
        lines.append(
            f"| {row['param']} | {row[('heat_sens_kwh_m2', 'minneapolis')]:+.1f} | "
            f"{row[('cool_sens_kwh_m2', 'colombo')]:+.1f} | {row[('cool_sens_kwh_m2', 'minneapolis')]:+.1f} | "
            f"{row[('eui_site_kwh_m2', 'colombo')]:+.1f} | {row[('eui_site_kwh_m2', 'minneapolis')]:+.1f} | "
            f"{row['max_abs_pct']:.1f}{'' if row['responds'] else ' ❌'} | {exp or '— (responds only)'} |")
    text = "\n".join(lines) + "\n"
    (batch / "oat_report.md").write_text(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true")
    g.add_argument("--report", action="store_true")
    ap.add_argument("--batch", type=Path, default=ROOT / "outputs" / BATCH)
    a = ap.parse_args(argv)
    if a.build:
        n = build()
        print(f"wrote {SAMPLES} ({n} runs)")
        return 0
    text = report(a.batch)
    print(text)
    return 0 if "— PASS" in text.splitlines()[0] else 1


if __name__ == "__main__":
    sys.exit(main())
