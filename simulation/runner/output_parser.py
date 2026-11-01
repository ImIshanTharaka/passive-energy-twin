#!/usr/bin/env python3
"""
output_parser.py - Extract the ML targets from an EnergyPlus eplusout.sql (Sub-task 2.5.3).

Extracts every candidate definition side by side. The CHOSEN targets (decisions
T1 and T4, closed 8 Oct 2026 on the 2.5 smoke-run evidence) are listed in
TARGETS; the other columns are kept as diagnostics for the thesis.

    T1  demand      = Zone Ideal Loads ZONE SENSIBLE heating/cooling.
                      "Total" adds moisture carried in by un-dehumidified outdoor air
                      (Colombo: total heating 10.3 vs sensible 0.76 kWh/m2) - not a
                      heating/cooling energy, and humidity control is out of scope.
    T4  peak load   = STEADY-STATE hourly peak: the highest hour of the year excluding
                      the setpoint-transition hours (TRANSITION_HOURS). Smoke run: the
                      06:00 recovery lasts ~2 h (Minneapolis 32.8 / 19.4 W/m2 at hours
                      ending 07 / 08 vs ~5-10 otherwise; all top-50 heating hours end at
                      07:00), and the hour after an evening setback shows spurious
                      "heating" from unconditioned outdoor air in hot climates.

    T1  demand definition   (a) Zone Ideal Loads ZONE sensible / total energy   (Decision A: zone load)
                            (b) Heating:/Cooling:EnergyTransfer meters
    T4  peak definition     ts      highest 10-min (timestep) rate, SensibleHeatGainSummary -
                                    dominated by 06:00 setback-recovery spikes (diagnostic only)
                            hourly  highest hour of the year, hourly Heating:/Cooling:EnergyTransfer
                            steady  highest hour EXCLUDING the setpoint-transition hours
                                    (TRANSITION_HOURS)  <- the chosen peak target
                            hourly/steady need the hourly meters (generator outputs.py, T4);
                            older IDFs without them give NaN. The pilot picks the target.

COLUMNS (all intensities per m2 of conditioned floor area, from the SQL Zones table)
-------
  cond_area_m2, n_zones, n_floors
  heat_meter_kwh_m2, cool_meter_kwh_m2                          (b)
  heat_sens_kwh_m2, heat_total_kwh_m2, cool_sens_kwh_m2, cool_total_kwh_m2   (a)
  cool_latent_frac              (cool_total - cool_sens) / cool_total
  lights_kwh_m2, equip_kwh_m2, elec_kwh_m2
  sys_heat_kwh_m2, sys_cool_kwh_m2  ABUPS End Uses "District Heating Water"/"District Cooling": the
                                ideal-loads SYSTEM energy, i.e. zone load PLUS outdoor-air (ventilation)
                                conditioning. Differs from the zone-load targets by design (Decision A).
  eui_site_kwh_m2               ABUPS Total Site Energy = elec + sys_heat + sys_cool (checked)
  peak_heat_ts_w_m2, peak_cool_ts_w_m2          timestep peaks, Total Facility
  peak_{heat,cool}_hourly_w_m2, _hourly_at   hourly peak and when ("MM-DD HH" = hour ENDING at HH:00)
  peak_{heat,cool}_steady_w_m2, _steady_at   same, recovery hour(s) excluded
  zone_cool_sens_<core|south|east|north|west>_kwh_m2   sensible cooling per m2 of that zone group,
                                summed over floors. Façade names are the BUILDING's own sides
                                (before the 'orientation' rotation).
  zone_heat_sens_<group>_kwh_m2 same for sensible heating - DIAGNOSTIC (not an ML target): which
                                facade drives heating in cold climates; web-tool facade view.
  unmet_heat_h, unmet_cool_h    SystemSummary "Time Setpoint Not Met", Facility (QA, expected ~0)
  ep_version

USAGE (from the repository root)
-----
  python runner/output_parser.py outputs/smoke_orient_000/eplusout.sql     # one run, printed
  python runner/output_parser.py outputs/pilot                             # batch -> outputs/pilot/results.csv
      (only runs logged "ok" in run_log.csv; failures go to results_errors.csv)
"""

from __future__ import annotations

import csv
import math
import re
import sqlite3
import sys
from pathlib import Path

J_PER_KWH = 3.6e6
ZONE_GROUPS = ("core", "south", "east", "north", "west")
_ZONE_RE = re.compile(r"^ZONE_F(\d+)_(CORE|SOUTH|EAST|NORTH|WEST)$")
AREA_TOL = 0.005                       # SQL zone area vs ABUPS conditioned area
HOURLY_SUM_TOL = 0.001                 # sum of 8,760 hourly values vs annual meter

# Setpoint-transition hours excluded from the steady-state peak (T4). EnergyPlus
# labels an hourly value by the hour it ENDS (Hour 7 = 06:00-07:00). Setpoints
# (internal_loads.py, DOE schedules):
#   Mon-Fri  setback -> occupied at 06:00, occupied -> setback at 22:00
#   Sat      setback -> occupied at 06:00, occupied -> setback at 18:00
#   Sun      setback all day (no transition)
# Excluded: the 2 hours after the morning step-up (recovery), and the 1 hour
# after the evening step-down. MONDAY gets a 3rd recovery hour (09): it follows
# 36 h of setback (Sat 18:00 -> Mon 06:00) instead of 8 h, and the re-run pilot
# (F-factor slab) showed a decaying heating tail 07 > 08 > 09 > 10 on Mondays in
# 33 of 35 runs whose steady peak fell at hour 09 (12 Oct 2026). Cooling showed
# no tail (flat 09-11 plateau), so the rule is driven by heating physics.
# Must change together with the schedules (test_output_parser checks this).
_WEEKDAY_TRANSITIONS = frozenset({7, 8, 23})
AFTER_FULL_SETBACK_DAY_EXTRA = frozenset({9})
TRANSITION_HOURS: dict[str, frozenset] = {
    "Monday": _WEEKDAY_TRANSITIONS | AFTER_FULL_SETBACK_DAY_EXTRA,
    **{d: _WEEKDAY_TRANSITIONS for d in ("Tuesday", "Wednesday", "Thursday", "Friday")},
    "Saturday": frozenset({7, 8, 19}),
}

# Chosen ML targets (T1, T4) -> parser column. Used by 2.7 to build the dataset.
TARGETS = {
    "heating_demand_kwh_m2": "heat_sens_kwh_m2",
    "cooling_demand_kwh_m2": "cool_sens_kwh_m2",
    "eui_kwh_m2": "eui_site_kwh_m2",
    "peak_heating_w_m2": "peak_heat_steady_w_m2",
    "peak_cooling_w_m2": "peak_cool_steady_w_m2",
    **{f"zone_cooling_{g}_kwh_m2": f"zone_cool_sens_{g}_kwh_m2"
       for g in ("core", "south", "east", "north", "west")},
}

V_HEAT_METER, V_COOL_METER = "Heating:EnergyTransfer", "Cooling:EnergyTransfer"
V_ELEC, V_LIGHTS, V_EQUIP = "Electricity:Facility", "InteriorLights:Electricity", "InteriorEquipment:Electricity"
V_IL = "Zone Ideal Loads Zone {} {} Energy"          # .format("Sensible"|"Total", "Heating"|"Cooling")

COLUMNS = (["run_id", "ep_version", "cond_area_m2", "n_zones", "n_floors",
            "heat_meter_kwh_m2", "cool_meter_kwh_m2",
            "heat_sens_kwh_m2", "heat_total_kwh_m2", "cool_sens_kwh_m2", "cool_total_kwh_m2",
            "cool_latent_frac", "lights_kwh_m2", "equip_kwh_m2", "elec_kwh_m2",
            "sys_heat_kwh_m2", "sys_cool_kwh_m2", "eui_site_kwh_m2",
            "peak_heat_ts_w_m2", "peak_cool_ts_w_m2",
            "peak_heat_hourly_w_m2", "peak_heat_hourly_at", "peak_cool_hourly_w_m2", "peak_cool_hourly_at",
            "peak_heat_steady_w_m2", "peak_heat_steady_at", "peak_cool_steady_w_m2", "peak_cool_steady_at"]
           + [f"zone_cool_sens_{g}_kwh_m2" for g in ZONE_GROUPS]
           + [f"zone_heat_sens_{g}_kwh_m2" for g in ZONE_GROUPS]
           + ["unmet_heat_h", "unmet_cool_h"])


class OutputError(Exception):
    """The SQL file is missing something the targets need."""


def _connect(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise OutputError(f"no such file: {path}")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)    # never modify the run's output


def _num(s) -> float:
    try:
        return float(str(s).strip())
    except (TypeError, ValueError):
        return math.nan


def _runperiod_sum(con, name: str, *, required=True, by_key=False):
    """Sum of a Run Period variable/meter (J). by_key -> {KeyValue: J}."""
    rows = con.execute(
        "SELECT d.KeyValue, d.Units, SUM(r.Value) FROM ReportData r JOIN ReportDataDictionary d "
        "ON r.ReportDataDictionaryIndex = d.ReportDataDictionaryIndex "
        "WHERE d.Name = ? AND d.ReportingFrequency = 'Run Period' GROUP BY d.KeyValue, d.Units",
        (name,)).fetchall()
    if not rows:
        if required:
            raise OutputError(f"output missing from SQL: '{name}' (Run Period)")
        return {} if by_key else math.nan
    if any(u != "J" for _, u, _ in rows):
        raise OutputError(f"'{name}': unexpected units {sorted({u for _, u, _ in rows})}")
    return {k: v for k, _, v in rows} if by_key else sum(v for _, _, v in rows)


def _hourly_series(con, name: str) -> list[tuple[int, int, int, str, float]]:
    """(month, day, hour_ending, day_type, J) for an Hourly meter; [] if not requested."""
    return con.execute(
        "SELECT t.Month, t.Day, t.Hour, t.DayType, r.Value FROM ReportData r "
        "JOIN ReportDataDictionary d ON r.ReportDataDictionaryIndex = d.ReportDataDictionaryIndex "
        "JOIN Time t ON r.TimeIndex = t.TimeIndex "
        "WHERE d.Name = ? AND d.ReportingFrequency = 'Hourly' AND COALESCE(t.WarmupFlag, 0) = 0 "
        "ORDER BY r.TimeIndex", (name,)).fetchall()


def is_transition_hour(hour_ending: int, day_type: str) -> bool:
    return hour_ending in TRANSITION_HOURS.get(day_type, ())


def hourly_peaks(series, area: float) -> dict:
    """{'hourly': (W/m2, 'MM-DD HH'), 'steady': (...)}; NaN/'' when the series is empty."""
    def best(rows):
        if not rows:
            return math.nan, ""
        m, d, h, _, j = max(rows, key=lambda r: r[4])
        return j / 3600.0 / area, f"{m:02d}-{d:02d} {h:02d}"
    return {"hourly": best(series),
            "steady": best([r for r in series if not is_transition_hour(r[2], r[3])])}


def _tab(con, report: str, table: str, row: str, col: str, *, report_for="Entire Facility",
         required=True) -> tuple[float, str]:
    r = con.execute(
        "SELECT Value, Units FROM TabularDataWithStrings WHERE ReportName=? AND ReportForString=? "
        "AND TableName=? AND RowName=? AND ColumnName=?",
        (report, report_for, table, row, col)).fetchone()
    if r is None:
        if required:
            raise OutputError(f"tabular value missing: {report} / {table} / {row} / {col}")
        return math.nan, ""
    return _num(r[0]), r[1]


def _zones(con) -> dict[str, tuple[str, int, float]]:
    """{zone: (group, floor_index, area_m2 incl. multiplier)} - validates naming and layout."""
    out = {}
    for name, area, mult in con.execute("SELECT ZoneName, FloorArea, Multiplier FROM Zones"):
        m = _ZONE_RE.match(name.upper())
        if not m:
            raise OutputError(f"unexpected zone name '{name}' (generator naming changed?)")
        out[name.upper()] = (m.group(2).lower(), int(m.group(1)), float(area) * float(mult))
    if not out:
        raise OutputError("no zones in SQL")
    floors = {f for _, f, _ in out.values()}
    if floors != set(range(len(floors))) or len(out) != 5 * len(floors):
        raise OutputError(f"zone layout is not 5 zones x {len(floors)} floors ({len(out)} zones)")
    return out


def parse_sql(path: str | Path) -> dict:
    """All target candidates for one run, as a flat dict (keys = COLUMNS minus run_id)."""
    path = Path(path)
    con = _connect(path)
    try:
        zones = _zones(con)
        area = sum(a for _, _, a in zones.values())
        abups_area, _ = _tab(con, "AnnualBuildingUtilityPerformanceSummary", "Building Area",
                             "Net Conditioned Building Area", "Area")
        if abs(abups_area - area) > AREA_TOL * area:
            raise OutputError(f"zone area {area:.1f} m2 != ABUPS conditioned area {abups_area:.1f} m2")
        to_kwh_m2 = lambda j: j / J_PER_KWH / area                                  # noqa: E731

        out: dict = {"cond_area_m2": round(area, 3), "n_zones": len(zones),
                     "n_floors": len({f for _, f, _ in zones.values()})}
        ver = con.execute("SELECT EnergyPlusVersion FROM Simulations").fetchone()
        m = re.search(r"Version\s+([\d.]+)", ver[0] if ver else "")
        out["ep_version"] = m.group(1) if m else ""

        out["heat_meter_kwh_m2"] = to_kwh_m2(_runperiod_sum(con, V_HEAT_METER))
        out["cool_meter_kwh_m2"] = to_kwh_m2(_runperiod_sum(con, V_COOL_METER))
        il = {}
        for kind in ("Sensible", "Total"):
            for mode in ("Heating", "Cooling"):
                per_key = _runperiod_sum(con, V_IL.format(kind, mode), by_key=True)
                if len(per_key) != len(zones):
                    raise OutputError(f"'{V_IL.format(kind, mode)}' for {len(per_key)} "
                                      f"of {len(zones)} zones")
                il[(kind, mode)] = per_key
                out[f"{mode[:4].lower()}_{'sens' if kind == 'Sensible' else 'total'}_kwh_m2"] = \
                    to_kwh_m2(sum(per_key.values()))
        ct, cs = out["cool_total_kwh_m2"], out["cool_sens_kwh_m2"]
        out["cool_latent_frac"] = (ct - cs) / ct if ct > 0 else 0.0

        out["lights_kwh_m2"] = to_kwh_m2(_runperiod_sum(con, V_LIGHTS))
        out["equip_kwh_m2"] = to_kwh_m2(_runperiod_sum(con, V_EQUIP))
        out["elec_kwh_m2"] = to_kwh_m2(_runperiod_sum(con, V_ELEC))
        site, unit = _tab(con, "AnnualBuildingUtilityPerformanceSummary", "Site and Source Energy",
                          "Total Site Energy", "Energy Per Conditioned Building Area")
        if unit != "MJ/m2":
            raise OutputError(f"ABUPS site energy in {unit!r}, expected MJ/m2")
        out["eui_site_kwh_m2"] = site / 3.6
        for key, fuel in (("sys_heat_kwh_m2", "District Heating Water"), ("sys_cool_kwh_m2", "District Cooling")):
            gj, unit = _tab(con, "AnnualBuildingUtilityPerformanceSummary", "End Uses",
                            "Total End Uses", fuel)
            if unit != "GJ":
                raise OutputError(f"ABUPS {fuel} in {unit!r}, expected GJ")
            out[key] = gj * 1e9 / J_PER_KWH / area

        for mode, col in (("heat", "Heating"), ("cool", "Cooling")):
            w, _ = _tab(con, "SensibleHeatGainSummary", f"Peak {col} Sensible Heat Gain Components",
                        "Total Facility", f"HVAC Zone Eq & Other Sensible Air {col}")
            out[f"peak_{mode}_ts_w_m2"] = abs(w) / area
            meter = V_HEAT_METER if mode == "heat" else V_COOL_METER
            series = _hourly_series(con, meter)
            if series:
                if len(series) not in (8760, 8784):
                    raise OutputError(f"hourly '{meter}': {len(series)} values, expected 8760")
                annual = out[f"{mode}_meter_kwh_m2"] * J_PER_KWH * area
                hsum = sum(r[4] for r in series)
                if abs(hsum - annual) > HOURLY_SUM_TOL * max(annual, 1.0):
                    raise OutputError(f"hourly '{meter}' sums to {hsum:.4g} J, annual is {annual:.4g} J")
            for kind, (w, at) in hourly_peaks(series, area).items():
                out[f"peak_{mode}_{kind}_w_m2"], out[f"peak_{mode}_{kind}_at"] = w, at

        for mode, prefix in (("Cooling", "zone_cool_sens"), ("Heating", "zone_heat_sens")):
            per_zone = il[("Sensible", mode)]
            for g in ZONE_GROUPS:
                names = [z for z, (grp, _, _) in zones.items() if grp == g]
                g_area = sum(zones[z][2] for z in names)
                g_j = sum(per_zone[f"{z} IDEAL LOADS AIR SYSTEM"] for z in names)
                out[f"{prefix}_{g}_kwh_m2"] = g_j / J_PER_KWH / g_area

        for mode, col in (("heat", "During Heating"), ("cool", "During Cooling")):
            out[f"unmet_{mode}_h"], _ = _tab(con, "SystemSummary", "Time Setpoint Not Met",
                                             "Facility", col, required=False)
        return out
    except sqlite3.Error as e:
        raise OutputError(f"SQLite error: {e}") from e
    finally:
        con.close()


def consistency_problems(r: dict) -> list[str]:
    """Physics/bookkeeping checks that must hold for every valid run."""
    p = []
    for k in COLUMNS:
        v = r.get(k)
        if isinstance(v, float) and not math.isnan(v) and v < -1e-9 and k.endswith(("_m2", "_h")):
            p.append(f"{k} negative ({v})")
    if r["heat_total_kwh_m2"] + 1e-6 < r["heat_sens_kwh_m2"]:
        p.append("total heating < sensible heating")
    if r["cool_total_kwh_m2"] + 1e-6 < r["cool_sens_kwh_m2"]:
        p.append("total cooling < sensible cooling")
    if abs(r["elec_kwh_m2"] - r["lights_kwh_m2"] - r["equip_kwh_m2"]) > 0.01 * max(r["elec_kwh_m2"], 1):
        p.append("electricity != lights + equipment (unexpected electric end use)")
    eui_parts = r["elec_kwh_m2"] + r["sys_heat_kwh_m2"] + r["sys_cool_kwh_m2"]
    if abs(r["eui_site_kwh_m2"] - eui_parts) > max(0.005 * r["eui_site_kwh_m2"], 0.05):
        p.append(f"EUI {r['eui_site_kwh_m2']:.2f} != elec + sys heat + sys cool ({eui_parts:.2f})")
    zone_sum = sum(r[f"zone_cool_sens_{g}_kwh_m2"] for g in ZONE_GROUPS)
    if zone_sum <= 0 and r["cool_sens_kwh_m2"] > 0:
        p.append("zone cooling all zero but building cooling > 0")
    return p


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def _fmt(v):
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.6g}"
    return v


def parse_batch(batch_dir: str | Path) -> tuple[list[dict], list[dict]]:
    """Parse every run logged 'ok' in <batch>/run_log.csv -> results.csv (+ results_errors.csv)."""
    batch_dir = Path(batch_dir)
    log = batch_dir / "run_log.csv"
    if not log.exists():
        raise OutputError(f"{log} not found - run the batch runner first")
    with log.open(newline="") as fh:
        latest = {r["run_id"]: r for r in csv.DictReader(fh)}
    ok_ids = sorted(rid for rid, r in latest.items() if r["status"] == "ok")
    results, errors = [], []
    for rid in ok_ids:
        try:
            r = {"run_id": rid, **parse_sql(batch_dir / rid / "eplusout.sql")}
            probs = consistency_problems(r)
            if probs:
                errors.append({"run_id": rid, "error": "; ".join(probs)})
            results.append(r)
        except OutputError as e:
            errors.append({"run_id": rid, "error": str(e)})
    with (batch_dir / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows({k: _fmt(r.get(k, math.nan)) for k in COLUMNS} for r in results)
    with (batch_dir / "results_errors.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run_id", "error"])
        w.writeheader()
        w.writerows(errors)
    return results, errors


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        print(__doc__)
        return 2
    target = Path(argv[0])
    if target.is_file():
        r = parse_sql(target)
        for k in COLUMNS[1:]:
            v = r[k]
            print(f"  {k:30} {v:12.3f}" if isinstance(v, float) else f"  {k:30} {v!s:>12}")
        probs = consistency_problems(r)
        print("consistency: OK" if not probs else "consistency PROBLEMS:\n  " + "\n  ".join(probs))
        return 0 if not probs else 1
    results, errors = parse_batch(target)
    print(f"{target}: {len(results)} runs parsed -> results.csv;  {len(errors)} problems -> results_errors.csv")
    for e in errors[:20]:
        print(f"  {e['run_id']}: {e['error']}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
