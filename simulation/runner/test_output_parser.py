"""
test_output_parser.py - Checks for runner/output_parser.py (Sub-task 2.5.3).

Uses the REAL EnergyPlus 24.1 SQL from the smoke run outputs/smoke_orient_000
(3 floors, 15,000 m2). Failure cases are made by damaging a temporary COPY of
it (the original is never touched - also checked).

Run:  python runner/test_output_parser.py
"""

from __future__ import annotations

import csv
import hashlib
import math
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner import output_parser as O      # noqa: E402

REAL = ROOT / "outputs" / "smoke_orient_000" / "eplusout.sql"
REAL_90 = ROOT / "outputs" / "smoke_orient_090" / "eplusout.sql"
_TMP = tempfile.TemporaryDirectory()


def copy(name="x.sql") -> Path:
    p = Path(_TMP.name) / name
    shutil.copy(REAL, p)
    return p


def damage(sql: str, *params) -> Path:
    p = copy(f"d{abs(hash(sql))}.sql")
    con = sqlite3.connect(p)
    con.execute(sql, params)
    con.commit()
    con.close()
    return p


def expect_error(path: Path, fragment: str):
    try:
        O.parse_sql(path)
    except O.OutputError as e:
        assert fragment in str(e), f"error {e!s} lacks {fragment!r}"
        return
    raise AssertionError(f"no OutputError (expected {fragment!r})")


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------

def check_real_run_values():
    """Pinned values from the smoke run (cross-checked by hand against the ABUPS tables)."""
    r = O.parse_sql(REAL)
    pinned = {"cond_area_m2": 15000.0, "heat_meter_kwh_m2": 5.083, "cool_meter_kwh_m2": 52.975,
              "heat_sens_kwh_m2": 5.264, "heat_total_kwh_m2": 6.785, "cool_sens_kwh_m2": 53.036,
              "cool_total_kwh_m2": 60.652, "lights_kwh_m2": 20.298, "equip_kwh_m2": 28.571,
              "sys_heat_kwh_m2": 21.960, "sys_cool_kwh_m2": 29.951, "eui_site_kwh_m2": 100.781}
    for k, v in pinned.items():
        assert abs(r[k] - v) < 1e-3, f"{k}: {r[k]} != {v}"
    assert (r["n_zones"], r["n_floors"], r["ep_version"]) == (15, 3, "24.1.0")
    assert O.consistency_problems(r) == []


def check_all_columns_present():
    r = O.parse_sql(REAL)
    assert set(O.COLUMNS) - {"run_id"} == set(r), set(O.COLUMNS) ^ (set(r) | {"run_id"})


def check_zone_groups_add_up_to_building():
    """Area-weighted zone-group cooling must equal the building total exactly."""
    r = O.parse_sql(REAL)
    con = sqlite3.connect(REAL)
    areas = {g: 0.0 for g in O.ZONE_GROUPS}
    for name, a, m in con.execute("SELECT ZoneName, FloorArea, Multiplier FROM Zones"):
        areas[name.split("_")[-1].lower()] += a * m
    for prefix, building in (("zone_cool_sens", "cool_sens_kwh_m2"), ("zone_heat_sens", "heat_sens_kwh_m2")):
        total = sum(r[f"{prefix}_{g}_kwh_m2"] * areas[g] for g in O.ZONE_GROUPS)
        assert abs(total / sum(areas.values()) / r[building] - 1) < 1e-12, prefix


def check_physics_of_smoke_run():
    r = O.parse_sql(REAL)
    assert r["cool_total_kwh_m2"] >= r["cool_sens_kwh_m2"] and 0 < r["cool_latent_frac"] < 0.5
    assert r["zone_cool_sens_core_kwh_m2"] < r["zone_cool_sens_west_kwh_m2"], "core has no solar"
    assert r["zone_cool_sens_north_kwh_m2"] < r["zone_cool_sens_south_kwh_m2"]
    assert r["zone_heat_sens_north_kwh_m2"] > r["zone_heat_sens_south_kwh_m2"], "north loses more, south gets sun"
    assert 50 < r["eui_site_kwh_m2"] < 300
    assert r["unmet_heat_h"] == 0 and r["unmet_cool_h"] == 0


def check_orientation_changes_facade_split_not_totals():
    a, b = O.parse_sql(REAL), O.parse_sql(REAL_90)
    assert abs(a["lights_kwh_m2"] - b["lights_kwh_m2"]) < 1e-9, "internal loads are orientation-free"
    assert abs(a["zone_cool_sens_south_kwh_m2"] - b["zone_cool_sens_south_kwh_m2"]) > 1.0
    assert abs(a["cool_sens_kwh_m2"] - b["cool_sens_kwh_m2"]) / a["cool_sens_kwh_m2"] < 0.05


def check_hourly_peaks_nan_without_hourly_meters():
    r = O.parse_sql(REAL)
    for k in ("peak_heat_hourly_w_m2", "peak_cool_hourly_w_m2", "peak_heat_steady_w_m2",
              "peak_cool_steady_w_m2"):
        assert math.isnan(r[k]), k
    assert r["peak_heat_hourly_at"] == "" and r["peak_cool_steady_at"] == ""
    assert r["peak_heat_ts_w_m2"] > 0 and r["peak_cool_ts_w_m2"] > 0


DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def add_hourly_meter(p: Path, meter: str, values: list[float], dict_index=90001):
    """Insert a synthetic Hourly meter (365 x 24 values; day 1 = Monday) into a SQL copy."""
    con = sqlite3.connect(p)
    con.execute("INSERT INTO ReportDataDictionary VALUES (?,1,'Sum','Facility','Zone',NULL,?,'Hourly',NULL,'J')",
                (dict_index, meter))
    for i, v in enumerate(values):
        doy, hour = divmod(i, 24)
        month, day = 1 + doy // 31, 1 + doy % 31          # labels only; need not be a real calendar
        ti = 100000 + dict_index * 10000 + i
        con.execute("INSERT OR IGNORE INTO Time (TimeIndex, Month, Day, Hour, Minute, DayType, WarmupFlag) "
                    "VALUES (?,?,?,?,0,?,0)", (ti, month, day, hour + 1, DAYS[doy % 7]))
        con.execute("INSERT INTO ReportData (TimeIndex, ReportDataDictionaryIndex, Value) VALUES (?,?,?)",
                    (ti, dict_index, v))
    con.commit()
    con.close()


def annual_j(meter: str) -> float:
    con = sqlite3.connect(REAL)
    v = con.execute("SELECT SUM(r.Value) FROM ReportData r JOIN ReportDataDictionary d ON "
                    "r.ReportDataDictionaryIndex = d.ReportDataDictionaryIndex WHERE d.Name=? "
                    "AND d.ReportingFrequency='Run Period'", (meter,)).fetchone()[0]
    con.close()
    return v


def spiky_year(total: float) -> list[float]:
    """Shares of the annual total placed in single hours (index = day*24 + hour_ending-1).
    Excluded (transition): Mon 07 0.30, Mon 08 0.20, Fri 23 0.12, Sat 19 0.08.
    Kept: Sun 07 0.18 (no setpoint change on Sunday) and Mon 10 0.12."""
    v = [0.0] * 8760
    for day, hour_ending, share in ((0, 7, 0.30), (0, 8, 0.20), (4, 23, 0.12), (5, 19, 0.08),
                                    (6, 7, 0.18), (0, 10, 0.12)):
        v[day * 24 + hour_ending - 1] = share * total
    return v


def check_hourly_and_steady_peaks():
    H = annual_j("Heating:EnergyTransfer")
    p = copy("hourly.sql")
    add_hourly_meter(p, "Heating:EnergyTransfer", spiky_year(H))
    r = O.parse_sql(p)
    area = sum(a * m for a, m in sqlite3.connect(REAL).execute("SELECT FloorArea, Multiplier FROM Zones"))
    assert abs(r["peak_heat_hourly_w_m2"] / (0.30 * H / 3600 / area) - 1) < 1e-9
    assert r["peak_heat_hourly_at"] == "01-01 07", r["peak_heat_hourly_at"]
    assert abs(r["peak_heat_steady_w_m2"] / (0.18 * H / 3600 / area) - 1) < 1e-9, \
        "Mon 07/08, Fri 23, Sat 19 must be excluded; Sunday 06-07 must NOT be"
    assert r["peak_heat_steady_at"] == "01-07 07", r["peak_heat_steady_at"]
    assert math.isnan(r["peak_cool_hourly_w_m2"]) and r["peak_cool_steady_at"] == ""
    assert abs(r["heat_meter_kwh_m2"] - 5.083) < 1e-3, "hourly rows must not leak into annual sums"


def check_transition_rule():
    T = O.is_transition_hour
    for d in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday"):
        assert T(7, d) and T(8, d) and T(23, d), d
        assert not T(6, d) and not T(19, d) and not T(22, d), d
        assert T(9, d) == (d == "Monday"), "3rd recovery hour only after the full-setback Sunday"
    assert T(7, "Saturday") and T(8, "Saturday") and T(19, "Saturday")
    assert not T(23, "Saturday") and not T(18, "Saturday")
    for h in range(1, 25):
        assert not T(h, "Sunday") and not T(h, "Holiday"), h


def check_transition_hours_match_schedules():
    """The exclusion must follow the generator's setpoint schedules (step at HH:00 ->
    exclude the hour(s) ending HH+1 [and HH+2 for a morning step-up])."""
    from idf_generator.internal_loads import internal_loads as IL
    for sched in (IL._HEATING_SETPOINT_SCHEDULE, IL._COOLING_SETPOINT_SCHEDULE):
        for pat in sched.day_patterns:
            steps = [iv.until_time for iv in pat.intervals[:-1]]          # times the value changes
            if pat.day_types == IL._WEEKDAYS_AND_DESIGN:
                days, expect = ("Tuesday",), {7, 8, 23}
            elif pat.day_types == IL._SATURDAY:
                days, expect = ("Saturday",), {7, 8, 19}
            else:
                assert steps == [], f"{sched.name}: unexpected transition on {pat.day_types}"
                continue
            up, down = (int(x[:2]) for x in steps)
            assert expect == {up + 1, up + 2, down + 1}, (sched.name, pat.day_types, steps)
            for d in days:
                assert O.TRANSITION_HOURS[d] == expect
    # Monday = weekday pattern + one extra recovery hour, because the day before
    # (Sunday, _ALL_OTHER_DAYS) has no occupied period at all
    sunday = [p for p in IL._HEATING_SETPOINT_SCHEDULE.day_patterns if p.day_types == IL._ALL_OTHER_DAYS][0]
    assert len(sunday.intervals) == 1, "Sunday must be setback all day for the Monday rule"
    assert O.TRANSITION_HOURS["Monday"] == O.TRANSITION_HOURS["Tuesday"] | {9}


def check_targets_map_to_real_columns():
    assert set(O.TARGETS.values()) <= set(O.COLUMNS)
    assert not any(c.startswith("zone_heat") for c in O.TARGETS.values()), "zone heating is diagnostic only"
    assert O.TARGETS["heating_demand_kwh_m2"] == "heat_sens_kwh_m2"        # T1
    assert O.TARGETS["peak_cooling_w_m2"] == "peak_cool_steady_w_m2"       # T4


def check_hourly_must_sum_to_annual():
    H = annual_j("Heating:EnergyTransfer")
    p = copy("hourly_bad_sum.sql")
    add_hourly_meter(p, "Heating:EnergyTransfer", [x * 1.01 for x in spiky_year(H)])
    expect_error(p, "sums to")


def check_hourly_must_cover_the_year():
    H = annual_j("Heating:EnergyTransfer")
    p = copy("hourly_short.sql")
    add_hourly_meter(p, "Heating:EnergyTransfer", spiky_year(H)[:8000])
    expect_error(p, "expected 8760")


def check_missing_zone_output_detected():
    p = damage("DELETE FROM ReportDataDictionary WHERE KeyValue='ZONE_F2_WEST IDEAL LOADS AIR SYSTEM' "
               "AND Name='Zone Ideal Loads Zone Sensible Cooling Energy'")
    expect_error(p, "14 of 15 zones")


def check_missing_meter_detected():
    p = damage("DELETE FROM ReportDataDictionary WHERE Name='Heating:EnergyTransfer'")
    expect_error(p, "Heating:EnergyTransfer")


def check_wrong_units_detected():
    p = damage("UPDATE ReportDataDictionary SET Units='kWh' WHERE Name='Cooling:EnergyTransfer'")
    expect_error(p, "unexpected units")


def check_unexpected_zone_name_detected():
    p = damage("UPDATE Zones SET ZoneName='PLENUM_1' WHERE ZoneName='ZONE_F1_CORE'")
    expect_error(p, "unexpected zone name")


def check_broken_layout_detected():
    p = damage("DELETE FROM Zones WHERE ZoneName='ZONE_F2_EAST'")
    expect_error(p, "zone layout")


def check_area_mismatch_detected():
    p = damage("UPDATE Zones SET FloorArea = FloorArea * 1.1 WHERE ZoneName='ZONE_F0_CORE'")
    expect_error(p, "ABUPS conditioned area")


def check_missing_or_corrupt_file():
    expect_error(Path(_TMP.name) / "nope.sql", "no such file")
    bad = Path(_TMP.name) / "bad.sql"
    bad.write_bytes(b"this is not a database" * 50)
    expect_error(bad, "SQLite error")


def check_original_sql_never_modified():
    before = md5(REAL)
    O.parse_sql(REAL)
    assert md5(REAL) == before


def check_consistency_rules_fire():
    r = O.parse_sql(REAL)
    assert O.consistency_problems({**r, "cool_total_kwh_m2": r["cool_sens_kwh_m2"] - 1})
    assert O.consistency_problems({**r, "eui_site_kwh_m2": r["eui_site_kwh_m2"] + 5})
    assert O.consistency_problems({**r, "elec_kwh_m2": r["elec_kwh_m2"] + 5})
    assert O.consistency_problems({**r, "heat_sens_kwh_m2": -1.0})


def check_batch_parses_only_ok_runs():
    b = Path(_TMP.name) / "batch"
    b.mkdir()
    for rid in ("r_ok", "r_ok2", "r_broken", "r_failed"):
        (b / rid).mkdir()
    shutil.copy(REAL, b / "r_ok" / "eplusout.sql")
    shutil.copy(REAL_90, b / "r_ok2" / "eplusout.sql")
    (b / "r_broken" / "eplusout.sql").write_bytes(b"x" * 100)
    with (b / "run_log.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "status"])
        w.writerows([["r_ok", "ok"], ["r_ok2", "failed"], ["r_ok2", "ok"],     # rerun: latest wins
                     ["r_broken", "ok"], ["r_failed", "failed"]])
    results, errors = O.parse_batch(b)
    assert [r["run_id"] for r in results] == ["r_ok", "r_ok2"]
    assert [e["run_id"] for e in errors] == ["r_broken"]
    with (b / "results.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == O.COLUMNS and len(rows) == 2
    assert rows[0]["peak_heat_hourly_w_m2"] == "", "NaN written as empty"
    assert abs(float(rows[0]["eui_site_kwh_m2"]) - 100.781) < 1e-3


ALL_CHECKS = [
    check_real_run_values,
    check_all_columns_present,
    check_zone_groups_add_up_to_building,
    check_physics_of_smoke_run,
    check_orientation_changes_facade_split_not_totals,
    check_hourly_peaks_nan_without_hourly_meters,
    check_hourly_and_steady_peaks,
    check_transition_rule,
    check_transition_hours_match_schedules,
    check_targets_map_to_real_columns,
    check_hourly_must_sum_to_annual,
    check_hourly_must_cover_the_year,
    check_missing_zone_output_detected,
    check_missing_meter_detected,
    check_wrong_units_detected,
    check_unexpected_zone_name_detected,
    check_broken_layout_detected,
    check_area_mismatch_detected,
    check_missing_or_corrupt_file,
    check_original_sql_never_modified,
    check_consistency_rules_fire,
    check_batch_parses_only_ok_runs,
]


def main():
    total = failed = 0
    print("=" * 72)
    print("OUTPUT PARSER (2.5.3) - CHECKS")
    print("=" * 72)
    if not REAL.exists():
        print(f"  ERROR smoke-run SQL not found: {REAL}")
        print("RESULTS: 0/1 passed, 1 failed")
        sys.exit(1)
    for chk in ALL_CHECKS:
        total += 1
        try:
            chk()
            print(f"  PASS  {chk.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {chk.__name__}: {e}")
        except Exception as e:                       # noqa: BLE001
            failed += 1
            print(f"  ERROR {chk.__name__}: {type(e).__name__}: {e}")
    print("=" * 72)
    print(f"RESULTS: {total - failed}/{total} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
