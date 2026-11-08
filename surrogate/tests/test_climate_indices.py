"""
test_climate_indices.py - checks for surrogate/climate_indices.py (Sub-task 3.4 step 1).

The key checks: (1) psychrometric functions against ASHRAE Fundamentals table values;
(2) the occupied-hour mask is rebuilt from the generator's own setpoint schedules and RunPeriod
start day (not from the module's constants); (3) every index for all 10 training EPWs is
recomputed by an independent pure-Python loop (datetime weekdays, math module, no numpy) and
must agree; (4) the degree-day routine reproduces the dataset's hdd18 column; (5) the holdout
climate is refused; (6) the join onto the dataset keeps row order and refuses unknown climates.
Only training EPWs and the training split are read.

Run:  ./venv/bin/python surrogate/tests/test_climate_indices.py   (or all suites: surrogate/run_tests.py)
"""

from __future__ import annotations

import datetime as dt
import math
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from climate.climate_features import daily_mean_temperature, degree_days, load_climate_registry, training_keys
from climate.epw_parser import parse_epw
from idf_generator.internal_loads import internal_loads as IL
from surrogate import climate_indices as CI
from surrogate import cv
from surrogate.paths import EPW_DIR, SIM_ROOT

_CACHE: dict = {}
REL_TOL = 1e-9


def _registry():
    if "reg" not in _CACHE:
        _CACHE["reg"] = load_climate_registry(check_files=False)
    return _CACHE["reg"]


def _epw(key):
    _CACHE.setdefault("epw", {})
    if key not in _CACHE["epw"]:
        _CACHE["epw"][key] = parse_epw(EPW_DIR / _registry()[key].epw_filename)
    return _CACHE["epw"][key]


def _table():
    if "table" not in _CACHE:
        _CACHE["table"] = CI.training_indices()
    return _CACHE["table"]


# ----------------------------------------------------------------------------

def check_psychrometrics_match_ashrae_tables():
    # ASHRAE Fundamentals 2017 ch. 1 Table 3 (saturation pressure, kPa)
    for t, p_kpa in ((-10.0, 0.25990), (0.0, 0.61121), (20.0, 2.3389), (24.0, 2.9852), (30.0, 4.2467)):
        got = float(CI.saturation_pressure(t)) / 1000
        assert abs(got - p_kpa) / p_kpa < 2e-4, (t, got, p_kpa)
    # Room reference 24 degC / 50 % RH at sea level: W ~ 0.00930 kg/kg, h ~ 47.8 kJ/kg
    p = 101325.0
    w = float(CI.humidity_ratio(0.5 * CI.saturation_pressure(24.0), p))
    h = float(CI.enthalpy(24.0, w))
    assert abs(w - 0.009299) < 2e-5 and abs(h - 47.82) < 0.05, (w, h)
    # Dry air: enthalpy is cp * t
    assert abs(float(CI.enthalpy(10.0, 0.0)) - 10.06) < 1e-12


def check_occupied_mask_matches_generator_schedules():
    src = (SIM_ROOT / "idf_generator" / "generator" / "generator.py").read_text()
    assert re.search(r'Day_of_Week_for_Start_Day="Sunday"', src), "RunPeriod start day changed"
    assert re.search(r'Use_Weather_File_Daylight_Saving_Period="No"', src), "daylight saving now used"
    occupied = {}
    for sched, occ_val in ((IL._HEATING_SETPOINT_SCHEDULE, IL.HEATING_SETPOINT_OCCUPIED),
                           (IL._COOLING_SETPOINT_SCHEDULE, IL.COOLING_SETPOINT_OCCUPIED)):
        for pat in sched.day_patterns:
            hours = set()
            for he in range(1, 25):                       # hour ending he covers (he-1, he)
                mid = he - 0.5
                start = 0.0
                for iv in pat.intervals:
                    hh, mm = iv.until_time.split(":")
                    end = int(hh) + int(mm) / 60
                    if start <= mid < end:
                        if iv.value == occ_val:
                            hours.add(he)
                        break
                    start = end
            for day_type in pat.day_types:
                prev = occupied.setdefault((sched.name, day_type), hours)
                assert prev == hours
    heat = {k[1]: v for k, v in occupied.items() if k[0] == IL._HEATING_SETPOINT_SCHEDULE.name}
    cool = {k[1]: v for k, v in occupied.items() if k[0] == IL._COOLING_SETPOINT_SCHEDULE.name}
    assert heat == cool, "heating and cooling occupied hours differ"
    expected = {wd: set(heat["Weekdays"]) for wd in (1, 2, 3, 4, 5)}
    expected[6] = set(heat["Saturday"])
    assert heat["AllOtherDays"] == set()
    assert {wd: set(h) for wd, h in CI.OCC_HOURS.items()} == expected
    mask = CI.occupied_mask(8760)
    days = mask.reshape(365, 24)
    first = dt.date(2023, 1, 1)                           # a year that starts on a Sunday
    assert first.weekday() == 6
    for d in range(365):
        wd = ((first + dt.timedelta(days=d)).weekday() + 1) % 7          # 0 = Sunday
        want = {h for h in range(1, 25) if h in expected.get(wd, set())}
        got = {h + 1 for h in np.flatnonzero(days[d])}
        assert got == want, (d, got, want)
    assert mask.sum() == 52 * 5 * 16 + 52 * 12 == 4784
    # spot hours: Mon 2 Jan hour-ending 06 off, 07 on, 22 on, 23 off; Sat 7 Jan 18 on, 19 off; Sun 1 Jan off
    idx = lambda day, he: (day - 1) * 24 + he - 1                     # noqa: E731
    assert not mask[idx(2, 6)] and mask[idx(2, 7)] and mask[idx(2, 22)] and not mask[idx(2, 23)]
    assert mask[idx(7, 18)] and not mask[idx(7, 19)] and not mask[idx(1, 12)]


def _indices_by_hand(epw):
    """Independent reimplementation: per-hour loop, datetime weekdays, math module only."""
    def pws(t):
        T = t + 273.15
        if T < 273.15:
            return math.exp(-5.6745359e3 / T + 6.3925247 - 9.677843e-3 * T + 6.2215701e-7 * T ** 2
                            + 2.0747825e-9 * T ** 3 - 9.484024e-13 * T ** 4 + 4.1635019 * math.log(T))
        return math.exp(-5.8002206e3 / T + 1.3914993 - 4.8640239e-2 * T + 4.1764768e-5 * T ** 2
                        - 1.4452093e-8 * T ** 3 + 6.5459673 * math.log(T))

    sums = dict.fromkeys(CI.INDEX_COLUMNS[:5], 0.0)
    first = dt.date(2023, 1, 1)
    for i in range(8760):
        day = first + dt.timedelta(days=i // 24)
        he = i % 24 + 1
        wd = day.weekday()                                  # Mon 0 .. Sun 6
        occ = (wd <= 4 and 7 <= he <= 22) or (wd == 5 and 7 <= he <= 18)
        if not occ:
            continue
        t, p = float(epw.dry_bulb[i]), float(epw.atm_pressure[i])
        pw = pws(float(epw.dew_point[i]))
        w_out = 0.621945 * pw / (p - pw)
        pr = 0.5 * pws(24.0)
        w_room = 0.621945 * pr / (p - pr)
        h_out = 1.006 * t + w_out * (2501 + 1.86 * t)
        h_room = 1.006 * 24 + w_room * (2501 + 1.86 * 24)
        sums["occ_enth_excess"] += max(0.0, h_out - h_room)
        sums["occ_latent_excess"] += max(0.0, w_out - w_room) * 2501
        sums["occ_band_dh"] += min(3.0, max(0.0, 24 - t))
        sums["occ_cool_dh"] += max(0.0, t - 24)
        sums["occ_heat_dh"] += max(0.0, 21 - t)
    for base, name in ((15.0, "hdd15"), (12.0, "hdd12")):
        sums[name] = sum(max(0.0, base - sum(float(x) for x in epw.dry_bulb[d * 24:(d + 1) * 24]) / 24)
                         for d in range(365))
    return sums


def check_indices_equal_independent_calculation():
    table = _table()
    assert list(table.index) == training_keys(_registry()) and len(table) == 10
    for key in table.index:
        hand = _indices_by_hand(_epw(key))
        for col in CI.INDEX_COLUMNS:
            got, want = float(table.loc[key, col]), hand[col]
            assert math.isclose(got, want, rel_tol=REL_TOL, abs_tol=1e-6), (key, col, got, want)


def check_degree_days_reproduce_dataset_hdd18():
    df, _ = cv.load_training()
    hdd18 = df.groupby("climate_key")["hdd18"].first()
    for key in _table().index:
        ours = degree_days(daily_mean_temperature(_epw(key).dry_bulb), 18.0, "heating")
        assert abs(ours - hdd18[key]) < 0.051, (key, ours, hdd18[key])   # dataset stores 1 decimal


def check_values_physically_ordered():
    t = _table()
    assert (t >= 0).all().all() and np.isfinite(t.to_numpy()).all()
    assert (t["hdd12"] <= t["hdd15"]).all()
    df, _ = cv.load_training()
    hdd18 = df.groupby("climate_key")["hdd18"].first()
    assert (t["hdd15"] <= hdd18[t.index] + 1e-9).all()
    for k in ("chennai", "colombo"):                        # tropical: no heating degree-days
        assert t.loc[k, "hdd15"] == 0 and t.loc[k, "occ_heat_dh"] < 10
    assert t["occ_band_dh"].idxmax() == "san_francisco"     # mildest climate
    assert set(t["occ_enth_excess"].nlargest(2).index) == {"chennai", "colombo"}
    assert (t["occ_band_dh"] <= CI.BAND_WIDTH * 4784 + 1e-9).all()


def check_holdout_refused():
    holdout = [k for k, r in _registry().items() if r.role == "holdout"]
    assert holdout == ["london"], holdout
    try:
        CI.indices_for_key("london")
    except CI.HoldoutClimateError:
        pass
    else:
        raise AssertionError("holdout climate indices were computed")
    try:
        CI.indices_for_key("atlantis")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown key accepted")


def check_calendar_guard():
    src = (EPW_DIR / _registry()["denver"].epw_filename).read_text(encoding="latin-1").splitlines(True)
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.epw"
        lines = [("HOLIDAYS/DAYLIGHT SAVINGS,Yes,0,0,1\n" if ln.upper().startswith("HOLIDAYS") else ln)
                 for ln in src]
        bad.write_text("".join(lines), encoding="latin-1")
        try:
            CI._check_calendar(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("holiday EPW accepted")
    CI._check_calendar(EPW_DIR / _registry()["denver"].epw_filename)     # real file passes


def check_join_keeps_rows_and_refuses_unknown():
    df, _ = cv.load_training()
    sub = df.sample(n=200, random_state=1)
    out = CI.add_indices(sub, _table())
    assert list(out.index) == list(sub.index) and list(out.columns[: len(sub.columns)]) == list(sub.columns)
    for col in CI.INDEX_COLUMNS:
        want = _table().loc[sub["climate_key"], col].to_numpy()
        assert np.array_equal(out[col].to_numpy(), want)
    bad = sub.head(3).copy()
    bad["climate_key"] = "london"
    try:
        CI.add_indices(bad, _table())
    except KeyError:
        pass
    else:
        raise AssertionError("row without indices accepted")
    assert set(CI.OA_LEVEL_INDICES) <= set(CI.INDEX_COLUMNS)


ALL_CHECKS = [check_psychrometrics_match_ashrae_tables, check_occupied_mask_matches_generator_schedules,
              check_indices_equal_independent_calculation, check_degree_days_reproduce_dataset_hdd18,
              check_values_physically_ordered, check_holdout_refused, check_calendar_guard,
              check_join_keeps_rows_and_refuses_unknown]


def main():
    total = failed = 0
    print("=" * 72)
    print("SURROGATE CLIMATE INDICES (3.4) - CHECKS")
    print("=" * 72)
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
