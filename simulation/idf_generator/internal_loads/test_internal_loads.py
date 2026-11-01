"""
test_internal_loads.py - Invariance checks for the internal_loads module.

Since this module produces fixed (not parameterized) data, tests focus
on STRUCTURAL invariants:
  - All named references resolve to a real schedule
  - All schedules reference a real ScheduleTypeLimits
  - Schedule intervals end at 24:00 (full day coverage)
  - Fraction-type values are in [0, 1]
  - Temperature values are in a physical range
  - Heating setpoint < cooling setpoint at every time
  - The 8 expected schedules and 4 type limits are present
  - No duplicate names

Run:
    python tests/test_internal_loads.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repository root on sys.path, so the idf_generator package imports
# correctly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import idf_generator.internal_loads.internal_loads as IL
TOLERANCE = 1e-6


def _time_to_hours(t: str) -> float:
    """Convert 'HH:MM' to a float hour value, e.g. '14:30' -> 14.5."""
    h, m = t.split(":")
    return int(h) + int(m) / 60.0


# ---------- Structural ----------

def check_expected_schedules_present():
    loads = IL.build_internal_loads()
    expected = {
        "OfficeOccupancy", "OfficeLighting", "OfficeEquipment",
        "OfficeActivity", "HeatingSetpoint", "CoolingSetpoint",
        "InfiltrationSchedule", "AlwaysOn",
    }
    actual = {s.name for s in loads.schedules}
    assert actual == expected, \
        f"Expected schedules {expected}, got {actual}"


def check_expected_type_limits_present():
    loads = IL.build_internal_loads()
    expected = {"Fraction", "Temperature", "ActivityLevel", "OnOff"}
    actual = {t.name for t in loads.schedule_type_limits}
    assert actual == expected, \
        f"Expected type limits {expected}, got {actual}"


def check_no_duplicate_schedule_names():
    loads = IL.build_internal_loads()
    names = [s.name for s in loads.schedules]
    assert len(names) == len(set(names)), f"Duplicate schedule names: {names}"


def check_no_duplicate_type_limits_names():
    loads = IL.build_internal_loads()
    names = [t.name for t in loads.schedule_type_limits]
    assert len(names) == len(set(names)), f"Duplicate type limit names: {names}"


def check_all_named_refs_resolve():
    loads = IL.build_internal_loads()
    refs = [
        loads.occupancy_schedule_name,
        loads.lighting_schedule_name,
        loads.equipment_schedule_name,
        loads.activity_schedule_name,
        loads.heating_setpoint_schedule_name,
        loads.cooling_setpoint_schedule_name,
        loads.infiltration_schedule_name,
        loads.always_on_schedule_name,
    ]
    for r in refs:
        loads.schedule_by_name(r)  # raises if missing


def check_all_schedules_reference_valid_type_limits():
    loads = IL.build_internal_loads()
    tl_names = {t.name for t in loads.schedule_type_limits}
    for s in loads.schedules:
        assert s.schedule_type_limits in tl_names, \
            f"Schedule {s.name!r} references unknown type limits {s.schedule_type_limits!r}"


# ---------- Schedule format ----------

def check_intervals_end_at_24():
    loads = IL.build_internal_loads()
    for s in loads.schedules:
        for dp in s.day_patterns:
            assert dp.intervals, f"{s.name}: empty day pattern"
            last = dp.intervals[-1]
            assert last.until_time == "24:00", \
                f"{s.name}: last interval ends at {last.until_time}, expected 24:00"


def check_intervals_monotonic():
    """Until times must be strictly increasing within a day pattern."""
    loads = IL.build_internal_loads()
    for s in loads.schedules:
        for dp in s.day_patterns:
            prev_h = 0.0
            for iv in dp.intervals:
                h = _time_to_hours(iv.until_time)
                assert h > prev_h, \
                    f"{s.name}: time {iv.until_time} not after previous {prev_h}"
                prev_h = h


def check_day_types_nonempty():
    loads = IL.build_internal_loads()
    for s in loads.schedules:
        for dp in s.day_patterns:
            assert len(dp.day_types) > 0, \
                f"{s.name}: day pattern with no day_types"


# ---------- Value ranges ----------

def check_fraction_values_in_range():
    loads = IL.build_internal_loads()
    for s in loads.schedules:
        if s.schedule_type_limits != "Fraction":
            continue
        for dp in s.day_patterns:
            for iv in dp.intervals:
                assert 0.0 <= iv.value <= 1.0, \
                    f"{s.name} value {iv.value} outside [0, 1]"


def check_temperature_values_reasonable():
    loads = IL.build_internal_loads()
    for s in loads.schedules:
        if s.schedule_type_limits != "Temperature":
            continue
        for dp in s.day_patterns:
            for iv in dp.intervals:
                assert 10.0 <= iv.value <= 35.0, \
                    f"{s.name} value {iv.value} unreasonable (expected 10..35 C)"


def check_onoff_values_binary():
    loads = IL.build_internal_loads()
    for s in loads.schedules:
        if s.schedule_type_limits != "OnOff":
            continue
        for dp in s.day_patterns:
            for iv in dp.intervals:
                assert iv.value in (0.0, 1.0), \
                    f"{s.name} OnOff value {iv.value} not 0 or 1"


def check_activity_level_positive():
    loads = IL.build_internal_loads()
    for s in loads.schedules:
        if s.schedule_type_limits != "ActivityLevel":
            continue
        for dp in s.day_patterns:
            for iv in dp.intervals:
                assert iv.value > 0, \
                    f"{s.name} activity value {iv.value} not positive"


# ---------- Setpoint consistency ----------

def check_heating_below_cooling():
    """At all four key values: heating < cooling. (No deadband overlap.)"""
    assert IL.HEATING_SETPOINT_OCCUPIED < IL.COOLING_SETPOINT_OCCUPIED, \
        "Heating occupied >= cooling occupied"
    assert IL.HEATING_SETPOINT_UNOCCUPIED < IL.COOLING_SETPOINT_UNOCCUPIED, \
        "Heating setback >= cooling setup"
    # Cross-checks:
    assert IL.HEATING_SETPOINT_UNOCCUPIED < IL.HEATING_SETPOINT_OCCUPIED, \
        "Heating setback >= heating occupied"
    assert IL.COOLING_SETPOINT_UNOCCUPIED > IL.COOLING_SETPOINT_OCCUPIED, \
        "Cooling setup <= cooling occupied"


def check_heating_setpoint_schedule_uses_correct_values():
    loads = IL.build_internal_loads()
    h = loads.schedule_by_name(loads.heating_setpoint_schedule_name)
    all_values = {iv.value for dp in h.day_patterns for iv in dp.intervals}
    assert IL.HEATING_SETPOINT_OCCUPIED in all_values, \
        f"Heating schedule missing occupied setpoint {IL.HEATING_SETPOINT_OCCUPIED}"
    assert IL.HEATING_SETPOINT_UNOCCUPIED in all_values, \
        f"Heating schedule missing unoccupied setpoint {IL.HEATING_SETPOINT_UNOCCUPIED}"


def check_cooling_setpoint_schedule_uses_correct_values():
    loads = IL.build_internal_loads()
    c = loads.schedule_by_name(loads.cooling_setpoint_schedule_name)
    all_values = {iv.value for dp in c.day_patterns for iv in dp.intervals}
    assert IL.COOLING_SETPOINT_OCCUPIED in all_values
    assert IL.COOLING_SETPOINT_UNOCCUPIED in all_values


# ---------- Density values ----------

def check_density_values_positive():
    loads = IL.build_internal_loads()
    assert loads.occupant_density > 0
    assert loads.lighting_power_density > 0
    assert loads.equipment_power_density > 0
    assert loads.activity_level > 0
    assert loads.oa_per_person > 0
    assert loads.oa_per_area > 0


def check_doe_density_values():
    """Match exactly the documented DOE Medium Office values."""
    loads = IL.build_internal_loads()
    assert loads.occupant_density == 0.0538
    assert loads.lighting_power_density == 6.9
    assert loads.equipment_power_density == 6.78
    assert loads.activity_level == 120.0
    assert loads.oa_per_person == 0.0025
    assert loads.oa_per_area == 0.0003


# ---------- Always-on sanity ----------

def check_always_on_is_actually_constant_1():
    loads = IL.build_internal_loads()
    a = loads.schedule_by_name(loads.always_on_schedule_name)
    # Single day pattern, single interval ending at 24:00 with value 1.0
    assert len(a.day_patterns) == 1
    assert a.day_patterns[0].intervals[-1].until_time == "24:00"
    for dp in a.day_patterns:
        for iv in dp.intervals:
            assert iv.value == 1.0


# ---------- Runner ----------

ALL_CHECKS = [
    check_expected_schedules_present,
    check_expected_type_limits_present,
    check_no_duplicate_schedule_names,
    check_no_duplicate_type_limits_names,
    check_all_named_refs_resolve,
    check_all_schedules_reference_valid_type_limits,
    check_intervals_end_at_24,
    check_intervals_monotonic,
    check_day_types_nonempty,
    check_fraction_values_in_range,
    check_temperature_values_reasonable,
    check_onoff_values_binary,
    check_activity_level_positive,
    check_heating_below_cooling,
    check_heating_setpoint_schedule_uses_correct_values,
    check_cooling_setpoint_schedule_uses_correct_values,
    check_density_values_positive,
    check_doe_density_values,
    check_always_on_is_actually_constant_1,
]


def main():
    total = failed = 0
    print("=" * 72)
    print("INTERNAL LOADS MODULE - STRUCTURAL CHECKS")
    print("=" * 72)
    for chk in ALL_CHECKS:
        total += 1
        try:
            chk()
            print(f"  PASS  {chk.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {chk.__name__}: {e}")
    print("=" * 72)
    print(f"RESULTS: {total - failed}/{total} passed, {failed} failed")
    print("=" * 72)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
