"""
test_hvac.py - Structural and integration checks for the hvac module.

Tests verify:
  1. One thermostat per HVACSet
  2. One IdealLoads system per geometry zone
  3. All systems reference the same thermostat
  4. Each system's zone_name matches a real geometry zone
  5. Setpoint schedule names are real (in internal_loads)
  6. Availability schedule name is real
  7. OA values match internal_loads
  8. All system names unique
  9. Default control features are off as expected
 10. Bad inputs are rejected
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repository root on sys.path, so the idf_generator package imports
# correctly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import idf_generator.geometry.geometry as G
import idf_generator.internal_loads.internal_loads as IL
import idf_generator.hvac.hvac as H
def _build(case=(5000.0, 1.5, 2, 3.5)):
    floor_area, ar, nf, fh = case
    geom = G.build_geometry(floor_area, ar, nf, fh)
    loads = IL.build_internal_loads()
    hvac = H.build_hvac(geom, loads)
    return geom, loads, hvac


GEOM_CASES = [
    (1500.0, 1.0, 1, 3.0),
    (10000.0, 3.0, 5, 4.5),
    (5000.0, 1.5, 3, 3.5),
]


# ---------- Structural ----------

def check_single_thermostat(case):
    _, _, hvac = _build(case)
    assert hvac.thermostat is not None, "No thermostat"
    assert isinstance(hvac.thermostat, H.ThermostatSpec)


def check_one_idealloads_per_zone(case):
    geom, _, hvac = _build(case)
    assert len(hvac.zone_systems) == len(geom.zones), \
        f"Expected {len(geom.zones)} IdealLoads, got {len(hvac.zone_systems)}"


def check_each_system_has_unique_name(case):
    _, _, hvac = _build(case)
    names = [s.name for s in hvac.zone_systems]
    assert len(names) == len(set(names)), \
        f"Duplicate system names: {names}"


# ---------- Cross-module references ----------

def check_systems_reference_thermostat(case):
    _, _, hvac = _build(case)
    for s in hvac.zone_systems:
        assert s.thermostat_name == hvac.thermostat.name, \
            f"{s.name}: thermostat_name {s.thermostat_name} != {hvac.thermostat.name}"


def check_thermostat_references_real_schedules(case):
    _, loads, hvac = _build(case)
    loads.schedule_by_name(hvac.thermostat.heating_setpoint_schedule_name)
    loads.schedule_by_name(hvac.thermostat.cooling_setpoint_schedule_name)


def check_availability_schedule_real(case):
    _, loads, hvac = _build(case)
    for s in hvac.zone_systems:
        loads.schedule_by_name(s.system_availability_schedule_name)


def check_each_system_matches_a_zone(case):
    geom, _, hvac = _build(case)
    zone_names = {z.name for z in geom.zones}
    for s in hvac.zone_systems:
        assert s.zone_name in zone_names, \
            f"{s.name}: zone_name {s.zone_name} not in geometry zones"


def check_oa_values_match_internal_loads(case):
    _, loads, hvac = _build(case)
    for s in hvac.zone_systems:
        assert s.oa_per_person == loads.oa_per_person, \
            f"{s.name}: OA per person {s.oa_per_person} != {loads.oa_per_person}"
        assert s.oa_per_area == loads.oa_per_area, \
            f"{s.name}: OA per area {s.oa_per_area} != {loads.oa_per_area}"


# ---------- Control features off ----------

def check_no_economizer(case):
    _, _, hvac = _build(case)
    for s in hvac.zone_systems:
        assert s.economizer_type == "NoEconomizer", \
            f"{s.name}: economizer is {s.economizer_type}"


def check_no_heat_recovery(case):
    _, _, hvac = _build(case)
    for s in hvac.zone_systems:
        assert s.heat_recovery_type == "None", \
            f"{s.name}: heat recovery is {s.heat_recovery_type}"


def check_no_humidity_control(case):
    _, _, hvac = _build(case)
    for s in hvac.zone_systems:
        assert s.dehumidification_control_type == "None"
        assert s.humidification_control_type == "None"


def check_unlimited_capacity(case):
    _, _, hvac = _build(case)
    for s in hvac.zone_systems:
        assert s.heating_limit == "NoLimit"
        assert s.cooling_limit == "NoLimit"


def check_no_dcv(case):
    _, _, hvac = _build(case)
    for s in hvac.zone_systems:
        assert s.demand_controlled_ventilation_type == "None"


def check_oa_method_is_sum(case):
    _, _, hvac = _build(case)
    for s in hvac.zone_systems:
        assert s.outdoor_air_method == "Sum"


# ---------- Naming consistency ----------

def check_system_names_derived_from_zone_names(case):
    _, _, hvac = _build(case)
    for s in hvac.zone_systems:
        assert s.name == f"{s.zone_name}_IdealLoads", \
            f"{s.name} does not match expected pattern for zone {s.zone_name}"


# ---------- Bad inputs ----------

def check_rejects_empty_geometry():
    """Geometry with no zones must be rejected."""
    # We can't easily construct an empty BuildingGeometry, but we can
    # mock one. Use a SimpleNamespace.
    from types import SimpleNamespace
    fake_geom = SimpleNamespace(zones=[])
    loads = IL.build_internal_loads()
    try:
        H.build_hvac(fake_geom, loads)  # type: ignore[arg-type]
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


# ---------- Runner ----------

PER_CASE = [
    check_single_thermostat,
    check_one_idealloads_per_zone,
    check_each_system_has_unique_name,
    check_systems_reference_thermostat,
    check_thermostat_references_real_schedules,
    check_availability_schedule_real,
    check_each_system_matches_a_zone,
    check_oa_values_match_internal_loads,
    check_no_economizer,
    check_no_heat_recovery,
    check_no_humidity_control,
    check_unlimited_capacity,
    check_no_dcv,
    check_oa_method_is_sum,
    check_system_names_derived_from_zone_names,
]

NO_CASE = [check_rejects_empty_geometry]


def main():
    total = failed = 0
    print("=" * 72)
    print("HVAC MODULE - STRUCTURAL & INTEGRATION CHECKS")
    print("=" * 72)
    for case in GEOM_CASES:
        label = f"area={case[0]:.0f}, AR={case[1]}, floors={case[2]}, h={case[3]}"
        print(f"\nCase: {label}")
        print("-" * 72)
        for chk in PER_CASE:
            total += 1
            try:
                chk(case)
                print(f"  PASS  {chk.__name__}")
            except AssertionError as e:
                failed += 1
                print(f"  FAIL  {chk.__name__}: {e}")
    print("\n" + "=" * 72)
    print("BAD-INPUT TESTS")
    print("=" * 72)
    for chk in NO_CASE:
        total += 1
        try:
            chk()
            print(f"  PASS  {chk.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {chk.__name__}: {e}")
    print("\n" + "=" * 72)
    print(f"RESULTS: {total - failed}/{total} passed, {failed} failed")
    print("=" * 72)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
