"""
test_constructions.py - Invariance and physical-correctness checks for
the constructions module.

Tests cover:
  1. Target R-values are met within tolerance
  2. Wall concrete heat capacity matches wall_thermal_mass target
  3. Every layer reference in a Construction exists in the Materials list
  4. All layer thicknesses are physically positive
  5. Bad inputs are correctly rejected
  6. get_construction_for_surface() returns the right name for every
     (surface_type, boundary_condition) combination

Run:
    python tests/test_constructions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repository root on sys.path, so the idf_generator package imports
# correctly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import idf_generator.construction.constructions as C
TOLERANCE = 1e-6


# ---------- Parameter grid ----------

# (wall_r, roof_r, floor_r, wall_mass)
TEST_CASES = [
    (1.0, 2.0, 0.5, 50.0),     # min envelope, min mass
    (8.0, 12.0, 5.0, 400.0),   # max envelope, max mass
    (4.0, 7.0, 2.5, 200.0),    # mid-range
    (2.5, 4.0, 1.0, 100.0),    # low-mid
    (6.0, 10.0, 3.5, 300.0),   # high-mid
]


def _build(case):
    wr, rr, fr, wm = case
    return C.build_constructions(
        wall_r_value=wr,
        roof_r_value=rr,
        floor_r_value=fr,
        wall_thermal_mass=wm,
    )


# ---------- R-value tests ----------

def check_wall_r_value(case):
    target = case[0]
    cs = _build(case)
    actual = cs.total_r_value("ExteriorWall")
    assert abs(actual - target) < TOLERANCE * target, \
        f"ExteriorWall R = {actual:.4f}, target {target:.4f}"


def check_roof_r_value(case):
    target = case[1]
    cs = _build(case)
    actual = cs.total_r_value("Roof")
    assert abs(actual - target) < TOLERANCE * target, \
        f"Roof R = {actual:.4f}, target {target:.4f}"


def check_floor_r_value(case):
    """floor_r_value becomes the slab F-factor (Decision G1)."""
    cs = _build(case)
    assert abs(cs.ground_floor_ffactor - C.ffactor_for_floor_r(case[2])) < 1e-12
    assert 0.2 < cs.ground_floor_ffactor < 1.3, cs.ground_floor_ffactor


# ---------- Thermal mass test ----------

def check_wall_concrete_mass(case):
    target = case[3]
    cs = _build(case)
    concrete = cs.material_by_name("WallConcrete")
    actual = concrete.heat_capacity  # kJ/m^2.K
    assert abs(actual - target) < TOLERANCE * target, \
        f"WallConcrete heat capacity = {actual:.4f}, target {target:.4f}"


# ---------- Structural tests ----------

def check_all_layers_in_materials_list(case):
    cs = _build(case)
    material_names = {m.name for m in cs.materials}
    for con in cs.constructions:
        for layer_name in con.layers:
            assert layer_name in material_names, \
                f"Construction {con.name!r} references unknown material {layer_name!r}"


def check_positive_thicknesses(case):
    cs = _build(case)
    for m in cs.materials:
        assert m.thickness > 0, \
            f"Material {m.name!r} has non-positive thickness {m.thickness}"


def check_no_duplicate_material_names(case):
    cs = _build(case)
    names = [m.name for m in cs.materials]
    assert len(names) == len(set(names)), \
        f"Duplicate material names: {names}"


def check_no_duplicate_construction_names(case):
    cs = _build(case)
    names = [c.name for c in cs.constructions]
    assert len(names) == len(set(names)), \
        f"Duplicate construction names: {names}"


def check_all_named_refs_exist(case):
    """The 6 named references must resolve to real constructions."""
    cs = _build(case)
    for ref in (cs.exterior_wall, cs.interior_wall, cs.roof,
                cs.interior_ceiling_floor, cs.interior_ceiling):
        cs.construction_by_name(ref)  # raises if missing


# ---------- Physical reasonableness ----------

def check_layer_thicknesses_reasonable(case):
    """Reject grossly unphysical layer thicknesses (>1 m or <1 mm)."""
    cs = _build(case)
    for m in cs.materials:
        assert 1e-4 <= m.thickness <= 1.0, \
            f"Material {m.name!r} thickness {m.thickness*1000:.2f} mm is out of range"


def check_construction_layer_counts(case):
    cs = _build(case)
    expected = {
        "ExteriorWall": 4,
        "InteriorWall": 3,
        "Roof": 4,
        "InteriorCeilingFloor": 3,
        "InteriorCeilingFloorReversed": 3,
    }
    assert {c.name for c in cs.constructions} == set(expected), \
        "the layered GroundFloor construction must be gone (F-factor per zone instead)"
    for name, n_expected in expected.items():
        con = cs.construction_by_name(name)
        assert len(con.layers) == n_expected, \
            f"Construction {name} has {len(con.layers)} layers, expected {n_expected}"


# ---------- Bad-input tests ----------

def check_rejects_negative_r_value():
    try:
        C.build_constructions(wall_r_value=-1.0, roof_r_value=4.0,
                              floor_r_value=2.0, wall_thermal_mass=100.0)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def check_rejects_zero_mass():
    try:
        C.build_constructions(wall_r_value=4.0, roof_r_value=4.0,
                              floor_r_value=2.0, wall_thermal_mass=0.0)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def check_rejects_r_value_too_low_for_fixed_layers():
    """
    With minimum concrete (5 mm clamp), fixed wall layers contribute
    about 0.13 m^2.K/W. Asking for R = 0.1 must fail.
    """
    try:
        C.build_constructions(wall_r_value=0.05, roof_r_value=4.0,
                              floor_r_value=2.0, wall_thermal_mass=50.0)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


# ---------- Surface-mapping tests ----------

def check_get_construction_exterior_wall():
    cs = _build((4.0, 4.0, 2.0, 100.0))
    name = C.get_construction_for_surface("Wall", "Outdoors", cs)
    assert name == "ExteriorWall", f"Got {name}"


def check_get_construction_interior_wall():
    cs = _build((4.0, 4.0, 2.0, 100.0))
    name = C.get_construction_for_surface("Wall", "Surface", cs)
    assert name == "InteriorWall", f"Got {name}"


def check_get_construction_roof():
    cs = _build((4.0, 4.0, 2.0, 100.0))
    name = C.get_construction_for_surface("Roof", "Outdoors", cs)
    assert name == "Roof", f"Got {name}"


def check_get_construction_interior_ceiling():
    cs = _build((4.0, 4.0, 2.0, 100.0))
    name = C.get_construction_for_surface("Ceiling", "Surface", cs)
    assert name == "InteriorCeilingFloorReversed", f"Got {name}"


def check_interzone_ceiling_mirrors_floor():
    """EnergyPlus requires the two sides of a floor/ceiling slab to be mirrored."""
    cs = _build((4.0, 4.0, 2.0, 100.0))
    floor = cs.construction_by_name(cs.interior_ceiling_floor)
    ceiling = cs.construction_by_name(cs.interior_ceiling)
    assert tuple(ceiling.layers) == tuple(reversed(floor.layers)), \
        f"ceiling {ceiling.layers} is not the reverse of floor {floor.layers}"


def check_get_construction_interior_floor():
    cs = _build((4.0, 4.0, 2.0, 100.0))
    name = C.get_construction_for_surface("Floor", "Surface", cs)
    assert name == "InteriorCeilingFloor", f"Got {name}"


def check_get_construction_ground_floor():
    """Ground floors are per-zone F-factor constructions, not a shared name."""
    cs = _build((4.0, 4.0, 2.0, 100.0))
    for bc in ("Ground", "GroundFCfactorMethod"):
        try:
            C.get_construction_for_surface("Floor", bc, cs)
            raise AssertionError("Expected ValueError")
        except ValueError:
            pass
    assert C.ground_floor_construction_name("Zone_F0_Core") == "GroundFloor_Zone_F0_Core"


# ---------- F-factor (Decision G1) ----------

def check_ffactor_reproduces_ashrae_table():
    """Exact at every tabulated point of ASHRAE 90.1 Table A6.3.1 (fully insulated, IP)."""
    for r_ip, f_ip in C.FFACTOR_TABLE_IP[1:]:
        f = C.ffactor_for_floor_r(r_ip / C.R_SI_TO_IP) / C.F_IP_TO_SI
        assert abs(f - f_ip) < 1e-9, (r_ip, f, f_ip)
    # R -> 0 tends to the uninsulated slab F-0.73
    assert abs(C.ffactor_for_floor_r(1e-9) / C.F_IP_TO_SI - 0.73) < 1e-6


def check_ffactor_strictly_decreasing_and_positive():
    rs = [0.5 + 0.05 * i for i in range(91)]            # the sampled range 0.5 .. 5.0
    fs = [C.ffactor_for_floor_r(r) for r in rs]
    assert all(b < a for a, b in zip(fs, fs[1:])), "more insulation must lower F"
    assert fs[-1] > 0.3, fs[-1]


def check_ffactor_extrapolation_continuous():
    r20 = 20.0 / C.R_SI_TO_IP
    lo, hi = C.ffactor_for_floor_r(r20 - 1e-9), C.ffactor_for_floor_r(r20 + 1e-9)
    assert abs(lo - hi) < 1e-6


def check_ffactor_units():
    assert abs(C.ffactor_for_floor_r(10.0 / C.R_SI_TO_IP) - 0.36 * 1.730735) < 1e-9
    try:
        C.ffactor_for_floor_r(0.0)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def check_get_construction_unknown_combo_rejected():
    cs = _build((4.0, 4.0, 2.0, 100.0))
    try:
        C.get_construction_for_surface("Wall", "Ground", cs)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


# ---------- Runner ----------

ALL_CHECKS = [
    check_wall_r_value,
    check_roof_r_value,
    check_floor_r_value,
    check_wall_concrete_mass,
    check_all_layers_in_materials_list,
    check_positive_thicknesses,
    check_no_duplicate_material_names,
    check_no_duplicate_construction_names,
    check_all_named_refs_exist,
    check_layer_thicknesses_reasonable,
    check_construction_layer_counts,
]

NO_ARG_CHECKS = [
    check_rejects_negative_r_value,
    check_rejects_zero_mass,
    check_rejects_r_value_too_low_for_fixed_layers,
    check_get_construction_exterior_wall,
    check_get_construction_interior_wall,
    check_get_construction_roof,
    check_get_construction_interior_ceiling,
    check_interzone_ceiling_mirrors_floor,
    check_get_construction_interior_floor,
    check_get_construction_ground_floor,
    check_get_construction_unknown_combo_rejected,
    check_ffactor_reproduces_ashrae_table,
    check_ffactor_strictly_decreasing_and_positive,
    check_ffactor_extrapolation_continuous,
    check_ffactor_units,
]


def main():
    total = 0
    failed = 0
    print("=" * 72)
    print("CONSTRUCTIONS MODULE - INVARIANCE TESTS")
    print("=" * 72)

    for case in TEST_CASES:
        label = f"wall_r={case[0]}, roof_r={case[1]}, floor_r={case[2]}, mass={case[3]}"
        print(f"\nCase: {label}")
        print("-" * 72)
        for chk in ALL_CHECKS:
            total += 1
            try:
                chk(case)
                print(f"  PASS  {chk.__name__}")
            except AssertionError as e:
                failed += 1
                print(f"  FAIL  {chk.__name__}: {e}")

    print("\n" + "=" * 72)
    print("BAD-INPUT & MAPPING TESTS")
    print("=" * 72)
    for chk in NO_ARG_CHECKS:
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
