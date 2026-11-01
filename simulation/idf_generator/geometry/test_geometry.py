"""
test_geometry.py — Sanity checks for the geometry module.

Tests verify geometric invariants that must hold for ANY valid building
geometry, regardless of parameter values. They are pure-math checks and
do not require EnergyPlus.

Run:
    python -m pytest test_geometry.py -v
or
    python test_geometry.py    # runs all checks and prints a summary
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Repository root on sys.path, so the idf_generator package imports
# correctly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import idf_generator.geometry.geometry as G
TOLERANCE = 1e-6


# ---------- Parameter grid for invariance tests ----------

# Each tuple: (floor_area, aspect_ratio, n_floors, floor_height)
# Covers corners of the design space.
TEST_CASES = [
    (1500.0, 1.0, 1, 3.0),    # min area, square, single floor, low ceiling
    (1500.0, 3.0, 5, 4.5),    # min area, max AR, max floors, high ceiling
    (10000.0, 1.0, 5, 3.0),   # max area, square, max floors
    (10000.0, 3.0, 1, 4.5),   # max area, max AR, single floor
    (5000.0, 1.5, 3, 3.5),    # middle-of-range
    (5000.0, 2.0, 2, 4.0),    # another mid-range
]


def _build(case):
    floor_area, ar, nf, fh = case
    return G.build_geometry(
        floor_area=floor_area,
        aspect_ratio=ar,
        n_floors=nf,
        floor_height=fh,
    )


# ---------- Individual checks ----------

def check_dimensions(case):
    """L*W must equal floor_area, L/W must equal aspect_ratio."""
    floor_area, ar, _, _ = case
    geom = _build(case)
    assert abs(geom.length * geom.width - floor_area) < TOLERANCE * floor_area, \
        f"L*W = {geom.length * geom.width:.6f} != {floor_area}"
    assert abs(geom.length / geom.width - ar) < TOLERANCE * ar, \
        f"L/W = {geom.length / geom.width:.6f} != {ar}"


def check_footprints_ccw(case):
    """Every zone footprint must be CCW from above (signed area > 0)."""
    geom = _build(case)
    for z in geom.zones:
        signed = G._polygon_signed_area_2d(z.footprint)
        assert signed > 0, \
            f"Zone {z.name} footprint is not CCW (signed area = {signed:.4f})"


def check_floor_area_conservation(case):
    """Sum of zone floor areas per floor must equal L*W."""
    geom = _build(case)
    expected = geom.length * geom.width
    for f in range(geom.n_floors):
        actual = sum(z.floor_area for z in geom.zones if z.floor_index == f)
        assert abs(actual - expected) < TOLERANCE * expected, \
            f"Floor {f}: sum of zone areas = {actual:.4f}, expected {expected:.4f}"


def check_total_volume(case):
    """Total zone volume must equal building footprint * total height."""
    geom = _build(case)
    expected = geom.length * geom.width * geom.building_height
    actual = sum(z.volume for z in geom.zones)
    assert abs(actual - expected) < TOLERANCE * expected, \
        f"Total volume = {actual:.4f}, expected {expected:.4f}"


def check_cardinal_walls(case):
    """Each floor must have exactly 4 exterior walls — one of each N/E/S/W."""
    geom = _build(case)
    for f in range(geom.n_floors):
        ext = [s for z in geom.zones if z.floor_index == f
               for s in z.surfaces
               if s.surface_type == "Wall" and s.boundary_condition == "Outdoors"]
        dirs = sorted([s.direction for s in ext])
        assert dirs == ["East", "North", "South", "West"], \
            f"Floor {f}: exterior wall directions = {dirs}"


def check_wall_normals_outward(case):
    """Every wall's outward normal must point AWAY from the zone centroid."""
    geom = _build(case)
    for z in geom.zones:
        zc = G._zone_centroid(z)
        for s in z.surfaces:
            if s.surface_type != "Wall":
                continue
            n = G._polygon_normal(s.vertices)
            wc = G._polygon_centroid(s.vertices)
            out_vec = (wc[0] - zc[0], wc[1] - zc[1], wc[2] - zc[2])
            dot = sum(a * b for a, b in zip(n, out_vec))
            assert dot > 0, \
                f"{s.name}: normal {n} not outward (dot with out_vec = {dot:.4f})"


def check_floor_normal_down(case):
    """Floor surfaces must have outward normal pointing -Z."""
    geom = _build(case)
    for z in geom.zones:
        floor = next(s for s in z.surfaces if s.surface_type == "Floor")
        n = G._polygon_normal(floor.vertices)
        assert n[2] < -0.99, \
            f"{floor.name}: floor normal {n} not pointing down (-Z)"


def check_ceiling_normal_up(case):
    """Ceiling/Roof surfaces must have outward normal pointing +Z."""
    geom = _build(case)
    for z in geom.zones:
        ceil = next(s for s in z.surfaces
                    if s.surface_type in ("Ceiling", "Roof"))
        n = G._polygon_normal(ceil.vertices)
        assert n[2] > 0.99, \
            f"{ceil.name}: ceiling normal {n} not pointing up (+Z)"


def check_interior_surfaces_matched(case):
    """Every interior surface must point to a real other-surface name."""
    geom = _build(case)
    all_names = {s.name for z in geom.zones for s in z.surfaces}
    for z in geom.zones:
        for s in z.surfaces:
            if s.boundary_condition == "Surface":
                assert s.outside_boundary_object is not None, \
                    f"{s.name} is interior but has no matched surface"
                assert s.outside_boundary_object in all_names, \
                    f"{s.name} matched to nonexistent {s.outside_boundary_object}"


def check_matched_pair_symmetry(case):
    """If A matches B, then B must match A."""
    geom = _build(case)
    by_name = {s.name: s for z in geom.zones for s in z.surfaces}
    for s in by_name.values():
        if s.outside_boundary_object is None:
            continue
        other = by_name[s.outside_boundary_object]
        assert other.outside_boundary_object == s.name, \
            f"Asymmetric match: {s.name} -> {s.outside_boundary_object} -> {other.outside_boundary_object}"


def check_matched_pair_area_equal(case):
    """Matched surfaces must have equal area."""
    geom = _build(case)
    by_name = {s.name: s for z in geom.zones for s in z.surfaces}
    seen: set[tuple[str, str]] = set()
    for s in by_name.values():
        if s.outside_boundary_object is None:
            continue
        pair = tuple(sorted([s.name, s.outside_boundary_object]))
        if pair in seen:
            continue
        seen.add(pair)
        other = by_name[s.outside_boundary_object]
        assert abs(s.area - other.area) < TOLERANCE * max(s.area, 1.0), \
            f"Area mismatch: {s.name} ({s.area:.4f}) vs {other.name} ({other.area:.4f})"


def check_bottom_floor_is_ground(case):
    """All floor surfaces on floor index 0 must be Ground."""
    geom = _build(case)
    for z in geom.zones:
        if z.floor_index != 0:
            continue
        floor = next(s for s in z.surfaces if s.surface_type == "Floor")
        assert floor.boundary_condition == "Ground", \
            f"{floor.name}: expected Ground, got {floor.boundary_condition}"


def check_exposed_perimeter(case):
    """Core zones have no exposed edge; perimeter zones together trace the
    building outline 2(L+W) on every floor (F-factor input, Decision G1)."""
    geom = _build(case)
    for f in range(geom.n_floors):
        zs = [z for z in geom.zones if z.floor_index == f]
        for z in zs:
            if z.role == "Core":
                assert z.exposed_perimeter == 0.0, z.name
            else:
                assert z.exposed_perimeter > 0, z.name
        total = sum(z.exposed_perimeter for z in zs)
        assert abs(total - 2 * (geom.length + geom.width)) < 1e-6, (total, geom.length, geom.width)


def check_top_floor_is_roof(case):
    """All ceiling surfaces on top floor must be Roof + Outdoors."""
    geom = _build(case)
    top = geom.n_floors - 1
    for z in geom.zones:
        if z.floor_index != top:
            continue
        ceil = next(s for s in z.surfaces
                    if s.surface_type in ("Ceiling", "Roof"))
        assert ceil.surface_type == "Roof", \
            f"{ceil.name}: expected Roof, got {ceil.surface_type}"
        assert ceil.boundary_condition == "Outdoors", \
            f"{ceil.name}: expected Outdoors, got {ceil.boundary_condition}"


def check_exterior_wall_count(case):
    """
    Sanity: total exterior wall AREA per floor must equal the building's
    perimeter * floor_height (modulo floating-point).
    """
    geom = _build(case)
    expected_perimeter = 2 * (geom.length + geom.width)
    for f in range(geom.n_floors):
        ext_area = sum(s.area for z in geom.zones if z.floor_index == f
                       for s in z.surfaces
                       if s.surface_type == "Wall"
                       and s.boundary_condition == "Outdoors")
        expected = expected_perimeter * geom.floor_height
        assert abs(ext_area - expected) < TOLERANCE * expected, \
            f"Floor {f}: exterior wall area {ext_area:.4f} != {expected:.4f}"


# ---------- Bad-input tests ----------

def check_rejects_too_small_building():
    """build_geometry must reject buildings smaller than 2 * perim_depth on a side."""
    try:
        # 50 m^2 with AR=1 -> W=L=7.07 m, less than 2*4.57=9.14 m
        G.build_geometry(floor_area=50.0, aspect_ratio=1.0, n_floors=1, floor_height=3.0)
        raise AssertionError("Expected ValueError, got none")
    except ValueError:
        pass


def check_rejects_bad_aspect_ratio():
    try:
        G.build_geometry(floor_area=1500.0, aspect_ratio=0.5, n_floors=1, floor_height=3.0)
        raise AssertionError("Expected ValueError, got none")
    except ValueError:
        pass


def check_rejects_zero_floors():
    try:
        G.build_geometry(floor_area=1500.0, aspect_ratio=1.0, n_floors=0, floor_height=3.0)
        raise AssertionError("Expected ValueError, got none")
    except ValueError:
        pass


# ---------- Runner ----------

ALL_CHECKS = [
    check_dimensions,
    check_footprints_ccw,
    check_floor_area_conservation,
    check_total_volume,
    check_cardinal_walls,
    check_wall_normals_outward,
    check_floor_normal_down,
    check_ceiling_normal_up,
    check_interior_surfaces_matched,
    check_matched_pair_symmetry,
    check_matched_pair_area_equal,
    check_bottom_floor_is_ground,
    check_exposed_perimeter,
    check_top_floor_is_roof,
    check_exterior_wall_count,
]

BAD_INPUT_CHECKS = [
    check_rejects_too_small_building,
    check_rejects_bad_aspect_ratio,
    check_rejects_zero_floors,
]


def main():
    total = 0
    failed = 0
    print("=" * 72)
    print("GEOMETRY MODULE — INVARIANCE TESTS")
    print("=" * 72)

    for case in TEST_CASES:
        label = f"area={case[0]:.0f}, AR={case[1]}, floors={case[2]}, h={case[3]}"
        print(f"\nCase: {label}")
        print("-" * 72)
        for check in ALL_CHECKS:
            total += 1
            try:
                check(case)
                print(f"  PASS  {check.__name__}")
            except AssertionError as e:
                failed += 1
                print(f"  FAIL  {check.__name__}: {e}")

    print("\n" + "=" * 72)
    print("BAD-INPUT TESTS")
    print("=" * 72)
    for check in BAD_INPUT_CHECKS:
        total += 1
        try:
            check()
            print(f"  PASS  {check.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {check.__name__}: {e}")

    print("\n" + "=" * 72)
    print(f"RESULTS: {total - failed}/{total} passed, {failed} failed")
    print("=" * 72)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
