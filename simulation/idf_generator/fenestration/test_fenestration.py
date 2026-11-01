"""
test_fenestration.py - Invariance and physical-correctness checks.

Tests verify:
  1. Window count = number of exterior walls
  2. Window area / parent wall area = WWR (within tolerance)
  3. Window vertices lie in the parent wall's plane
  4. Window vertices stay inside the wall's u,v range (no overflow)
  5. Window's outward normal matches the wall's outward normal
  6. Overhang outward normal points +Z (upward)
  7. Left fin outward normal = -u_hat of parent wall
  8. Right fin outward normal = +u_hat of parent wall
  9. Overhang & fin counts correct (per non-zero parameters)
 10. VT clipped to <= SHGC
 11. Bad inputs rejected

Run:
    python tests/test_fenestration.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Repository root on sys.path, so the idf_generator package imports
# correctly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import idf_generator.geometry.geometry as G
import idf_generator.fenestration.fenestration as F
TOLERANCE = 1e-6
ANGLE_TOL = 1e-4  # for normal vector comparisons


def _build_geom(case):
    floor_area, ar, nf, fh = case
    return G.build_geometry(
        floor_area=floor_area,
        aspect_ratio=ar,
        n_floors=nf,
        floor_height=fh,
    )


def _build_fen(geom, **overrides):
    defaults = dict(
        wwr_north=0.3, wwr_east=0.4, wwr_south=0.5, wwr_west=0.6,
        glazing_u=2.5, glazing_shgc=0.4, glazing_vt=0.3,
        overhang_depth_n=0.0, overhang_depth_e=0.5,
        overhang_depth_s=1.0, overhang_depth_w=1.5,
        sidefin_depth=0.5,
    )
    defaults.update(overrides)
    return F.build_fenestration(geom, **defaults)


GEOM_CASES = [
    (1500.0, 1.0, 1, 3.0),
    (1500.0, 3.0, 5, 4.5),
    (10000.0, 1.0, 5, 3.0),
    (10000.0, 3.0, 1, 4.5),
    (5000.0, 1.5, 3, 3.5),
    (5000.0, 2.0, 2, 4.0),
]


# ---------- Structural ----------

def check_window_count(case):
    geom = _build_geom(case)
    fen = _build_fen(geom)
    expected = sum(len(z.exterior_walls) for z in geom.zones)
    assert len(fen.windows) == expected, \
        f"Window count {len(fen.windows)} != exterior wall count {expected}"


def check_one_window_per_wall(case):
    geom = _build_geom(case)
    fen = _build_fen(geom)
    parent_walls = [w.parent_wall_name for w in fen.windows]
    assert len(parent_walls) == len(set(parent_walls)), \
        f"Some walls have multiple windows: duplicates in {parent_walls}"


# ---------- Area / WWR ----------

def check_window_area_matches_wwr(case):
    geom = _build_geom(case)
    fen = _build_fen(geom, wwr_north=0.3, wwr_east=0.4, wwr_south=0.5, wwr_west=0.6)
    expected_wwr = {"North": 0.3, "East": 0.4, "South": 0.5, "West": 0.6}
    # Build a wall name -> direction map
    wall_dirs = {s.name: s.direction for z in geom.zones
                 for s in z.surfaces if s.surface_type == "Wall" and s.boundary_condition == "Outdoors"}
    wall_areas = {s.name: G._polygon_area_3d(s.vertices) for z in geom.zones
                  for s in z.surfaces if s.surface_type == "Wall" and s.boundary_condition == "Outdoors"}
    for window in fen.windows:
        direction = wall_dirs[window.parent_wall_name]
        wwr_actual = window.area / wall_areas[window.parent_wall_name]
        assert abs(wwr_actual - expected_wwr[direction]) < TOLERANCE, \
            f"{window.name}: WWR = {wwr_actual:.4f}, expected {expected_wwr[direction]}"


# ---------- Coplanarity ----------

def _plane_signed_distance(p: tuple, plane_point: tuple, plane_normal: tuple) -> float:
    d = (p[0] - plane_point[0]) * plane_normal[0] \
      + (p[1] - plane_point[1]) * plane_normal[1] \
      + (p[2] - plane_point[2]) * plane_normal[2]
    return d


def check_window_in_wall_plane(case):
    geom = _build_geom(case)
    fen = _build_fen(geom)
    walls = {s.name: s for z in geom.zones for s in z.surfaces
             if s.surface_type == "Wall" and s.boundary_condition == "Outdoors"}
    for window in fen.windows:
        wall = walls[window.parent_wall_name]
        n_hat = G._polygon_normal(wall.vertices)
        A_bot = wall.vertices[1]
        for v in window.vertices:
            d = _plane_signed_distance(v, A_bot, n_hat)
            assert abs(d) < 1e-6, \
                f"{window.name}: vertex {v} off wall plane by {d:.6e}"


# ---------- Bounds ----------

def check_window_inside_wall(case):
    geom = _build_geom(case)
    fen = _build_fen(geom)
    walls = {s.name: s for z in geom.zones for s in z.surfaces
             if s.surface_type == "Wall" and s.boundary_condition == "Outdoors"}
    for window in fen.windows:
        wall = walls[window.parent_wall_name]
        A_top, A_bot, B_bot, B_top = wall.vertices
        u_vec = F._sub(B_bot, A_bot)
        v_vec = F._sub(A_top, A_bot)
        L = F._norm(u_vec)
        H = F._norm(v_vec)
        u_hat = F._scale(u_vec, 1/L)
        v_hat = F._scale(v_vec, 1/H)
        for vert in window.vertices:
            local = F._sub(vert, A_bot)
            u = F._dot(local, u_hat)
            v = F._dot(local, v_hat)
            assert -TOLERANCE <= u <= L + TOLERANCE, \
                f"{window.name}: u-coord {u:.4f} outside [0, {L:.4f}]"
            assert -TOLERANCE <= v <= H + TOLERANCE, \
                f"{window.name}: v-coord {v:.4f} outside [0, {H:.4f}]"


# ---------- Normals ----------

def _normals_aligned(n1, n2, tol=ANGLE_TOL):
    """Check that two unit vectors point in the same direction."""
    dot = n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2]
    return dot > 1.0 - tol


def check_window_normal_matches_wall(case):
    geom = _build_geom(case)
    fen = _build_fen(geom)
    walls = {s.name: s for z in geom.zones for s in z.surfaces
             if s.surface_type == "Wall" and s.boundary_condition == "Outdoors"}
    for window in fen.windows:
        wall = walls[window.parent_wall_name]
        n_wall = G._polygon_normal(wall.vertices)
        n_win = F._polygon_unit_normal(window.vertices)
        assert _normals_aligned(n_wall, n_win), \
            f"{window.name}: window normal {n_win} != wall normal {n_wall}"


def check_overhang_normal_up(case):
    geom = _build_geom(case)
    fen = _build_fen(geom, overhang_depth_n=1.0, overhang_depth_e=1.0,
                    overhang_depth_s=1.0, overhang_depth_w=1.0)
    overhangs = [s for s in fen.shading if s.kind == "Overhang"]
    for oh in overhangs:
        n = F._polygon_unit_normal(oh.vertices)
        assert _normals_aligned(n, (0.0, 0.0, 1.0)), \
            f"{oh.name}: overhang normal {n} not +Z"


def check_sidefin_normals_outward(case):
    geom = _build_geom(case)
    fen = _build_fen(geom, sidefin_depth=0.5)
    walls = {s.name: s for z in geom.zones for s in z.surfaces
             if s.surface_type == "Wall" and s.boundary_condition == "Outdoors"}
    for fin in fen.shading:
        if fin.kind not in ("SideFinLeft", "SideFinRight"):
            continue
        wall = walls[fin.base_surface_name]
        A_top, A_bot, B_bot, B_top = wall.vertices
        u_vec = F._sub(B_bot, A_bot)
        u_hat = F._scale(u_vec, 1/F._norm(u_vec))

        n_fin = F._polygon_unit_normal(fin.vertices)
        if fin.kind == "SideFinLeft":
            expected = F._scale(u_hat, -1.0)
        else:  # SideFinRight
            expected = u_hat
        assert _normals_aligned(n_fin, expected), \
            f"{fin.name} ({fin.kind}): normal {n_fin} != expected {expected}"


# ---------- Shading counts ----------

def check_shading_counts_match_params(case):
    geom = _build_geom(case)
    n_ext = sum(len(z.exterior_walls) for z in geom.zones)
    n_per_dir = n_ext // 4  # walls per cardinal direction (= n_floors)

    # All zeros -> no shading
    fen = _build_fen(
        geom,
        overhang_depth_n=0.0, overhang_depth_e=0.0,
        overhang_depth_s=0.0, overhang_depth_w=0.0,
        sidefin_depth=0.0,
    )
    assert len(fen.shading) == 0, \
        f"Expected no shading, got {len(fen.shading)}"

    # Only south overhangs and fins
    fen = _build_fen(
        geom,
        overhang_depth_n=0.0, overhang_depth_e=0.0,
        overhang_depth_s=1.0, overhang_depth_w=0.0,
        sidefin_depth=0.5,
    )
    overhangs = [s for s in fen.shading if s.kind == "Overhang"]
    fins = [s for s in fen.shading if s.kind in ("SideFinLeft", "SideFinRight")]
    assert len(overhangs) == n_per_dir, \
        f"Expected {n_per_dir} overhangs (south only), got {len(overhangs)}"
    assert len(fins) == 2 * n_ext, \
        f"Expected {2*n_ext} fins (all walls), got {len(fins)}"


# ---------- Glazing constraint ----------

def check_vt_clipped_to_shgc(case):
    geom = _build_geom(case)
    fen = _build_fen(geom, glazing_shgc=0.3, glazing_vt=0.6)
    assert abs(fen.glazing.visible_transmittance - 0.3) < TOLERANCE, \
        f"VT={fen.glazing.visible_transmittance}, expected clip to 0.3"
    # And when VT < SHGC, no clipping should occur
    fen = _build_fen(geom, glazing_shgc=0.6, glazing_vt=0.3)
    assert abs(fen.glazing.visible_transmittance - 0.3) < TOLERANCE, \
        f"VT={fen.glazing.visible_transmittance}, expected unchanged 0.3"


# ---------- Bad inputs ----------

def check_rejects_wwr_out_of_range():
    geom = _build_geom((1500.0, 1.0, 1, 3.0))
    try:
        _build_fen(geom, wwr_north=1.5)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass
    try:
        _build_fen(geom, wwr_north=-0.1)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def check_tiny_shading_not_created():
    """Depths below MIN_SHADING_DEPTH would collapse in EnergyPlus (vertex tolerance 0.01 m)."""
    geom = _build_geom(GEOM_CASES[0])
    n_per_dir = sum(len(z.exterior_walls) for z in geom.zones) // 4
    fen = _build_fen(geom, overhang_depth_n=0.005, overhang_depth_e=0.0199,
                     overhang_depth_s=0.02, overhang_depth_w=0.0, sidefin_depth=0.015)
    kinds = [s.kind for s in fen.shading]
    assert kinds.count("Overhang") == n_per_dir, "only the 0.02 m south overhangs are created"
    assert not [k for k in kinds if k.startswith("SideFin")], "0.015 m fins must be skipped"
    for s in fen.shading:                      # every created edge clears EnergyPlus' tolerance
        v = s.vertices
        for a, b in zip(v, v[1:] + v[:1]):
            assert math.dist(a, b) >= 0.01, (s.name, math.dist(a, b))


def check_rejects_negative_overhang():
    geom = _build_geom((1500.0, 1.0, 1, 3.0))
    try:
        _build_fen(geom, overhang_depth_s=-0.5)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


# ---------- Runner ----------

PER_CASE = [
    check_window_count,
    check_one_window_per_wall,
    check_window_area_matches_wwr,
    check_window_in_wall_plane,
    check_window_inside_wall,
    check_window_normal_matches_wall,
    check_overhang_normal_up,
    check_sidefin_normals_outward,
    check_shading_counts_match_params,
    check_vt_clipped_to_shgc,
]

NO_CASE = [
    check_rejects_wwr_out_of_range,
    check_tiny_shading_not_created,
    check_rejects_negative_overhang,
]


def main():
    total = failed = 0
    print("=" * 72)
    print("FENESTRATION MODULE - INVARIANCE TESTS")
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
