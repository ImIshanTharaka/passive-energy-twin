"""
fenestration.py - Windows, glazing, and shading geometry.

This module places one window on every exterior wall of the building,
plus optional shading surfaces (overhang above each window, plus a left
and right side fin). It consumes BuildingGeometry from geometry.py and
the fenestration design parameters, and produces a FenestrationSet of
data structures that generator.py translates into:

    WindowMaterial:SimpleGlazingSystem  - one per simulation
    Construction (for the window)       - one per simulation
    FenestrationSurface:Detailed        - one per exterior wall
    Shading:Zone:Detailed               - up to 3 per window:
                                          (overhang + 2 fins, optional)

PARAMETERS
----------
  WWR (window-to-wall ratio) per cardinal direction:
      wwr_north, wwr_east, wwr_south, wwr_west   each in [0.10, 0.80]

  Glazing (single value, all facades):
      glazing_u           [0.8, 5.8]  W/m^2.K   NFRC U-factor
      glazing_shgc        [0.10, 0.80]          Solar Heat Gain Coefficient
      glazing_vt          [0.10, 0.80]          Visible Transmittance
      Physical constraint: VT <= SHGC. Enforced by clipping at construct
      time (VT_used = min(VT, SHGC)).

  Shading depths:
      overhang_depth_n/e/s/w   each in [0.0, 1.5]  m
      sidefin_depth            [0.0, 1.0]  m
      A device shallower than MIN_SHADING_DEPTH (0.02 m) is NOT created
      (2.5 pilot, design P15: a 5 mm overhang). EnergyPlus merges vertices
      closer than 0.01 m (SurfaceGeometry.cc, checkPopCoincidentVertex), so
      such a device collapses to a zero-area "degenerate surface" - a SEVERE
      error. A <2 cm device casts no meaningful shade, so results are unchanged.

WINDOW PLACEMENT
----------------
One centered window per exterior wall. Window dimensions:
    W_window = L_wall * sqrt(WWR)
    H_window = H_wall * sqrt(WWR)
so that W_window * H_window = WWR * L_wall * H_wall (target area met)
and the window's aspect ratio matches the wall's. Margin between window
and wall edge is uniform on all 4 sides.

VERTEX CONVENTIONS
------------------
Wall-local frame (per wall, derived from its vertex list
[A_top, A_bot, B_bot, B_top]):
    u_hat = unit(B_bot - A_bot)      along wall length
    v_hat = unit(A_top - A_bot)      vertical (= +Z)
    n_hat = unit(u_hat x v_hat)      outward normal
    L_wall = |B_bot - A_bot|
    H_wall = |A_top - A_bot|
    Origin at A_bot.
A wall-local (u, v) point in world coordinates:
    P(u, v) = A_bot + u * u_hat + v * v_hat
With outward offset d:
    P(u, v, d) = P(u, v) + d * n_hat

Window order [UL, LL, LR, UR] in the wall-local frame:
    UL = (u_left,  v_top)
    LL = (u_left,  v_bot)
    LR = (u_right, v_bot)
    UR = (u_right, v_top)
Matches the parent wall's [A_top, A_bot, B_bot, B_top] ordering, so the
window's outward normal matches the wall's.

Overhang order [inside-left, outside-left, outside-right, inside-right]
yields outward normal +Z (verified analytically below). Outside corners
are at depth d along n_hat.

Left fin order [LL, LL+s*n_hat, UL+s*n_hat, UL] -> outward normal -u_hat
Right fin order [LR, UR, UR+s*n_hat, LR+s*n_hat] -> outward normal +u_hat

GLAZING (SimpleGlazingSystem)
------------------------------
EnergyPlus's SimpleGlazingSystem accepts U-factor, SHGC, and VT directly
and infers all other optical properties internally. This matches the
abstraction level of the design parameters. VT is physically bounded
above by SHGC (visible light is a subset of solar), so we clip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

# Shading devices shallower than this are not created (EnergyPlus vertex
# tolerance is 0.01 m; 2x for margin). See module docstring.
MIN_SHADING_DEPTH = 0.02    # m

if TYPE_CHECKING:
    from ..geometry.geometry import BuildingGeometry, Surface


# ---------- Type aliases ----------

Vertex = tuple[float, float, float]
CardinalDirection = Literal["North", "East", "South", "West"]


# ---------- Data classes ----------

@dataclass(frozen=True)
class GlazingSpec:
    """
    SimpleGlazingSystem properties. One per simulation; applied to every
    window via a single Construction referenced by name.
    """
    material_name: str
    construction_name: str
    u_factor: float                     # W/m^2.K (NFRC, includes air films)
    shgc: float                         # 0..1
    visible_transmittance: float        # 0..1, clipped to <= shgc


@dataclass(frozen=True)
class WindowSurface:
    """One window. Translates to FenestrationSurface:Detailed in IDF."""
    name: str
    parent_wall_name: str               # required for FenestrationSurface
    construction_name: str
    vertices: tuple[Vertex, ...]        # 4 vertices, same convention as wall

    @property
    def area(self) -> float:
        return _polygon_area_3d(self.vertices)


@dataclass(frozen=True)
class ShadingSurface:
    """
    One shading element (overhang or side fin). Translates to
    Shading:Zone:Detailed in IDF.

    kind: "Overhang" or "SideFinLeft" or "SideFinRight". Used only for
    naming and reporting; EnergyPlus does not distinguish.
    """
    name: str
    base_surface_name: str              # parent wall
    kind: Literal["Overhang", "SideFinLeft", "SideFinRight"]
    vertices: tuple[Vertex, ...]

    @property
    def area(self) -> float:
        return _polygon_area_3d(self.vertices)


@dataclass(frozen=True)
class FenestrationSet:
    """Complete fenestration output for one simulation."""
    glazing: GlazingSpec
    windows: tuple[WindowSurface, ...]
    shading: tuple[ShadingSurface, ...]


# ---------- Top-level API ----------

def build_fenestration(
    geometry: "BuildingGeometry",
    wwr_north: float,
    wwr_east: float,
    wwr_south: float,
    wwr_west: float,
    glazing_u: float,
    glazing_shgc: float,
    glazing_vt: float,
    overhang_depth_n: float,
    overhang_depth_e: float,
    overhang_depth_s: float,
    overhang_depth_w: float,
    sidefin_depth: float,
) -> FenestrationSet:
    """
    Build all windows and shading surfaces for the given geometry.

    Returns
    -------
    FenestrationSet with the GlazingSpec (single), one WindowSurface per
    exterior wall, and 0..3 ShadingSurfaces per window.

    Raises
    ------
    ValueError on invalid parameter values.
    """
    _validate(
        wwr_north, wwr_east, wwr_south, wwr_west,
        glazing_u, glazing_shgc, glazing_vt,
        overhang_depth_n, overhang_depth_e, overhang_depth_s, overhang_depth_w,
        sidefin_depth,
    )

    # Apply physical constraint: VT <= SHGC
    vt_clipped = min(glazing_vt, glazing_shgc)

    glazing = GlazingSpec(
        material_name="GlazingMaterial",
        construction_name="WindowConstruction",
        u_factor=glazing_u,
        shgc=glazing_shgc,
        visible_transmittance=vt_clipped,
    )

    wwr_by_dir: dict[CardinalDirection, float] = {
        "North": wwr_north,
        "East":  wwr_east,
        "South": wwr_south,
        "West":  wwr_west,
    }
    overhang_by_dir: dict[CardinalDirection, float] = {
        "North": overhang_depth_n,
        "East":  overhang_depth_e,
        "South": overhang_depth_s,
        "West":  overhang_depth_w,
    }

    windows: list[WindowSurface] = []
    shading: list[ShadingSurface] = []

    for zone in geometry.zones:
        for wall in zone.exterior_walls:
            # wall.direction is one of "North"/"East"/"South"/"West" for
            # exterior walls (enforced by geometry.py).
            direction: CardinalDirection = wall.direction  # type: ignore[assignment]
            wwr = wwr_by_dir[direction]
            overhang_d = overhang_by_dir[direction]

            window, shades = _place_window_on_wall(
                wall=wall,
                wwr=wwr,
                overhang_depth=overhang_d,
                sidefin_depth=sidefin_depth,
                construction_name=glazing.construction_name,
            )
            windows.append(window)
            shading.extend(shades)

    return FenestrationSet(
        glazing=glazing,
        windows=tuple(windows),
        shading=tuple(shading),
    )


# ---------- Per-wall window + shading placement ----------

def _place_window_on_wall(
    wall: "Surface",
    wwr: float,
    overhang_depth: float,
    sidefin_depth: float,
    construction_name: str,
) -> tuple[WindowSurface, list[ShadingSurface]]:
    """
    Construct the window and (optionally) overhang + 2 side fins for one
    exterior wall.
    """
    # ---- Wall-local frame
    A_top, A_bot, B_bot, B_top = wall.vertices
    u_vec = _sub(B_bot, A_bot)
    v_vec = _sub(A_top, A_bot)
    L_wall = _norm(u_vec)
    H_wall = _norm(v_vec)
    u_hat = _scale(u_vec, 1.0 / L_wall)
    v_hat = _scale(v_vec, 1.0 / H_wall)
    n_vec = _cross(u_hat, v_hat)            # already unit since u, v unit and orthogonal
    n_hat = n_vec                            # keep name explicit

    # ---- Window dimensions and position (wall-local)
    W_win = L_wall * math.sqrt(wwr)
    H_win = H_wall * math.sqrt(wwr)
    u_c = L_wall / 2.0
    v_c = H_wall / 2.0
    u_left  = u_c - W_win / 2.0
    u_right = u_c + W_win / 2.0
    v_top   = v_c + H_win / 2.0
    v_bot   = v_c - H_win / 2.0

    def to_3d(u: float, v: float, n: float = 0.0) -> Vertex:
        """Wall-local (u, v, n) -> 3D world coordinates."""
        return (
            A_bot[0] + u * u_hat[0] + v * v_hat[0] + n * n_hat[0],
            A_bot[1] + u * u_hat[1] + v * v_hat[1] + n * n_hat[1],
            A_bot[2] + u * u_hat[2] + v * v_hat[2] + n * n_hat[2],
        )

    # ---- Window vertices [UL, LL, LR, UR] (matches wall's vertex ordering)
    UL = to_3d(u_left,  v_top)
    LL = to_3d(u_left,  v_bot)
    LR = to_3d(u_right, v_bot)
    UR = to_3d(u_right, v_top)
    window = WindowSurface(
        name=f"{wall.name}_Window",
        parent_wall_name=wall.name,
        construction_name=construction_name,
        vertices=(UL, LL, LR, UR),
    )

    shading: list[ShadingSurface] = []

    # ---- Overhang (skip if shallower than MIN_SHADING_DEPTH)
    if overhang_depth >= MIN_SHADING_DEPTH:
        overhang = ShadingSurface(
            name=f"{wall.name}_Overhang",
            base_surface_name=wall.name,
            kind="Overhang",
            vertices=(
                to_3d(u_left,  v_top, 0.0),               # inside-left
                to_3d(u_left,  v_top, overhang_depth),    # outside-left
                to_3d(u_right, v_top, overhang_depth),    # outside-right
                to_3d(u_right, v_top, 0.0),               # inside-right
            ),
        )
        shading.append(overhang)

    # ---- Side fins (skip if shallower than MIN_SHADING_DEPTH)
    if sidefin_depth >= MIN_SHADING_DEPTH:
        # Left fin: outward normal = -u_hat
        left_fin = ShadingSurface(
            name=f"{wall.name}_SideFinLeft",
            base_surface_name=wall.name,
            kind="SideFinLeft",
            vertices=(
                to_3d(u_left, v_bot, 0.0),                # LL
                to_3d(u_left, v_bot, sidefin_depth),      # LL + s*n_hat
                to_3d(u_left, v_top, sidefin_depth),      # UL + s*n_hat
                to_3d(u_left, v_top, 0.0),                # UL
            ),
        )
        # Right fin: outward normal = +u_hat
        right_fin = ShadingSurface(
            name=f"{wall.name}_SideFinRight",
            base_surface_name=wall.name,
            kind="SideFinRight",
            vertices=(
                to_3d(u_right, v_bot, 0.0),               # LR
                to_3d(u_right, v_top, 0.0),               # UR
                to_3d(u_right, v_top, sidefin_depth),     # UR + s*n_hat
                to_3d(u_right, v_bot, sidefin_depth),     # LR + s*n_hat
            ),
        )
        shading.append(left_fin)
        shading.append(right_fin)

    return window, shading


# ---------- Validation ----------

def _validate(
    wwr_n: float, wwr_e: float, wwr_s: float, wwr_w: float,
    u: float, shgc: float, vt: float,
    oh_n: float, oh_e: float, oh_s: float, oh_w: float,
    sf: float,
) -> None:
    for name, val in [("wwr_north", wwr_n), ("wwr_east", wwr_e),
                      ("wwr_south", wwr_s), ("wwr_west", wwr_w)]:
        if not (0.0 < val < 1.0):
            raise ValueError(f"{name} must be in (0, 1), got {val}")
    if u <= 0:
        raise ValueError(f"glazing_u must be positive, got {u}")
    if not (0.0 < shgc <= 1.0):
        raise ValueError(f"glazing_shgc must be in (0, 1], got {shgc}")
    if not (0.0 < vt <= 1.0):
        raise ValueError(f"glazing_vt must be in (0, 1], got {vt}")
    for name, val in [("overhang_depth_n", oh_n), ("overhang_depth_e", oh_e),
                      ("overhang_depth_s", oh_s), ("overhang_depth_w", oh_w)]:
        if val < 0:
            raise ValueError(f"{name} must be >= 0, got {val}")
    if sf < 0:
        raise ValueError(f"sidefin_depth must be >= 0, got {sf}")


# ---------- Vector helpers ----------

def _sub(a: Vertex, b: Vertex) -> Vertex:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(v: Vertex, k: float) -> Vertex:
    return (v[0] * k, v[1] * k, v[2] * k)


def _norm(v: Vertex) -> float:
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def _cross(a: Vertex, b: Vertex) -> Vertex:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Vertex, b: Vertex) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _polygon_area_3d(vertices: tuple[Vertex, ...] | list[Vertex]) -> float:
    """Newell's method - area of a planar polygon in 3D."""
    n = len(vertices)
    if n < 3:
        return 0.0
    nx = ny = nz = 0.0
    for i in range(n):
        x1, y1, z1 = vertices[i]
        x2, y2, z2 = vertices[(i + 1) % n]
        nx += (y1 - y2) * (z1 + z2)
        ny += (z1 - z2) * (x1 + x2)
        nz += (x1 - x2) * (y1 + y2)
    return 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)


def _polygon_unit_normal(vertices: tuple[Vertex, ...] | list[Vertex]) -> Vertex:
    """Unit normal via Newell's method."""
    nx = ny = nz = 0.0
    n = len(vertices)
    for i in range(n):
        x1, y1, z1 = vertices[i]
        x2, y2, z2 = vertices[(i + 1) % n]
        nx += (y1 - y2) * (z1 + z2)
        ny += (z1 - z2) * (x1 + x2)
        nz += (x1 - x2) * (y1 + y2)
    mag = math.sqrt(nx * nx + ny * ny + nz * nz)
    if mag < 1e-12:
        return (0.0, 0.0, 0.0)
    return (nx / mag, ny / mag, nz / mag)
