"""
geometry.py — Geometric model for the 5-zone perimeter-plus-core building.

This module computes 3D coordinates for every surface in a parameterized
multi-floor building. It produces pure data structures (dataclasses) that
are independent of EnergyPlus / eppy. The generator translates these into
IDF objects.

LAYOUT
------
DOE Medium Office trapezoidal 5-zone layout. Each floor has 5 zones:
Core (central rectangle) and 4 perimeter trapezoids (N, E, S, W) meeting
at 45 degree diagonals at the building corners.

       (0,W) ─────────────── (L,W)
         │  \\               /  │
         │    \\  N perim  /    │
         │     \\         /     │
         │   (pd,W-pd)──(L-pd,W-pd)
         │ W   │            │   E │
         │ p   │   Core     │   p │
         │     │            │     │
         │   (pd,pd)────(L-pd,pd)
         │     /           \\    │
         │    /  S perim    \\   │
         │  /                \\  │
       (0,0) ─────────────── (L,0)

  pd = perimeter depth (4.57 m, DOE standard, = 15 ft)
  L  = building length along +X (east-west)
  W  = building width along +Y (north-south)
  L = aspect_ratio * W,  L * W = floor_area_per_floor.

COORDINATE SYSTEM
-----------------
Right-handed Cartesian:  +X east, +Y north, +Z up.
Origin: SW corner of the building footprint at ground level.

Building rotation (orientation) is applied at the IDF level via the
Building object's `North Axis` field — NOT in this module. This module
always produces a building with its long axis along +X.

VERTEX ORDERING
---------------
EnergyPlus GlobalGeometryRules will be set to:
    StartingVertexPosition = UpperLeftCorner
    VertexEntryDirection   = Counterclockwise
    CoordinateSystem       = World

The right-hand rule applied to (V2-V1) x (V3-V1) must yield the OUTWARD
normal (away from the zone interior).

Rules used here (all footprints stored CCW from above):
  - Floor surface:   reverse of footprint order at z=z_bottom  (normal -Z)
  - Ceiling/Roof:    footprint order at z=z_top                (normal +Z)
  - Wall surface:    for edge V_i -> V_{i+1} walking the footprint CCW,
                     vertex order is [A_top, A_bot, B_bot, B_top]
                     where A=V_i, B=V_{i+1}. Yields outward normal.

ADJACENCY
---------
Every interior wall is shared between exactly two zones. Walking the
footprint CCW means a shared edge is walked in OPPOSITE directions by
the two zones — which is exactly the convention EnergyPlus expects for
matched interior surfaces (same 4 corners listed in reverse order).

The WALL_ADJACENCY table below encodes, for each (zone_role, wall_index),
either ("__Outdoors__", cardinal_direction) for exterior walls or
(other_zone_role, other_wall_index) for interior walls.

Vertical (floor/ceiling) adjacency is resolved separately by matching
each non-bottom floor surface to the ceiling of the same zone role on
the floor below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

# ---------- Type aliases ----------

Vertex = tuple[float, float, float]  # (x, y, z) in meters

SurfaceType = Literal["Floor", "Ceiling", "Roof", "Wall"]
BoundaryCondition = Literal["Outdoors", "Ground", "Surface", "Adiabatic"]
CardinalDirection = Literal["North", "East", "South", "West"]
ZoneRole = Literal["Core", "North", "East", "South", "West"]

# Sentinel used in WALL_ADJACENCY to mean "this wall is exterior".
_EXTERIOR = "__Outdoors__"


# ---------- Data classes ----------

@dataclass
class Surface:
    """One planar surface of a zone (floor, ceiling, roof, or single wall)."""
    name: str
    zone_name: str
    surface_type: SurfaceType
    vertices: list[Vertex]
    boundary_condition: BoundaryCondition
    # Name of the matched surface when boundary_condition == "Surface".
    # Filled in by _resolve_adjacencies().
    outside_boundary_object: str | None = None
    # For exterior walls: which cardinal direction the OUTSIDE faces.
    # None for floors, ceilings, roofs, interior walls, and diagonals.
    direction: CardinalDirection | None = None

    @property
    def area(self) -> float:
        """Area of the planar polygon, via Newell's method (in m^2)."""
        return _polygon_area_3d(self.vertices)


@dataclass
class Zone:
    """A single thermal zone."""
    name: str
    floor_index: int
    role: ZoneRole
    footprint: list[Vertex]    # at z = z_bottom, CCW from above
    floor_area: float          # m^2
    floor_height: float        # m
    volume: float              # m^3
    surfaces: list[Surface] = field(default_factory=list)

    @property
    def exterior_walls(self) -> list[Surface]:
        return [s for s in self.surfaces
                if s.surface_type == "Wall" and s.boundary_condition == "Outdoors"]

    @property
    def exposed_perimeter(self) -> float:
        """Length (m) of the zone's footprint edges that are exterior walls -
        the 'PerimeterExposed' of an F-factor ground floor. 0 for core zones."""
        total = 0.0
        for w in self.exterior_walls:
            # wall vertices are [A_top, A_bottom, B_bottom, B_top]
            (ax, ay, _), (bx, by, _) = w.vertices[1], w.vertices[2]
            total += ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
        return total


@dataclass
class BuildingGeometry:
    """Complete geometric description of the building."""
    length: float          # m, along +X
    width: float           # m, along +Y
    n_floors: int
    floor_height: float
    perim_depth: float
    zones: list[Zone]

    @property
    def floor_area_per_floor(self) -> float:
        return sum(z.floor_area for z in self.zones if z.floor_index == 0)

    @property
    def total_floor_area(self) -> float:
        return sum(z.floor_area for z in self.zones)

    @property
    def building_height(self) -> float:
        return self.n_floors * self.floor_height


# ---------- Adjacency table ----------
# Hand-derived from the trapezoidal footprint ordering. See module docstring.
# For each (zone_role, wall_index):
#   ( _EXTERIOR, cardinal_direction )  -> exterior wall
#   ( other_zone_role, other_wall_index ) -> interior wall, matched pair

WALL_ADJACENCY: dict[tuple[ZoneRole, int],
                     tuple[str, CardinalDirection] | tuple[ZoneRole, int]] = {
    # Core - all 4 walls interior, one to each perimeter zone.
    ("Core",  0): ("South", 2),
    ("Core",  1): ("East",  3),
    ("Core",  2): ("North", 0),
    ("Core",  3): ("West",  1),

    # South perimeter
    ("South", 0): (_EXTERIOR, "South"),   # outer south facade
    ("South", 1): ("East",   0),          # SE diagonal
    ("South", 2): ("Core",   0),
    ("South", 3): ("West",   0),          # SW diagonal

    # East perimeter
    ("East",  0): ("South",  1),          # SE diagonal
    ("East",  1): (_EXTERIOR, "East"),    # outer east facade
    ("East",  2): ("North",  1),          # NE diagonal
    ("East",  3): ("Core",   1),

    # North perimeter
    ("North", 0): ("Core",   2),
    ("North", 1): ("East",   2),          # NE diagonal
    ("North", 2): (_EXTERIOR, "North"),   # outer north facade
    ("North", 3): ("West",   2),          # NW diagonal

    # West perimeter
    ("West",  0): ("South",  3),          # SW diagonal
    ("West",  1): ("Core",   3),
    ("West",  2): ("North",  3),          # NW diagonal
    ("West",  3): (_EXTERIOR, "West"),    # outer west facade
}

ZONE_ORDER: list[ZoneRole] = ["Core", "South", "East", "North", "West"]


# ---------- Top-level API ----------

def build_geometry(
    floor_area: float,
    aspect_ratio: float,
    n_floors: int,
    floor_height: float,
    perim_depth: float = 4.57,
) -> BuildingGeometry:
    """
    Construct the full BuildingGeometry from high-level parameters.

    Parameters
    ----------
    floor_area : m^2, area of ONE floor (not the whole building)
    aspect_ratio : L/W, must be >= 1
    n_floors : integer, 1..5
    floor_height : m, floor-to-floor height
    perim_depth : m, depth of perimeter zones. DOE default 4.57 m (15 ft)

    Returns
    -------
    BuildingGeometry with all zones, surfaces, and matched boundary
    conditions resolved.
    """
    if not isinstance(n_floors, int) or n_floors < 1:
        raise ValueError(f"n_floors must be an integer >= 1, got {n_floors!r}")
    if floor_height <= 0:
        raise ValueError(f"floor_height must be positive, got {floor_height}")
    if perim_depth <= 0:
        raise ValueError(f"perim_depth must be positive, got {perim_depth}")

    length, width = compute_building_dimensions(floor_area, aspect_ratio)
    footprints = compute_zone_footprints(length, width, perim_depth)

    zones: list[Zone] = []
    for floor_idx in range(n_floors):
        z_bot = floor_idx * floor_height
        z_top = z_bot + floor_height
        for role in ZONE_ORDER:
            zone = _build_zone(
                role=role,
                floor_index=floor_idx,
                footprint_at_origin=footprints[role],
                z_bottom=z_bot,
                z_top=z_top,
                n_floors=n_floors,
            )
            zones.append(zone)

    _resolve_adjacencies(zones)

    return BuildingGeometry(
        length=length,
        width=width,
        n_floors=n_floors,
        floor_height=floor_height,
        perim_depth=perim_depth,
        zones=zones,
    )


# ---------- Building dimensions ----------

def compute_building_dimensions(floor_area: float,
                                aspect_ratio: float) -> tuple[float, float]:
    """
    Solve L*W = area, L/W = AR for (L, W).

    Returns
    -------
    (length_along_x, width_along_y) in meters.
    """
    if floor_area <= 0:
        raise ValueError(f"floor_area must be positive, got {floor_area}")
    if aspect_ratio < 1.0:
        raise ValueError(
            f"aspect_ratio must be >= 1 (length:width). Got {aspect_ratio}. "
            "Rotation by 90 deg should be expressed via orientation, not by "
            "swapping L and W."
        )
    width = math.sqrt(floor_area / aspect_ratio)
    length = aspect_ratio * width
    return length, width


# ---------- Zone footprints (2D layout at z=0) ----------

def compute_zone_footprints(
    length: float,
    width: float,
    perim_depth: float,
) -> dict[ZoneRole, list[Vertex]]:
    """
    Return the 4-vertex footprint of each of the 5 zones, listed CCW
    when viewed from above (+Z). All z-coordinates are 0.

    Raises
    ------
    ValueError if perim_depth is too large for the building footprint.
    """
    L, W, pd = length, width, perim_depth

    if L - 2 * pd <= 0 or W - 2 * pd <= 0:
        raise ValueError(
            f"Building footprint too small for perimeter depth {pd:.2f} m. "
            f"Got L={L:.2f}, W={W:.2f}. Need both L and W > {2*pd:.2f} m."
        )

    # All footprints listed CCW from above; verified by shoelace signed
    # area > 0 in the test suite.
    return {
        "Core":  [(pd,    pd,    0.0),
                  (L-pd,  pd,    0.0),
                  (L-pd,  W-pd,  0.0),
                  (pd,    W-pd,  0.0)],

        "South": [(0.0,   0.0,   0.0),
                  (L,     0.0,   0.0),
                  (L-pd,  pd,    0.0),
                  (pd,    pd,    0.0)],

        "East":  [(L-pd,  pd,    0.0),
                  (L,     0.0,   0.0),
                  (L,     W,     0.0),
                  (L-pd,  W-pd,  0.0)],

        "North": [(pd,    W-pd,  0.0),
                  (L-pd,  W-pd,  0.0),
                  (L,     W,     0.0),
                  (0.0,   W,     0.0)],

        "West":  [(0.0,   0.0,   0.0),
                  (pd,    pd,    0.0),
                  (pd,    W-pd,  0.0),
                  (0.0,   W,     0.0)],
    }


# ---------- Zone construction ----------

def _zone_name(floor_index: int, role: ZoneRole) -> str:
    return f"Zone_F{floor_index}_{role}"


def _surface_name(zone_name: str, kind: str) -> str:
    return f"{zone_name}_{kind}"


def _build_zone(
    role: ZoneRole,
    floor_index: int,
    footprint_at_origin: list[Vertex],
    z_bottom: float,
    z_top: float,
    n_floors: int,
) -> Zone:
    """Build one Zone with its floor, ceiling/roof, and 4 wall surfaces."""

    footprint = [(x, y, z_bottom) for (x, y, _) in footprint_at_origin]
    floor_height = z_top - z_bottom
    floor_area = _polygon_area_2d(footprint)
    volume = floor_area * floor_height
    zone_name = _zone_name(floor_index, role)

    is_bottom_floor = (floor_index == 0)
    is_top_floor = (floor_index == n_floors - 1)

    # Floor surface: reversed footprint order at z_bottom -> outward normal -Z.
    floor_surface = Surface(
        name=_surface_name(zone_name, "Floor"),
        zone_name=zone_name,
        surface_type="Floor",
        vertices=list(reversed(footprint)),
        boundary_condition="Ground" if is_bottom_floor else "Surface",
    )

    # Ceiling/Roof: footprint order at z_top -> outward normal +Z.
    ceiling_verts: list[Vertex] = [(x, y, z_top) for (x, y, _) in footprint]
    if is_top_floor:
        ceiling_surface = Surface(
            name=_surface_name(zone_name, "Roof"),
            zone_name=zone_name,
            surface_type="Roof",
            vertices=ceiling_verts,
            boundary_condition="Outdoors",
        )
    else:
        ceiling_surface = Surface(
            name=_surface_name(zone_name, "Ceiling"),
            zone_name=zone_name,
            surface_type="Ceiling",
            vertices=ceiling_verts,
            boundary_condition="Surface",
        )

    # Walls: 4 walls, one per consecutive footprint edge.
    walls: list[Surface] = []
    n = len(footprint)
    for i in range(n):
        a_low = footprint[i]
        b_low = footprint[(i + 1) % n]
        a_top: Vertex = (a_low[0], a_low[1], z_top)
        b_top: Vertex = (b_low[0], b_low[1], z_top)
        # Order [A_top, A_bot, B_bot, B_top] -> outward normal.
        verts = [a_top, a_low, b_low, b_top]

        adj = WALL_ADJACENCY[(role, i)]
        if adj[0] == _EXTERIOR:
            direction: CardinalDirection | None = adj[1]  # type: ignore[assignment]
            bc: BoundaryCondition = "Outdoors"
        else:
            direction = None
            bc = "Surface"

        walls.append(Surface(
            name=_surface_name(zone_name, f"Wall{i}"),
            zone_name=zone_name,
            surface_type="Wall",
            vertices=verts,
            boundary_condition=bc,
            direction=direction,
        ))

    return Zone(
        name=zone_name,
        floor_index=floor_index,
        role=role,
        footprint=footprint,
        floor_area=floor_area,
        floor_height=floor_height,
        volume=volume,
        surfaces=[floor_surface, ceiling_surface] + walls,
    )


# ---------- Adjacency resolution ----------

def _resolve_adjacencies(zones: list[Zone]) -> None:
    """
    Mutate `outside_boundary_object` on every interior surface to point
    to its matched counterpart.
    """
    by_floor_role: dict[tuple[int, ZoneRole], Zone] = {
        (z.floor_index, z.role): z for z in zones
    }

    for zone in zones:
        # --- Vertical adjacency: floor of zone F_n <-> ceiling of zone F_{n-1}
        if zone.floor_index > 0:
            zone_below = by_floor_role[(zone.floor_index - 1, zone.role)]
            this_floor = next(s for s in zone.surfaces
                              if s.surface_type == "Floor")
            below_ceiling = next(s for s in zone_below.surfaces
                                 if s.surface_type == "Ceiling")
            this_floor.outside_boundary_object = below_ceiling.name
            below_ceiling.outside_boundary_object = this_floor.name

        # --- Horizontal adjacency: inter-zone walls on the same floor
        wall_surfaces = [s for s in zone.surfaces if s.surface_type == "Wall"]
        for wall_idx, wall in enumerate(wall_surfaces):
            adj = WALL_ADJACENCY[(zone.role, wall_idx)]
            if adj[0] == _EXTERIOR:
                continue
            other_role, other_wall_idx = adj  # type: ignore[misc]
            other_zone = by_floor_role[(zone.floor_index, other_role)]
            other_walls = [s for s in other_zone.surfaces
                           if s.surface_type == "Wall"]
            wall.outside_boundary_object = other_walls[other_wall_idx].name


# ---------- Geometric utilities ----------

def _polygon_area_2d(vertices: list[Vertex]) -> float:
    """Shoelace area for a polygon, ignoring the z-coordinate (m^2)."""
    n = len(vertices)
    a = 0.0
    for i in range(n):
        x1, y1, _ = vertices[i]
        x2, y2, _ = vertices[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def _polygon_signed_area_2d(vertices: list[Vertex]) -> float:
    """Signed shoelace area. > 0 if CCW from +Z view, < 0 if CW."""
    n = len(vertices)
    a = 0.0
    for i in range(n):
        x1, y1, _ = vertices[i]
        x2, y2, _ = vertices[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return a / 2.0


def _polygon_area_3d(vertices: list[Vertex]) -> float:
    """Area of a planar polygon in 3D, via Newell's method."""
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


def _polygon_normal(vertices: list[Vertex]) -> tuple[float, float, float]:
    """Unit outward normal of a planar polygon, via Newell's method."""
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


def _polygon_centroid(vertices: list[Vertex]) -> Vertex:
    n = len(vertices)
    return (
        sum(v[0] for v in vertices) / n,
        sum(v[1] for v in vertices) / n,
        sum(v[2] for v in vertices) / n,
    )


def _zone_centroid(zone: Zone) -> Vertex:
    """Approximate centroid of the zone (footprint centroid at mid-height)."""
    fp = zone.footprint
    cx = sum(v[0] for v in fp) / len(fp)
    cy = sum(v[1] for v in fp) / len(fp)
    cz = fp[0][2] + zone.floor_height / 2
    return (cx, cy, cz)
