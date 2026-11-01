"""
constructions.py - Material and construction (assembly) specifications.

This module translates the four sampled envelope parameters into a set of
EnergyPlus Material objects and Construction (assembly) objects. It is
independent of EnergyPlus / eppy: it produces pure data structures
(dataclasses) that generator.py later translates into IDF objects.

PARAMETERS HANDLED
------------------
  wall_r_value       (1.0 - 8.0  m^2.K/W) -> total R of exterior wall
  roof_r_value       (2.0 - 12.0 m^2.K/W) -> total R of roof
  floor_r_value      (0.5 - 5.0  m^2.K/W) -> R of the slab-on-grade insulation,
                                              converted to an F-factor (see below)
  wall_thermal_mass  (50 - 400   kJ/m^2.K) -> heat capacity of the concrete
                                              MASS LAYER in the wall (NOT
                                              the whole-wall capacity)

CONVENTIONS
-----------
R-value: Layer R = thickness / conductivity (m^2.K/W). Surface air-film
resistances (~0.04 outside, ~0.12 inside) are NOT part of the target
R-value; EnergyPlus computes them internally during simulation.

Layer order: outside -> inside, matching EnergyPlus's Construction object
convention. For interzone surfaces (matched pairs) EnergyPlus does NOT
reverse layers automatically: each side must name a construction whose
layers are listed from ITS OWN outside to inside, i.e. the two sides use
mirror-image constructions. Symmetric assemblies (InteriorWall) can share
one construction; asymmetric ones (the floor/ceiling slab) need a
reversed twin. EnergyPlus warns when a pair is not mirrored.

ASSEMBLIES PRODUCED
-------------------
  ExteriorWall:
      Stucco | Insulation(var) | Concrete(var) | Gypsum
      Mass wall with external insulation - the ASHRAE 90.1 baseline mass
      wall, used in the DOE Medium Office prototype. Concrete on the
      interior side maximizes the effect of thermal mass on indoor temp.

  Roof:
      Membrane | Insulation(var) | ConcreteDeck | AcousticTile
      Low-slope commercial roof. Roof mass is fixed (not parameterized).

  Ground floor: F-FACTOR slab-on-grade (Decision G1, 11 Oct 2026)
      Not a layered construction. As in the DOE Medium Office prototype,
      each ground-floor surface gets a Construction:FfactorGroundFloor
      (one per zone: F-factor, floor area, EXPOSED perimeter) with outside
      boundary condition GroundFCfactorMethod. Heat loss ~ F x exposed
      perimeter x (T_zone - T_ground): it scales with the slab EDGE, not its
      area, which is how large slabs actually lose heat. Core zones have no
      exposed perimeter -> EnergyPlus treats them as R ~ 177 m2K/W (near
      adiabatic; HeatBalanceManager.cc). Ground temperatures come from the
      EPW header (0.5 m) automatically because generator.py writes NO
      Site:GroundTemperature:FCfactorMethod object.
      floor_r_value -> F-factor: ffactor_for_floor_r() below.
      WHY (2.5 pilot): the previous layered slab + a constant 18 C ground in
      every climate made Colombo slabs lose heat to "cold" ground (heating
      fell with floor insulation, rho = -0.80) and hid Minneapolis' frozen
      winter ground (floor insulation had no effect, rho = +0.01).

  InteriorWall (symmetric):
      Gypsum | SoundInsulation | Gypsum
      Metal-stud partition between zones. Shared by matched pair.

  InteriorCeilingFloor (asymmetric) - used by FLOOR surfaces:
      AcousticTile | ConcreteDeck | Carpet
      Between vertically stacked floors. From the floor-surface
      perspective (outward normal -Z), "outside" is the acoustic tile
      below; the upper face (carpet) is the room floor.

  InteriorCeilingFloorReversed - used by CEILING surfaces:
      Carpet | ConcreteDeck | AcousticTile
      The same slab seen from the room below (outward normal +Z):
      "outside" is the carpet of the room above, the acoustic tile faces
      the room. Mirror of InteriorCeilingFloor, as EnergyPlus requires.

MATERIAL PROPERTIES
-------------------
All material thermal properties (k, density, cp) are taken from the
ASHRAE Handbook of Fundamentals (Chapter 26) and the EnergyPlus reference
data library. Specific values are tabulated in the constants section
below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------- Type aliases ----------

Roughness = Literal[
    "VeryRough", "Rough", "MediumRough",
    "MediumSmooth", "Smooth", "VerySmooth",
]


# ---------- Data classes ----------

@dataclass(frozen=True)
class MaterialSpec:
    """
    EnergyPlus 'Material' object specification (opaque solid layer).

    Properties
    ----------
    name : unique IDF object name
    roughness : one of EnergyPlus's 6 roughness categories (affects film
                coefficient calculation for exterior surfaces)
    thickness : m
    conductivity : W/m.K
    density : kg/m^3
    specific_heat : J/kg.K
    thermal_absorptance, solar_absorptance, visible_absorptance :
        long-wave, short-wave, and visible absorptance (0-1).
        Defaults represent typical interior/painted finishes.
    """
    name: str
    roughness: Roughness
    thickness: float
    conductivity: float
    density: float
    specific_heat: float
    thermal_absorptance: float = 0.9
    solar_absorptance: float = 0.7
    visible_absorptance: float = 0.7

    @property
    def r_value(self) -> float:
        """Layer thermal resistance, m^2.K/W."""
        return self.thickness / self.conductivity

    @property
    def heat_capacity(self) -> float:
        """Effective heat capacity per unit area, kJ/m^2.K."""
        return self.thickness * self.density * self.specific_heat / 1000.0


@dataclass(frozen=True)
class ConstructionSpec:
    """
    EnergyPlus 'Construction' object specification.

    Layers are listed outside to inside. For interzone surfaces, the
    layer order is interpreted from the perspective of the surface that
    owns the construction; the matched surface sees layers reversed.
    """
    name: str
    layers: tuple[str, ...]  # material names, outside -> inside

    @property
    def total_r_value(self) -> float:
        """Sum of layer R-values; lookup-requires the materials list."""
        # Caller can recompute via the materials list. Helper below.
        raise AttributeError(
            "Use compute_total_r() with the materials list, "
            "since ConstructionSpec stores names not refs."
        )


@dataclass(frozen=True)
class ConstructionSet:
    """
    Complete set of materials and constructions for one simulation.

    Named string fields point to the construction names that should be
    applied to surfaces of each type. generator.py uses
    get_construction_for_surface() to make the assignment.
    """
    materials: tuple[MaterialSpec, ...]
    constructions: tuple[ConstructionSpec, ...]

    # Named references for surface assignment
    exterior_wall: str
    interior_wall: str
    roof: str
    interior_ceiling_floor: str     # floor surfaces (bc=Surface)
    interior_ceiling: str           # ceiling surfaces (bc=Surface): mirror of the above
    ground_floor_ffactor: float     # W/m.K; per-zone constructions: ground_floor_construction_name()

    def material_by_name(self, name: str) -> MaterialSpec:
        for m in self.materials:
            if m.name == name:
                return m
        raise KeyError(f"Material {name!r} not in ConstructionSet")

    def construction_by_name(self, name: str) -> ConstructionSpec:
        for c in self.constructions:
            if c.name == name:
                return c
        raise KeyError(f"Construction {name!r} not in ConstructionSet")

    def total_r_value(self, construction_name: str) -> float:
        """Sum of layer R-values for the given construction."""
        c = self.construction_by_name(construction_name)
        return sum(self.material_by_name(m).r_value for m in c.layers)

    def total_heat_capacity(self, construction_name: str) -> float:
        """Sum of layer heat capacities (kJ/m^2.K) for the construction."""
        c = self.construction_by_name(construction_name)
        return sum(self.material_by_name(m).heat_capacity for m in c.layers)


# ---------- Fixed material constants ----------
# Sources: ASHRAE Handbook of Fundamentals Ch. 26, EnergyPlus reference
# data library (Materials.idf). Values are typical mid-range.

EXTERIOR_STUCCO = MaterialSpec(
    name="ExteriorStucco",
    roughness="Smooth",
    thickness=0.020,            # 2 cm
    conductivity=0.50,
    density=1500.0,
    specific_heat=870.0,
    solar_absorptance=0.70,     # medium-tone finish
)

INTERIOR_GYPSUM = MaterialSpec(
    name="InteriorGypsum",
    roughness="MediumSmooth",
    thickness=0.013,            # 1/2 inch
    conductivity=0.16,
    density=800.0,
    specific_heat=1090.0,
)

ROOF_MEMBRANE = MaterialSpec(
    name="RoofMembrane",
    roughness="MediumRough",
    thickness=0.010,
    conductivity=0.16,
    density=1100.0,
    specific_heat=1460.0,
    solar_absorptance=0.70,
)

STRUCTURAL_CONCRETE_DECK = MaterialSpec(
    name="StructuralConcreteDeck",
    roughness="MediumRough",
    thickness=0.100,            # 10 cm structural deck
    conductivity=1.31,
    density=2240.0,
    specific_heat=836.0,
)

ACOUSTIC_TILE = MaterialSpec(
    name="AcousticTile",
    roughness="MediumSmooth",
    thickness=0.020,
    conductivity=0.06,
    density=368.0,
    specific_heat=590.0,
)

CARPET = MaterialSpec(
    name="Carpet",
    roughness="VeryRough",
    thickness=0.010,
    conductivity=0.06,
    density=288.0,
    specific_heat=1380.0,
)

INTERIOR_SOUND_INSULATION = MaterialSpec(
    name="InteriorSoundInsulation",
    roughness="MediumRough",
    thickness=0.092,            # fits standard 4-inch (92 mm) stud cavity
    conductivity=0.045,
    density=30.0,
    specific_heat=1000.0,
)

# Variable-thickness materials use these properties; only thickness changes.
INSULATION_K = 0.040            # W/m.K - generic fibrous insulation
INSULATION_DENSITY = 30.0       # kg/m^3
INSULATION_CP = 1000.0          # J/kg.K

WALL_CONCRETE_K = 1.31
WALL_CONCRETE_DENSITY = 2240.0
WALL_CONCRETE_CP = 836.0

# Stability floor for the variable concrete layer.
MIN_CONCRETE_THICKNESS = 0.005  # 5 mm


# ---------- Slab-on-grade F-factor (Decision G1) ----------
# ASHRAE 90.1-2019 Normative Appendix A, Table A6.3.1, UNHEATED slab,
# "fully insulated slab" row (continuous insulation under the whole slab and
# at its edge - the configuration this generator always modelled), plus the
# uninsulated value. IP units: R in h.ft2.F/Btu, F in Btu/h.ft.F.
# Verified against the ENERGY STAR F-Factor Guide (2023), which reproduces
# Table A6.3.1, and the IECC/ICC slab tables (uninsulated F-0.73).
FFACTOR_TABLE_IP: tuple[tuple[float, float], ...] = (
    (0.0, 0.730),
    (5.0, 0.460),
    (7.5, 0.410),
    (10.0, 0.360),
    (15.0, 0.300),
    (20.0, 0.261),
)
R_SI_TO_IP = 5.678263          # m2.K/W -> h.ft2.F/Btu
F_IP_TO_SI = 1.730735          # Btu/h.ft.F -> W/m.K


def ffactor_for_floor_r(floor_r_value: float) -> float:
    """F-factor (W/m.K) of a fully insulated unheated slab whose insulation has
    R = floor_r_value (m2.K/W).

    Interpolation is linear in 1/F (a resistance) between table points, which
    is how added insulation behaves (resistances in series) and keeps F
    strictly decreasing. Beyond the table's R-20 IP (3.52 m2.K/W) the last
    segment's slope is extended: floor_r_value up to 5.0 m2.K/W (R-28.4 IP)
    is EXTRAPOLATED - documented assumption.
    """
    if floor_r_value <= 0:
        raise ValueError(f"floor_r_value must be positive, got {floor_r_value}")
    r_ip = floor_r_value * R_SI_TO_IP
    pts = FFACTOR_TABLE_IP
    for (r0, f0), (r1, f1) in zip(pts, pts[1:]):
        if r_ip <= r1:
            break
    inv = 1.0 / f0 + (1.0 / f1 - 1.0 / f0) * (r_ip - r0) / (r1 - r0)
    return F_IP_TO_SI / inv


GROUND_FLOOR_PREFIX = "GroundFloor"


def ground_floor_construction_name(zone_name: str) -> str:
    """One F-factor construction per ground-floor zone (area/perimeter differ)."""
    return f"{GROUND_FLOOR_PREFIX}_{zone_name}"


# ---------- Top-level API ----------

def build_constructions(
    wall_r_value: float,
    roof_r_value: float,
    floor_r_value: float,
    wall_thermal_mass: float,
) -> ConstructionSet:
    """
    Build the complete material + construction set from the 4 envelope
    parameters.

    Parameters
    ----------
    wall_r_value : target R for the exterior wall assembly (m^2.K/W).
        Sum of material layer resistances, excluding surface air films.
    roof_r_value : target R for the roof assembly.
    floor_r_value : R of the slab insulation; becomes ground_floor_ffactor.
    wall_thermal_mass : effective heat capacity of the wall's concrete
        MASS LAYER ONLY (kJ/m^2.K). The whole wall's heat capacity will
        be approximately this value + 37 kJ/m^2.K (contribution from
        stucco + gypsum). See module docstring.

    Returns
    -------
    ConstructionSet with materials, constructions, and surface-assignment
    references.

    Raises
    ------
    ValueError if any input is non-positive, or if an R-value is so low
    that the fixed (non-insulation) layers alone exceed it.
    """
    _validate_inputs(wall_r_value, roof_r_value, floor_r_value, wall_thermal_mass)

    # --- Step 1: variable-thickness wall concrete (mass layer)
    wall_concrete = _concrete_layer_for_mass(
        target_mass=wall_thermal_mass,
        name="WallConcrete",
    )

    # --- Step 2: variable-thickness insulations
    # Compute fixed R contributions BEFORE adding insulation
    r_fixed_wall = (
        EXTERIOR_STUCCO.r_value
        + wall_concrete.r_value
        + INTERIOR_GYPSUM.r_value
    )
    wall_insulation = _insulation_layer_for_r_value(
        target_r=wall_r_value,
        r_other_layers=r_fixed_wall,
        name="WallInsulation",
    )

    r_fixed_roof = (
        ROOF_MEMBRANE.r_value
        + STRUCTURAL_CONCRETE_DECK.r_value
        + ACOUSTIC_TILE.r_value
    )
    roof_insulation = _insulation_layer_for_r_value(
        target_r=roof_r_value,
        r_other_layers=r_fixed_roof,
        name="RoofInsulation",
    )

    ground_floor_ffactor = ffactor_for_floor_r(floor_r_value)

    # --- Step 3: assemble material list (unique names)
    materials: tuple[MaterialSpec, ...] = (
        # Wall layers
        EXTERIOR_STUCCO,
        wall_insulation,
        wall_concrete,
        INTERIOR_GYPSUM,
        # Roof layers
        ROOF_MEMBRANE,
        roof_insulation,
        STRUCTURAL_CONCRETE_DECK,   # shared with interior floor/ceiling
        ACOUSTIC_TILE,              # shared with interior floor/ceiling
        CARPET,                     # interior floor/ceiling
        # Interior partition
        INTERIOR_SOUND_INSULATION,
    )

    # --- Step 4: assemble constructions
    constructions: tuple[ConstructionSpec, ...] = (
        ConstructionSpec(
            name="ExteriorWall",
            layers=(
                EXTERIOR_STUCCO.name,
                wall_insulation.name,
                wall_concrete.name,
                INTERIOR_GYPSUM.name,
            ),
        ),
        ConstructionSpec(
            name="InteriorWall",
            layers=(
                INTERIOR_GYPSUM.name,
                INTERIOR_SOUND_INSULATION.name,
                INTERIOR_GYPSUM.name,
            ),
        ),
        ConstructionSpec(
            name="Roof",
            layers=(
                ROOF_MEMBRANE.name,
                roof_insulation.name,
                STRUCTURAL_CONCRETE_DECK.name,
                ACOUSTIC_TILE.name,
            ),
        ),
        ConstructionSpec(
            name="InteriorCeilingFloor",
            # outside (bottom of floor surface) -> inside (top, room floor)
            layers=(
                ACOUSTIC_TILE.name,
                STRUCTURAL_CONCRETE_DECK.name,
                CARPET.name,
            ),
        ),
        ConstructionSpec(
            name="InteriorCeilingFloorReversed",
            # the same slab from the room below: outside (carpet above) -> inside (tile)
            layers=(
                CARPET.name,
                STRUCTURAL_CONCRETE_DECK.name,
                ACOUSTIC_TILE.name,
            ),
        ),
    )

    return ConstructionSet(
        materials=materials,
        constructions=constructions,
        exterior_wall="ExteriorWall",
        interior_wall="InteriorWall",
        roof="Roof",
        interior_ceiling_floor="InteriorCeilingFloor",
        interior_ceiling="InteriorCeilingFloorReversed",
        ground_floor_ffactor=ground_floor_ffactor,
    )


# ---------- Surface-to-construction mapping ----------

def get_construction_for_surface(
    surface_type: str,        # geometry.SurfaceType: Floor, Ceiling, Roof, Wall
    boundary_condition: str,  # geometry.BoundaryCondition
    construction_set: ConstructionSet,
) -> str:
    """
    Decide which construction to assign to a Surface, given its type
    and boundary condition. Centralizes the mapping so generator.py
    doesn't hard-code it.
    """
    st = surface_type
    bc = boundary_condition

    if st == "Wall" and bc == "Outdoors":
        return construction_set.exterior_wall
    if st == "Wall" and bc == "Surface":
        return construction_set.interior_wall
    if st == "Roof":
        # Roof always has Outdoors BC; we don't double-check here.
        return construction_set.roof
    if st == "Ceiling" and bc == "Surface":
        return construction_set.interior_ceiling
    if st == "Floor" and bc == "Surface":
        return construction_set.interior_ceiling_floor
    if st == "Floor" and bc in ("Ground", "GroundFCfactorMethod"):
        raise ValueError("ground floors use one F-factor construction per zone: "
                         "ground_floor_construction_name(zone_name)")

    raise ValueError(
        f"No construction defined for surface_type={st!r} with "
        f"boundary_condition={bc!r}"
    )


# ---------- Helpers ----------

def _validate_inputs(
    wall_r_value: float,
    roof_r_value: float,
    floor_r_value: float,
    wall_thermal_mass: float,
) -> None:
    if wall_r_value <= 0:
        raise ValueError(f"wall_r_value must be positive, got {wall_r_value}")
    if roof_r_value <= 0:
        raise ValueError(f"roof_r_value must be positive, got {roof_r_value}")
    if floor_r_value <= 0:
        raise ValueError(f"floor_r_value must be positive, got {floor_r_value}")
    if wall_thermal_mass <= 0:
        raise ValueError(
            f"wall_thermal_mass must be positive, got {wall_thermal_mass}"
        )


def _concrete_layer_for_mass(target_mass: float, name: str) -> MaterialSpec:
    """
    Compute concrete thickness (m) so that the layer's effective heat
    capacity matches the target.

    C [kJ/m^2.K] = thickness [m] * density [kg/m^3] * cp [J/kg.K] / 1000
    => thickness = C * 1000 / (density * cp)
    """
    thickness = (target_mass * 1000.0) / (WALL_CONCRETE_DENSITY * WALL_CONCRETE_CP)
    thickness = max(thickness, MIN_CONCRETE_THICKNESS)
    return MaterialSpec(
        name=name,
        roughness="MediumRough",
        thickness=thickness,
        conductivity=WALL_CONCRETE_K,
        density=WALL_CONCRETE_DENSITY,
        specific_heat=WALL_CONCRETE_CP,
    )


def _insulation_layer_for_r_value(
    target_r: float,
    r_other_layers: float,
    name: str,
) -> MaterialSpec:
    """
    Compute insulation thickness so the total construction R-value
    (excluding air films) matches target_r.
    """
    r_insulation = target_r - r_other_layers
    if r_insulation <= 0:
        raise ValueError(
            f"Target R-value {target_r:.3f} is less than the sum of "
            f"non-insulation layer R-values ({r_other_layers:.3f}) "
            f"for {name}. Cannot achieve target with positive insulation."
        )
    thickness = r_insulation * INSULATION_K
    return MaterialSpec(
        name=name,
        roughness="MediumRough",
        thickness=thickness,
        conductivity=INSULATION_K,
        density=INSULATION_DENSITY,
        specific_heat=INSULATION_CP,
    )
