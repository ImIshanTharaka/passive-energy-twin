"""
generator.py - Parametric IDF generator (orchestrator).

This is the ONLY module that depends on eppy. It consumes the pure-data
specifications from the other six modules and translates them into a
complete, runnable EnergyPlus IDF.

PUBLIC API
----------
    build_idf(params: dict, idd_path: str) -> eppy.modeleditor.IDF

    params: flat dict of all 22 sampled design-space parameters.
    idd_path: path to the EnergyPlus IDD file (e.g.,
              '/Applications/EnergyPlus-24-1-0/Energy+.idd').

    Returns an eppy IDF object. Call idf.saveas(path) to write to disk.

ASSEMBLY SEQUENCE
-----------------
  1. Skeleton IDF (Version, SimulationControl, Building, Timestep, ...)
  2. Geometry    -> Zone + BuildingSurface:Detailed
  3. Constructions -> Material + Construction
  4. Fenestration  -> WindowMaterial:SimpleGlazingSystem +
                      Construction + FenestrationSurface:Detailed +
                      Shading:Zone:Detailed
  5. Internal loads -> ScheduleTypeLimits + Schedule:Compact +
                       People + Lights + ElectricEquipment +
                       ZoneInfiltration:DesignFlowRate
  6. HVAC          -> HVACTemplate:Thermostat +
                      HVACTemplate:Zone:IdealLoadsAirSystem
  7. Outputs       -> Output:Variable + Output:Meter +
                      Output:Table:SummaryReports +
                      Output:SQLite + OutputControl:Table:Style

DESIGN NOTES
------------
- The generator does NO physics or geometry computation. All that
  work is done in the spec modules. This module is purely a translator.
- Each spec type has a dedicated _add_* helper. The helpers are called
  in the order above and mutate the IDF object in place.
- n_floors is rounded to the nearest integer if not already int.
- orientation is applied via Building.North_Axis.
- infiltration_ach (ACH at 50 Pa) is converted to m^3/s per m^2 of
  exterior surface for EnergyPlus's ZoneInfiltration object, using
  zone volume and a standard pressure-corrected flow coefficient.
"""

from __future__ import annotations

from io import StringIO
from typing import Any

from eppy.modeleditor import IDF

from ..geometry.geometry import BuildingGeometry, Zone, Surface, build_geometry
from ..construction.constructions import (
    ConstructionSet, MaterialSpec, ConstructionSpec,
    build_constructions, get_construction_for_surface, ground_floor_construction_name,
)
from ..fenestration.fenestration import (
    FenestrationSet, GlazingSpec, WindowSurface, ShadingSurface,
    build_fenestration,
)
from ..internal_loads.internal_loads import InternalLoadsSet, build_internal_loads
from ..hvac.hvac import HVACSet, ThermostatSpec, IdealLoadsZoneSpec, build_hvac
from ..outputs.outputs import OutputSet, build_outputs


# ============================================================
# Constants
# ============================================================

# Minimal IDF seed — eppy requires at least a Version object to
# initialize. build_idf() overwrites it with the version read from the
# IDD header, so the IDF always matches the EnergyPlus that runs it.
_SEED_IDF = """\
Version,
    24.1;
"""

# Oldest EnergyPlus whose field names this generator uses. EnergyPlus 9.6
# introduced Spaces and renamed, e.g., Zone_or_ZoneList_Name ->
# Zone_or_ZoneList_or_Space_or_SpaceList_Name and Watts_per_Zone_Floor_Area
# -> Watts_per_Floor_Area. Target and tested version: 24.1.
MIN_EPLUS_VERSION = (9, 6)

# EnergyPlus timestep (timesteps per hour). 6 = 10-minute intervals.
TIMESTEPS_PER_HOUR = 6

# ACH50 to natural infiltration conversion factor.
# The ACH50 value from the design space is measured at 50 Pa. Real
# operating infiltration at natural pressure (~4 Pa) is approximately
# ACH50 / N, where N is a divisor. The Sherman-Grimsrud correlation
# and LBL model typically use N ~ 20 for commercial buildings.
# So: ACH_natural ≈ ACH50 / 20.
# EnergyPlus ZoneInfiltration uses flow rate per zone floor area or
# per exterior surface area or ACH. We use AirChanges/Hour method
# with the natural ACH value.
ACH50_TO_NATURAL_DIVISOR = 20.0

# Ground floors (Decision G1, 11 Oct 2026): F-factor slab-on-grade, one
# Construction:FfactorGroundFloor per ground-floor zone, outside boundary
# GroundFCfactorMethod. NO Site:GroundTemperature:* object is written on
# purpose: without Site:GroundTemperature:FCfactorMethod EnergyPlus uses the
# 0.5 m ground temperatures from the EPW header (SiteFCFactorMethodGround
# Temperatures.cc), so ground temperature follows the climate automatically.
# The former constant 18 C Site:GroundTemperature:BuildingSurface was wrong
# for every climate (2.5 pilot, check G).
GROUND_BC_EPLUS = "GroundFCfactorMethod"


# ============================================================
# Public API
# ============================================================

def _idd_version(idd_path: str) -> str:
    """Return 'MAJOR.MINOR' from the IDD header line '!IDD_Version 24.1.0'.

    EnergyPlus treats a Version mismatch between IDF and program as fatal,
    so the IDF version is taken from the IDD actually used, never hardcoded.
    Raises ValueError for a malformed header or an IDD older than
    MIN_EPLUS_VERSION.
    """
    with open(idd_path, encoding="latin-1") as handle:
        first = handle.readline().strip()
    if not first.startswith("!IDD_Version"):
        raise ValueError(f"{idd_path}: first line is not '!IDD_Version ...': {first!r}")
    try:
        major, minor = (int(x) for x in first.split()[1].split(".")[:2])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"{idd_path}: cannot parse version from {first!r}") from exc
    if (major, minor) < MIN_EPLUS_VERSION:
        raise ValueError(
            f"IDD is EnergyPlus {major}.{minor}; this generator needs "
            f"{MIN_EPLUS_VERSION[0]}.{MIN_EPLUS_VERSION[1]} or newer (target 24.1), "
            f"because it uses the Space-aware field names introduced in 9.6."
        )
    return f"{major}.{minor}"


_KEY_CACHE: dict[tuple[str, int], frozenset[str]] = {}


def _lowered(obj_type: str, index: int, keys: list[str]) -> frozenset[str]:
    """Lower-cased allowed values for one field, cached per (object type, position)."""
    cache_key = (obj_type, index)
    if cache_key not in _KEY_CACHE:
        _KEY_CACHE[cache_key] = frozenset(k.lower() for k in keys)
    return _KEY_CACHE[cache_key]


def _validate_choices(idf: IDF) -> None:
    """Reject any option value not allowed by the IDD's key list.

    eppy accepts any text for a choice field; EnergyPlus rejects an invalid
    one only when the simulation starts. Checking here turns a fatal error
    in every batch run into one immediate Python error at build time.
    Comparison is case-insensitive, as in EnergyPlus.
    """
    # Access values and their IDD definitions by position (obj.obj and
    # obj.objidd are parallel lists). Looking fields up by name with
    # getfieldidd() made this ~8x slower than building the IDF itself.
    problems = []
    for objects in idf.idfobjects.values():
        for obj in objects:
            values, idd = obj.obj, obj.objidd
            for i in range(1, min(len(values), len(idd))):
                value = values[i]
                if value in ("", None):
                    continue
                keys = idd[i].get("key")
                if keys and str(value).lower() not in _lowered(obj.key, i, keys):
                    shown = ", ".join(keys[:8]) + (", ..." if len(keys) > 8 else "")
                    problems.append(f"{obj.key}.{obj.fieldnames[i]} = {value!r}  (allowed: {shown})")
    if problems:
        raise ValueError(
            "Option values not valid for this EnergyPlus version:\n  " + "\n  ".join(problems)
        )


def build_idf(params: dict[str, Any], idd_path: str) -> IDF:
    """
    Build a complete EnergyPlus IDF from the sampled parameter dict.

    Parameters
    ----------
    params : dict with keys matching the 22 design-space parameters:
        Geometry:    floor_area, aspect_ratio, n_floors, floor_height
        Envelope:    wall_r_value, roof_r_value, floor_r_value,
                     wall_thermal_mass, infiltration_ach
        Fenestration: wwr_north, wwr_east, wwr_south, wwr_west,
                     glazing_u, glazing_shgc, glazing_vt
        Shading:     overhang_depth_n/e/s/w, sidefin_depth
        Orientation: orientation
    idd_path : path to the Energy+.idd file for the target EnergyPlus
               version (e.g. '/Applications/EnergyPlus-24-1-0/Energy+.idd')

    Returns
    -------
    eppy IDF object. Call idf.saveas(path) to write to disk.
    """
    _validate_params(params)

    # Initialize eppy; stamp the IDF with the IDD's own version
    version = _idd_version(idd_path)
    IDF.setiddname(idd_path)
    idf = IDF(StringIO(_SEED_IDF))
    idf.idfobjects["Version"][0].Version_Identifier = version

    # Round n_floors to nearest integer (LHS produces continuous values)
    n_floors = int(round(params["n_floors"]))

    # --- 1. Skeleton ---
    _add_skeleton(idf, orientation=params["orientation"])

    # --- 2. Geometry ---
    geom = build_geometry(
        floor_area=params["floor_area"],
        aspect_ratio=params["aspect_ratio"],
        n_floors=n_floors,
        floor_height=params["floor_height"],
    )
    _add_geometry(idf, geom)

    # --- 3. Constructions ---
    constr = build_constructions(
        wall_r_value=params["wall_r_value"],
        roof_r_value=params["roof_r_value"],
        floor_r_value=params["floor_r_value"],
        wall_thermal_mass=params["wall_thermal_mass"],
    )
    _add_constructions(idf, constr, geom)

    # --- 4. Fenestration ---
    fen = build_fenestration(
        geometry=geom,
        wwr_north=params["wwr_north"],
        wwr_east=params["wwr_east"],
        wwr_south=params["wwr_south"],
        wwr_west=params["wwr_west"],
        glazing_u=params["glazing_u"],
        glazing_shgc=params["glazing_shgc"],
        glazing_vt=params["glazing_vt"],
        overhang_depth_n=params["overhang_depth_n"],
        overhang_depth_e=params["overhang_depth_e"],
        overhang_depth_s=params["overhang_depth_s"],
        overhang_depth_w=params["overhang_depth_w"],
        sidefin_depth=params["sidefin_depth"],
    )
    _add_fenestration(idf, fen, constr)

    # --- 5. Internal loads ---
    loads = build_internal_loads()
    _add_internal_loads(idf, loads, geom, params["infiltration_ach"])

    # --- 6. HVAC ---
    hvac_set = build_hvac(geom, loads)
    _add_hvac(idf, hvac_set)

    # --- 7. Outputs ---
    out = build_outputs()
    _add_outputs(idf, out)

    _validate_choices(idf)
    return idf


# ============================================================
# Step 1: Skeleton IDF
# ============================================================

def _add_skeleton(idf: IDF, orientation: float) -> None:
    """
    Add the minimal set of IDF objects that every simulation needs.
    """
    # (Version is set in build_idf() from the IDD header.)

    # SimulationControl
    idf.newidfobject(
        "SimulationControl",
        # Sizing off: the model has no SizingPeriod (design-day) objects to
        # size against, and ideal loads with NoLimit capacity never need it.
        # Peak loads are taken from the annual run's demand tables instead.
        Do_Zone_Sizing_Calculation="No",
        Do_System_Sizing_Calculation="No",
        Do_Plant_Sizing_Calculation="No",
        Run_Simulation_for_Sizing_Periods="No",
        Run_Simulation_for_Weather_File_Run_Periods="Yes",
    )

    # Building
    idf.newidfobject(
        "Building",
        Name="ParametricOffice",
        North_Axis=orientation,
        Terrain="City",
        Loads_Convergence_Tolerance_Value=0.04,
        Temperature_Convergence_Tolerance_Value=0.4,
        Solar_Distribution="FullInteriorAndExterior",
        Maximum_Number_of_Warmup_Days=25,
        Minimum_Number_of_Warmup_Days=6,
    )

    # Timestep
    idf.newidfobject("Timestep", Number_of_Timesteps_per_Hour=TIMESTEPS_PER_HOUR)

    # RunPeriod — full calendar year
    idf.newidfobject(
        "RunPeriod",
        Name="AnnualRun",
        Begin_Month=1, Begin_Day_of_Month=1,
        End_Month=12, End_Day_of_Month=31,
        Day_of_Week_for_Start_Day="Sunday",
        Use_Weather_File_Holidays_and_Special_Days="Yes",
        Use_Weather_File_Daylight_Saving_Period="No",
        Apply_Weekend_Holiday_Rule="No",
        Use_Weather_File_Rain_Indicators="Yes",
        Use_Weather_File_Snow_Indicators="Yes",
    )

    # GlobalGeometryRules — matches our vertex ordering convention
    idf.newidfobject(
        "GlobalGeometryRules",
        Starting_Vertex_Position="UpperLeftCorner",
        Vertex_Entry_Direction="Counterclockwise",
        # "Relative" is REQUIRED for orientation to work: with "World",
        # EnergyPlus ignores Building North Axis entirely (it warns
        # "Any non-zero Building/Zone North Axes ... are ignored"), so the
        # orientation design parameter would have no effect on any run.
        # Safe here because every Zone origin is (0, 0, 0).
        Coordinate_System="Relative",
    )

    # SurfaceConvectionAlgorithm
    idf.newidfobject(
        "SurfaceConvectionAlgorithm:Inside",
        Algorithm="TARP",
    )
    idf.newidfobject(
        "SurfaceConvectionAlgorithm:Outside",
        Algorithm="DOE-2",
    )

    # HeatBalanceAlgorithm
    idf.newidfobject(
        "HeatBalanceAlgorithm",
        Algorithm="ConductionTransferFunction",
    )


# ============================================================
# Step 2: Geometry -> Zone + BuildingSurface:Detailed
# ============================================================

def _add_geometry(idf: IDF, geom: BuildingGeometry) -> None:
    """Create Zone and BuildingSurface:Detailed objects."""

    for zone in geom.zones:
        idf.newidfobject(
            "Zone",
            Name=zone.name,
            Direction_of_Relative_North=0,
            X_Origin=0, Y_Origin=0, Z_Origin=0,
            Type=1,
            Multiplier=1,
            Ceiling_Height="autocalculate",
            Volume="autocalculate",
        )

        for surface in zone.surfaces:
            _add_surface(idf, surface)


def _add_surface(idf: IDF, surface: Surface) -> None:
    """Translate one geometry.Surface into a BuildingSurface:Detailed."""

    # Map our surface types to EnergyPlus surface types
    ep_surface_type = surface.surface_type
    if ep_surface_type == "Ceiling":
        ep_surface_type = "Ceiling"
    # "Floor", "Wall", "Roof" map directly

    # Sun and wind exposure
    if surface.boundary_condition == "Outdoors":
        sun_exposed = "SunExposed"
        wind_exposed = "WindExposed"
    else:
        sun_exposed = "NoSun"
        wind_exposed = "NoWind"

    obj = idf.newidfobject(
        "BuildingSurface:Detailed",
        Name=surface.name,
        Surface_Type=ep_surface_type,
        Construction_Name="",           # filled in by _assign_constructions
        Zone_Name=surface.zone_name,
        Outside_Boundary_Condition=(GROUND_BC_EPLUS if surface.boundary_condition == "Ground"
                                    else surface.boundary_condition),
        Outside_Boundary_Condition_Object=surface.outside_boundary_object or "",
        Sun_Exposure=sun_exposed,
        Wind_Exposure=wind_exposed,
        View_Factor_to_Ground="autocalculate",
        Number_of_Vertices=len(surface.vertices),
    )

    # Set vertex coordinates
    for i, (x, y, z) in enumerate(surface.vertices):
        setattr(obj, f"Vertex_{i+1}_Xcoordinate", round(x, 6))
        setattr(obj, f"Vertex_{i+1}_Ycoordinate", round(y, 6))
        setattr(obj, f"Vertex_{i+1}_Zcoordinate", round(z, 6))


# ============================================================
# Step 3: Constructions -> Material + Construction
# ============================================================

def _add_constructions(idf: IDF, constr: ConstructionSet, geom: BuildingGeometry) -> None:
    """Create Material, Construction and (ground floors) Construction:FfactorGroundFloor
    objects, then assign constructions to surfaces."""

    # Materials
    for mat in constr.materials:
        idf.newidfobject(
            "Material",
            Name=mat.name,
            Roughness=mat.roughness,
            Thickness=round(mat.thickness, 6),
            Conductivity=mat.conductivity,
            Density=mat.density,
            Specific_Heat=mat.specific_heat,
            Thermal_Absorptance=mat.thermal_absorptance,
            Solar_Absorptance=mat.solar_absorptance,
            Visible_Absorptance=mat.visible_absorptance,
        )

    # Constructions
    for con in constr.constructions:
        kwargs: dict[str, Any] = {"Name": con.name}
        kwargs["Outside_Layer"] = con.layers[0]
        for i, layer_name in enumerate(con.layers[1:], start=2):
            kwargs[f"Layer_{i}"] = layer_name
        idf.newidfobject("Construction", **kwargs)

    # Ground floors: one F-factor construction per ground-floor zone
    for zone in geom.zones:
        if zone.floor_index != 0:
            continue
        idf.newidfobject(
            "Construction:FfactorGroundFloor",
            Name=ground_floor_construction_name(zone.name),
            FFactor=round(constr.ground_floor_ffactor, 6),
            Area=round(zone.floor_area, 6),
            PerimeterExposed=round(zone.exposed_perimeter, 6),
        )

    # Assign constructions to existing surfaces
    for bsd in idf.idfobjects["BuildingSurface:Detailed"]:
        if bsd.Outside_Boundary_Condition == GROUND_BC_EPLUS:
            bsd.Construction_Name = ground_floor_construction_name(bsd.Zone_Name)
            continue
        con_name = get_construction_for_surface(
            surface_type=bsd.Surface_Type,
            boundary_condition=bsd.Outside_Boundary_Condition,
            construction_set=constr,
        )
        bsd.Construction_Name = con_name


# ============================================================
# Step 4: Fenestration
# ============================================================

def _add_fenestration(
    idf: IDF,
    fen: FenestrationSet,
    constr: ConstructionSet,
) -> None:
    """Create glazing material, window construction, windows, and shading."""

    # WindowMaterial:SimpleGlazingSystem
    gl = fen.glazing
    idf.newidfobject(
        "WindowMaterial:SimpleGlazingSystem",
        Name=gl.material_name,
        UFactor=gl.u_factor,
        Solar_Heat_Gain_Coefficient=gl.shgc,
        Visible_Transmittance=gl.visible_transmittance,
    )

    # Construction for windows
    idf.newidfobject(
        "Construction",
        Name=gl.construction_name,
        Outside_Layer=gl.material_name,
    )

    # Windows (FenestrationSurface:Detailed)
    for win in fen.windows:
        obj = idf.newidfobject(
            "FenestrationSurface:Detailed",
            Name=win.name,
            Surface_Type="Window",
            Construction_Name=win.construction_name,
            Building_Surface_Name=win.parent_wall_name,
            View_Factor_to_Ground="autocalculate",
            Number_of_Vertices=len(win.vertices),
        )
        for i, (x, y, z) in enumerate(win.vertices):
            setattr(obj, f"Vertex_{i+1}_Xcoordinate", round(x, 6))
            setattr(obj, f"Vertex_{i+1}_Ycoordinate", round(y, 6))
            setattr(obj, f"Vertex_{i+1}_Zcoordinate", round(z, 6))

    # Shading surfaces (Shading:Zone:Detailed)
    for shade in fen.shading:
        obj = idf.newidfobject(
            "Shading:Zone:Detailed",
            Name=shade.name,
            Base_Surface_Name=shade.base_surface_name,
            Transmittance_Schedule_Name="",
            Number_of_Vertices=len(shade.vertices),
        )
        for i, (x, y, z) in enumerate(shade.vertices):
            setattr(obj, f"Vertex_{i+1}_Xcoordinate", round(x, 6))
            setattr(obj, f"Vertex_{i+1}_Ycoordinate", round(y, 6))
            setattr(obj, f"Vertex_{i+1}_Zcoordinate", round(z, 6))


# ============================================================
# Step 5: Internal loads + schedules + infiltration
# ============================================================

def _add_internal_loads(
    idf: IDF,
    loads: InternalLoadsSet,
    geom: BuildingGeometry,
    infiltration_ach50: float,
) -> None:
    """Create schedules, then per-zone People/Lights/Equipment/Infiltration."""

    # --- Schedule type limits ---
    for tl in loads.schedule_type_limits:
        kwargs: dict[str, Any] = {"Name": tl.name}
        if tl.lower_limit is not None:
            kwargs["Lower_Limit_Value"] = tl.lower_limit
        if tl.upper_limit is not None:
            kwargs["Upper_Limit_Value"] = tl.upper_limit
        kwargs["Numeric_Type"] = tl.numeric_type
        kwargs["Unit_Type"] = tl.unit_type
        idf.newidfobject("ScheduleTypeLimits", **kwargs)

    # --- Schedules (Schedule:Compact) ---
    for sched in loads.schedules:
        _add_schedule_compact(idf, sched)

    # --- Per-zone load objects ---
    natural_ach = infiltration_ach50 / ACH50_TO_NATURAL_DIVISOR

    for zone in geom.zones:
        # People
        idf.newidfobject(
            "People",
            Name=f"{zone.name}_People",
            Zone_or_ZoneList_or_Space_or_SpaceList_Name=zone.name,
            Number_of_People_Schedule_Name=loads.occupancy_schedule_name,
            Number_of_People_Calculation_Method="People/Area",
            People_per_Floor_Area=loads.occupant_density,
            Fraction_Radiant=0.3,
            Sensible_Heat_Fraction="autocalculate",
            Activity_Level_Schedule_Name=loads.activity_schedule_name,
        )

        # Lights
        idf.newidfobject(
            "Lights",
            Name=f"{zone.name}_Lights",
            Zone_or_ZoneList_or_Space_or_SpaceList_Name=zone.name,
            Schedule_Name=loads.lighting_schedule_name,
            Design_Level_Calculation_Method="Watts/Area",
            Watts_per_Floor_Area=loads.lighting_power_density,
            Fraction_Radiant=0.42,
            Fraction_Visible=0.18,
            Return_Air_Fraction=0.0,
        )

        # Electric equipment
        idf.newidfobject(
            "ElectricEquipment",
            Name=f"{zone.name}_Equipment",
            Zone_or_ZoneList_or_Space_or_SpaceList_Name=zone.name,
            Schedule_Name=loads.equipment_schedule_name,
            Design_Level_Calculation_Method="Watts/Area",
            Watts_per_Floor_Area=loads.equipment_power_density,
            Fraction_Radiant=0.5,
            Fraction_Latent=0.0,
            Fraction_Lost=0.0,
        )

        # Infiltration
        idf.newidfobject(
            "ZoneInfiltration:DesignFlowRate",
            Name=f"{zone.name}_Infiltration",
            Zone_or_ZoneList_or_Space_or_SpaceList_Name=zone.name,
            Schedule_Name=loads.infiltration_schedule_name,
            Design_Flow_Rate_Calculation_Method="AirChanges/Hour",
            Air_Changes_per_Hour=natural_ach,
            Constant_Term_Coefficient=1.0,
            Temperature_Term_Coefficient=0.0,
            Velocity_Term_Coefficient=0.0,
            Velocity_Squared_Term_Coefficient=0.0,
        )


def _add_schedule_compact(idf: IDF, sched: Any) -> None:
    """
    Write one Schedule:Compact object from a ScheduleSpec.

    Schedule:Compact uses a variable-length field list. eppy handles
    this through the extensible field mechanism. We build the field
    list manually and set it via the object's field properties.
    """
    # Build the field value list that follows Name and ScheduleTypeLimits
    field_values: list[str] = []
    field_values.append(f"Through: {sched.through_date}")

    for dp in sched.day_patterns:
        day_str = "For: " + " ".join(dp.day_types)
        field_values.append(day_str)
        for iv in dp.intervals:
            field_values.append(f"Until: {iv.until_time}")
            field_values.append(str(iv.value))

    # Create the object with enough extensible fields
    obj = idf.newidfobject("Schedule:Compact")
    obj.Name = sched.name
    obj.Schedule_Type_Limits_Name = sched.schedule_type_limits

    # eppy's Schedule:Compact has Field_1, Field_2, ... for the
    # extensible fields. Set them in order.
    for i, val in enumerate(field_values):
        field_name = f"Field_{i + 1}"
        try:
            setattr(obj, field_name, val)
        except Exception as exc:
            # Never truncate silently: a shortened schedule changes occupancy,
            # lighting or setpoint hours with no visible error anywhere.
            raise RuntimeError(
                f"Schedule:Compact {sched.name!r}: could not set {field_name} "
                f"(field {i + 1} of {len(field_values)})"
            ) from exc


# ============================================================
# Step 6: HVAC
# ============================================================

def _add_hvac(idf: IDF, hvac_set: HVACSet) -> None:
    """Create HVACTemplate:Thermostat and HVACTemplate:Zone:IdealLoadsAirSystem."""

    t = hvac_set.thermostat
    idf.newidfobject(
        "HVACTemplate:Thermostat",
        Name=t.name,
        Heating_Setpoint_Schedule_Name=t.heating_setpoint_schedule_name,
        Constant_Heating_Setpoint="",
        Cooling_Setpoint_Schedule_Name=t.cooling_setpoint_schedule_name,
        Constant_Cooling_Setpoint="",
    )

    for sys in hvac_set.zone_systems:
        idf.newidfobject(
            "HVACTemplate:Zone:IdealLoadsAirSystem",
            Zone_Name=sys.zone_name,
            Template_Thermostat_Name=sys.thermostat_name,
            System_Availability_Schedule_Name=sys.system_availability_schedule_name,
            Maximum_Heating_Supply_Air_Temperature=sys.max_heating_supply_temp,
            Minimum_Cooling_Supply_Air_Temperature=sys.min_cooling_supply_temp,
            Maximum_Heating_Supply_Air_Humidity_Ratio=sys.max_heating_supply_humidity_ratio,
            Minimum_Cooling_Supply_Air_Humidity_Ratio=sys.min_cooling_supply_humidity_ratio,
            Heating_Limit=sys.heating_limit,
            Cooling_Limit=sys.cooling_limit,
            Dehumidification_Control_Type=sys.dehumidification_control_type,
            Humidification_Control_Type=sys.humidification_control_type,
            Outdoor_Air_Method=sys.outdoor_air_method,
            Outdoor_Air_Flow_Rate_per_Person=sys.oa_per_person,
            Outdoor_Air_Flow_Rate_per_Zone_Floor_Area=sys.oa_per_area,
            Outdoor_Air_Flow_Rate_per_Zone=sys.oa_per_zone,
            Demand_Controlled_Ventilation_Type=sys.demand_controlled_ventilation_type,
            Outdoor_Air_Economizer_Type=sys.economizer_type,
            Heat_Recovery_Type=sys.heat_recovery_type,
        )


# ============================================================
# Step 7: Outputs
# ============================================================

def _add_outputs(idf: IDF, out: OutputSet) -> None:
    """Create output request objects."""

    # Output:Variable
    for v in out.variables:
        idf.newidfobject(
            "Output:Variable",
            Key_Value=v.key_value,
            Variable_Name=v.variable_name,
            Reporting_Frequency=v.reporting_frequency,
        )

    # Output:Meter
    for m in out.meters:
        idf.newidfobject(
            "Output:Meter",
            Key_Name=m.meter_name,
            Reporting_Frequency=m.reporting_frequency,
        )

    # Output:Table:SummaryReports
    # Fields are Report_1_Name, Report_2_Name, ... An earlier version used
    # wrong field names inside try/except-pass, so this object was silently
    # empty and EnergyPlus produced no peak-demand or unmet-hours tables.
    # Errors are no longer swallowed.
    obj = idf.newidfobject("Output:Table:SummaryReports")
    for i, report_name in enumerate(out.summary_reports.report_names):
        setattr(obj, f"Report_{i + 1}_Name", report_name)

    # OutputControl:Table:Style
    idf.newidfobject(
        "OutputControl:Table:Style",
        Column_Separator=out.table_style,
    )

    # Output:SQLite
    idf.newidfobject(
        "Output:SQLite",
        Option_Type=out.sqlite_mode,
    )


# ============================================================
# Validation
# ============================================================

_REQUIRED_PARAMS = {
    "floor_area", "aspect_ratio", "n_floors", "floor_height",
    "wall_r_value", "roof_r_value", "floor_r_value",
    "wall_thermal_mass", "infiltration_ach",
    "wwr_north", "wwr_east", "wwr_south", "wwr_west",
    "glazing_u", "glazing_shgc", "glazing_vt",
    "overhang_depth_n", "overhang_depth_e",
    "overhang_depth_s", "overhang_depth_w",
    "sidefin_depth",
    "orientation",
}


def _validate_params(params: dict[str, Any]) -> None:
    """Check that all required parameters are present."""
    missing = _REQUIRED_PARAMS - set(params.keys())
    if missing:
        raise ValueError(
            f"Missing required parameters: {sorted(missing)}"
        )
