"""
test_generator.py - Integration tests for the IDF generator.

These tests verify that build_idf() produces a well-formed IDF with
the expected object counts and relationships, built against the
EnergyPlus 24.1 IDD - the version the simulations actually run on.
(Earlier versions of these tests used eppy's bundled 9.2 IDD, which hid
seven field-name incompatibilities with 24.1.)

Run:
    python tests/test_generator.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Repository root on sys.path, so the idf_generator package imports
# correctly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import idf_generator.generator.generator as GEN
def _resolve_idd() -> str:
    """Locate Energy+.idd.

    Order: $EPLUS_IDD, then Energy+.idd beside the EnergyPlus binary named in
    energyplus_path.txt, then the default macOS install location.
    """
    override = os.environ.get("EPLUS_IDD")
    if override:
        return override
    exe_file = Path(__file__).resolve().parents[2] / "energyplus_path.txt"
    if exe_file.exists():
        return str(Path(exe_file.read_text().strip()).parent / "Energy+.idd")
    return "/Applications/EnergyPlus-24-1-0/Energy+.idd"


IDD_PATH = _resolve_idd()


def _default_params(**overrides) -> dict:
    """Return a full parameter dict with sensible defaults."""
    params = {
        "floor_area": 5000.0,
        "aspect_ratio": 1.5,
        "n_floors": 2,
        "floor_height": 3.5,
        "wall_r_value": 4.0,
        "roof_r_value": 7.0,
        "floor_r_value": 2.5,
        "wall_thermal_mass": 200.0,
        "infiltration_ach": 1.0,
        "wwr_north": 0.3,
        "wwr_east": 0.4,
        "wwr_south": 0.5,
        "wwr_west": 0.6,
        "glazing_u": 2.5,
        "glazing_shgc": 0.4,
        "glazing_vt": 0.3,
        "overhang_depth_n": 0.0,
        "overhang_depth_e": 0.5,
        "overhang_depth_s": 1.0,
        "overhang_depth_w": 1.5,
        "sidefin_depth": 0.5,
        "orientation": 90.0,
    }
    params.update(overrides)
    return params


# ---------- Core functionality ----------

def check_build_idf_runs():
    """build_idf() completes without error."""
    params = _default_params()
    idf = GEN.build_idf(params, IDD_PATH)
    assert idf is not None


def check_zone_count():
    """5 zones per floor * n_floors."""
    params = _default_params(n_floors=3)
    idf = GEN.build_idf(params, IDD_PATH)
    zones = idf.idfobjects["Zone"]
    assert len(zones) == 15, f"Expected 15 zones, got {len(zones)}"


def check_surface_count():
    """6 surfaces per zone (1 floor + 1 ceiling/roof + 4 walls)."""
    params = _default_params(n_floors=2)
    idf = GEN.build_idf(params, IDD_PATH)
    surfaces = idf.idfobjects["BuildingSurface:Detailed"]
    expected = 5 * 2 * 6  # 5 zones/floor * 2 floors * 6 surfaces/zone
    assert len(surfaces) == expected, \
        f"Expected {expected} surfaces, got {len(surfaces)}"


def check_window_count():
    """1 window per exterior wall = 4 walls/floor * n_floors."""
    params = _default_params(n_floors=2)
    idf = GEN.build_idf(params, IDD_PATH)
    windows = idf.idfobjects["FenestrationSurface:Detailed"]
    expected = 4 * 2  # 4 cardinal walls * 2 floors
    assert len(windows) == expected, \
        f"Expected {expected} windows, got {len(windows)}"


def check_material_count():
    """10 materials (12 before Decision G1 removed FloorInsulation + GroundSlabConcrete)."""
    params = _default_params()
    idf = GEN.build_idf(params, IDD_PATH)
    mats = idf.idfobjects["Material"]
    assert len(mats) == 10, f"Expected 10 materials, got {len(mats)}"


def check_construction_count():
    """5 layered opaque (incl. mirrored interzone ceiling) + 1 window = 6 constructions;
    ground floors are Construction:FfactorGroundFloor (Decision G1)."""
    params = _default_params()
    idf = GEN.build_idf(params, IDD_PATH)
    constructions = idf.idfobjects["Construction"]
    assert len(constructions) == 6, \
        f"Expected 6 constructions, got {len(constructions)}"


# ---------- Ground floor: F-factor slab (Decision G1) ----------

def check_ground_floor_ffactor_constructions():
    from idf_generator.construction.constructions import ffactor_for_floor_r
    params = _default_params(floor_r_value=3.0)
    idf = GEN.build_idf(params, IDD_PATH)
    ff = {c.Name: c for c in idf.idfobjects["Construction:FfactorGroundFloor"]}
    assert len(ff) == 5, f"one per ground-floor zone, got {len(ff)}"
    for c in ff.values():
        assert abs(float(c.FFactor) - ffactor_for_floor_r(3.0)) < 1e-5, c.FFactor
        assert float(c.Area) > 0
    core = ff["GroundFloor_Zone_F0_Core"]
    assert float(core.PerimeterExposed) == 0.0, "core has no exposed slab edge"
    geom_l = sum(float(c.PerimeterExposed) for c in ff.values())
    zones_f0 = [s for s in idf.idfobjects["BuildingSurface:Detailed"]
                if s.Outside_Boundary_Condition == "GroundFCfactorMethod"]
    assert len(zones_f0) == 5
    for s in zones_f0:
        assert s.Surface_Type == "Floor" and s.Construction_Name == f"GroundFloor_{s.Zone_Name}", s.Name
        assert abs(float(ff[s.Construction_Name].Area) - _poly_area(s)) < 1e-3, s.Name
    assert geom_l > 0


def _poly_area(s) -> float:
    n = int(s.Number_of_Vertices)
    pts = [(float(getattr(s, f"Vertex_{i}_Xcoordinate")), float(getattr(s, f"Vertex_{i}_Ycoordinate")))
           for i in range(1, n + 1)]
    return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))) / 2


def check_no_ground_temperature_objects():
    """No Site:GroundTemperature:* -> EnergyPlus takes FCfactorMethod temps from the EPW."""
    idf = GEN.build_idf(_default_params(), IDD_PATH)
    for obj in ("Site:GroundTemperature:BuildingSurface", "Site:GroundTemperature:FCfactorMethod"):
        assert not idf.idfobjects[obj], f"{obj} must not be written (Decision G1)"
    assert not [s for s in idf.idfobjects["BuildingSurface:Detailed"]
                if s.Outside_Boundary_Condition == "Ground"], "no plain 'Ground' boundaries left"


def check_more_floor_insulation_lowers_ffactor():
    lo = GEN.build_idf(_default_params(floor_r_value=0.5), IDD_PATH)
    hi = GEN.build_idf(_default_params(floor_r_value=5.0), IDD_PATH)
    f = lambda idf: float(idf.idfobjects["Construction:FfactorGroundFloor"][0].FFactor)  # noqa: E731
    assert f(hi) < f(lo), (f(lo), f(hi))


def check_schedule_count():
    """8 schedules."""
    params = _default_params()
    idf = GEN.build_idf(params, IDD_PATH)
    scheds = idf.idfobjects["Schedule:Compact"]
    assert len(scheds) == 8, f"Expected 8 schedules, got {len(scheds)}"


def check_hvac_thermostat():
    params = _default_params()
    idf = GEN.build_idf(params, IDD_PATH)
    thermos = idf.idfobjects["HVACTemplate:Thermostat"]
    assert len(thermos) == 1, f"Expected 1 thermostat, got {len(thermos)}"


def check_hvac_ideal_loads_count():
    """One IdealLoads per zone."""
    params = _default_params(n_floors=2)
    idf = GEN.build_idf(params, IDD_PATH)
    ideal = idf.idfobjects["HVACTemplate:Zone:IdealLoadsAirSystem"]
    expected = 5 * 2
    assert len(ideal) == expected, \
        f"Expected {expected} IdealLoads, got {len(ideal)}"


def check_per_zone_loads_count():
    """One People, Lights, Equipment, Infiltration per zone."""
    params = _default_params(n_floors=2)
    idf = GEN.build_idf(params, IDD_PATH)
    n_zones = 10
    for obj_type in ["People", "Lights", "ElectricEquipment",
                     "ZoneInfiltration:DesignFlowRate"]:
        objs = idf.idfobjects[obj_type]
        assert len(objs) == n_zones, \
            f"Expected {n_zones} {obj_type}, got {len(objs)}"


def check_output_variables_present():
    params = _default_params()
    idf = GEN.build_idf(params, IDD_PATH)
    vars_ = idf.idfobjects["Output:Variable"]
    assert len(vars_) >= 9, f"Expected >= 9 output variables, got {len(vars_)}"


def check_output_meters_present():
    params = _default_params()
    idf = GEN.build_idf(params, IDD_PATH)
    meters = idf.idfobjects["Output:Meter"]
    assert len(meters) >= 6, f"Expected >= 6 output meters, got {len(meters)}"


def check_sqlite_enabled():
    params = _default_params()
    idf = GEN.build_idf(params, IDD_PATH)
    sql = idf.idfobjects["Output:SQLite"]
    assert len(sql) == 1


def check_building_orientation():
    params = _default_params(orientation=135.0)
    idf = GEN.build_idf(params, IDD_PATH)
    bldg = idf.idfobjects["Building"][0]
    assert bldg.North_Axis == 135.0, \
        f"Expected orientation 135.0, got {bldg.North_Axis}"
    # Setting North Axis is not enough: EnergyPlus ignores it unless the
    # coordinate system is Relative. This passed silently before the fix.
    rules = idf.idfobjects["GlobalGeometryRules"][0]
    assert rules.Coordinate_System == "Relative", \
        f"Coordinate_System={rules.Coordinate_System!r}: North Axis would be ignored"
    for z in idf.idfobjects["Zone"]:
        assert float(z.X_Origin or 0) == float(z.Y_Origin or 0) == float(z.Z_Origin or 0) == 0.0, \
            f"{z.Name} has a non-zero origin - Relative coordinates would shift it"


def check_n_floors_rounding():
    """n_floors=2.7 should round to 3 -> 15 zones."""
    params = _default_params(n_floors=2.7)
    idf = GEN.build_idf(params, IDD_PATH)
    zones = idf.idfobjects["Zone"]
    assert len(zones) == 15, f"Expected 15 zones (3 floors), got {len(zones)}"


def check_infiltration_rate_converted():
    """ACH50=1.0 -> natural ACH = 1.0/20 = 0.05."""
    params = _default_params(infiltration_ach=1.0)
    idf = GEN.build_idf(params, IDD_PATH)
    infils = idf.idfobjects["ZoneInfiltration:DesignFlowRate"]
    for inf in infils:
        assert abs(inf.Air_Changes_per_Hour - 0.05) < 1e-6, \
            f"Expected ACH 0.05, got {inf.Air_Changes_per_Hour}"


def check_glazing_material_created():
    params = _default_params()
    idf = GEN.build_idf(params, IDD_PATH)
    glaz = idf.idfobjects["WindowMaterial:SimpleGlazingSystem"]
    assert len(glaz) == 1
    assert glaz[0].UFactor == 2.5
    assert glaz[0].Solar_Heat_Gain_Coefficient == 0.4


def check_shading_surfaces_created():
    """With overhangs on E/S/W and fins everywhere, expect shading."""
    params = _default_params(
        n_floors=1,
        overhang_depth_n=0.0, overhang_depth_e=0.5,
        overhang_depth_s=1.0, overhang_depth_w=1.5,
        sidefin_depth=0.5,
    )
    idf = GEN.build_idf(params, IDD_PATH)
    shades = idf.idfobjects["Shading:Zone:Detailed"]
    # 3 overhangs (E, S, W; N=0) + 4*2=8 fins = 11
    assert len(shades) == 11, \
        f"Expected 11 shading surfaces, got {len(shades)}"


def check_no_shading_when_zero():
    params = _default_params(
        n_floors=1,
        overhang_depth_n=0.0, overhang_depth_e=0.0,
        overhang_depth_s=0.0, overhang_depth_w=0.0,
        sidefin_depth=0.0,
    )
    idf = GEN.build_idf(params, IDD_PATH)
    shades = idf.idfobjects["Shading:Zone:Detailed"]
    assert len(shades) == 0, f"Expected 0 shading, got {len(shades)}"


# ---------- Bad inputs ----------

def check_rejects_missing_params():
    params = {"floor_area": 5000.0}  # incomplete
    try:
        GEN.build_idf(params, IDD_PATH)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def check_idf_saveable():
    """The IDF can be written to a string without error."""
    params = _default_params()
    idf = GEN.build_idf(params, IDD_PATH)
    text = idf.idfstr()
    assert len(text) > 1000, f"IDF text too short ({len(text)} chars)"
    assert "ParametricOffice" in text


# ---------- Corner cases ----------

def check_single_floor():
    params = _default_params(n_floors=1)
    idf = GEN.build_idf(params, IDD_PATH)
    zones = idf.idfobjects["Zone"]
    assert len(zones) == 5


def check_max_floors():
    params = _default_params(n_floors=5)
    idf = GEN.build_idf(params, IDD_PATH)
    zones = idf.idfobjects["Zone"]
    assert len(zones) == 25


# ---------- Runner ----------


# ---------- EnergyPlus version compatibility ----------

def check_version_matches_idd():
    """The IDF must declare the IDD's version - a mismatch is fatal in EnergyPlus."""
    idf = GEN.build_idf(_default_params(), IDD_PATH)
    declared = str(idf.idfobjects["Version"][0].Version_Identifier)
    expected = GEN._idd_version(IDD_PATH)
    assert declared == expected, f"IDF says {declared}, IDD is {expected}"


def check_idd_version_parser():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        def version_of(header):
            p = Path(td) / "x.idd"
            p.write_text(header + "\n")
            return GEN._idd_version(str(p))
        assert version_of("!IDD_Version 24.1.0") == "24.1"
        assert version_of("!IDD_Version 9.6.0") == "9.6"
        for bad in ("!IDD_Version 9.2.0", "!IDD_Version garbage", "! not an idd"):
            try:
                version_of(bad)
            except ValueError:
                continue
            raise AssertionError(f"accepted {bad!r}")


def check_space_aware_field_names():
    """Loads reference zones and densities via the 9.6+ field names."""
    idf = GEN.build_idf(_default_params(), IDD_PATH)
    zone_names = {z.Name for z in idf.idfobjects["Zone"]}
    for obj_type in ("People", "Lights", "ElectricEquipment", "ZoneInfiltration:DesignFlowRate"):
        for obj in idf.idfobjects[obj_type]:
            ref = obj.Zone_or_ZoneList_or_Space_or_SpaceList_Name
            assert ref in zone_names, f"{obj_type} {obj.Name} references unknown zone {ref!r}"
    assert all(float(o.People_per_Floor_Area) > 0 for o in idf.idfobjects["People"])
    assert all(float(o.Watts_per_Floor_Area) > 0 for o in idf.idfobjects["Lights"])
    assert all(float(o.Watts_per_Floor_Area) > 0 for o in idf.idfobjects["ElectricEquipment"])


def check_outside_convection_algorithm_is_valid():
    idf = GEN.build_idf(_default_params(), IDD_PATH)
    algo = idf.idfobjects["SurfaceConvectionAlgorithm:Outside"][0].Algorithm
    assert algo == "DOE-2", algo


# ---------- Silent-failure regressions ----------

def check_summary_reports_requested():
    """Peak-load and unmet-hours targets come from these tables (2.7)."""
    import idf_generator.outputs.outputs as O
    idf = GEN.build_idf(_default_params(), IDD_PATH)
    objs = idf.idfobjects["Output:Table:SummaryReports"]
    assert len(objs) == 1, f"{len(objs)} SummaryReports objects"
    wanted = list(O.build_outputs().summary_reports.report_names)
    got = [getattr(objs[0], f"Report_{i + 1}_Name") for i in range(len(wanted))]
    assert got == wanted, f"requested {wanted}, IDF has {got}"
    for needed in ("DemandEndUseComponentsSummary", "AnnualBuildingUtilityPerformanceSummary",
                   "SystemSummary"):
        assert needed in got, f"{needed} missing - a Phase 2 target depends on it"


def check_schedules_complete():
    """Every Schedule:Compact carries every field its spec defines."""
    from idf_generator.internal_loads.internal_loads import build_internal_loads
    idf = GEN.build_idf(_default_params(), IDD_PATH)
    written = {s.Name: sum(1 for f in s.fieldnames[3:] if getattr(s, f, "") not in ("", None))
               for s in idf.idfobjects["Schedule:Compact"]}
    for spec in build_internal_loads().schedules:
        expected = 1 + sum(1 + 2 * len(dp.intervals) for dp in spec.day_patterns)
        assert written.get(spec.name) == expected, (
            f"{spec.name}: {written.get(spec.name)} fields written, {expected} expected")


def check_sizing_disabled():
    """No design days exist, so sizing must be off."""
    idf = GEN.build_idf(_default_params(), IDD_PATH)
    sc = idf.idfobjects["SimulationControl"][0]
    assert sc.Do_Zone_Sizing_Calculation == "No", sc.Do_Zone_Sizing_Calculation
    assert sc.Do_System_Sizing_Calculation == "No", sc.Do_System_Sizing_Calculation
    assert not idf.idfobjects["SizingPeriod:DesignDay"], "design days present - revisit sizing"


def check_invalid_option_value_rejected_at_build_time():
    """A bad option must fail in Python, not 3,400 times inside EnergyPlus."""
    idf = GEN.build_idf(_default_params(), IDD_PATH)
    idf.idfobjects["Output:Table:SummaryReports"][0].Report_1_Name = "NoSuchReport"
    try:
        GEN._validate_choices(idf)
    except ValueError as exc:
        assert "NoSuchReport" in str(exc), exc
        return
    raise AssertionError("invalid report name was accepted")


def check_built_idf_passes_choice_validation():
    idf = GEN.build_idf(_default_params(), IDD_PATH)
    GEN._validate_choices(idf)          # raises if anything is invalid


def check_interzone_pairs_are_mirrored():
    """Every interzone floor/ceiling pair uses mirror-image constructions."""
    idf = GEN.build_idf(_default_params(n_floors=3), IDD_PATH)
    surfaces = {s.Name: s for s in idf.idfobjects["BuildingSurface:Detailed"]}
    layers = {c.Name: [f for f in c.fieldvalues[2:] if f] for c in idf.idfobjects["Construction"]}
    pairs = 0
    for s in surfaces.values():
        if s.Surface_Type == "Ceiling" and s.Outside_Boundary_Condition == "Surface":
            other = surfaces[s.Outside_Boundary_Condition_Object]
            assert layers[s.Construction_Name] == list(reversed(layers[other.Construction_Name])), \
                f"{s.Name} / {other.Name} constructions are not mirrored"
            pairs += 1
    assert pairs == 10, f"expected 10 interzone ceiling/floor pairs for 3 floors, found {pairs}"

ALL_CHECKS = [
    check_interzone_pairs_are_mirrored,
    check_invalid_option_value_rejected_at_build_time,
    check_built_idf_passes_choice_validation,
    check_summary_reports_requested,
    check_schedules_complete,
    check_sizing_disabled,
    check_version_matches_idd,
    check_idd_version_parser,
    check_space_aware_field_names,
    check_outside_convection_algorithm_is_valid,
    check_build_idf_runs,
    check_zone_count,
    check_surface_count,
    check_window_count,
    check_material_count,
    check_construction_count,
    check_ground_floor_ffactor_constructions,
    check_no_ground_temperature_objects,
    check_more_floor_insulation_lowers_ffactor,
    check_schedule_count,
    check_hvac_thermostat,
    check_hvac_ideal_loads_count,
    check_per_zone_loads_count,
    check_output_variables_present,
    check_output_meters_present,
    check_sqlite_enabled,
    check_building_orientation,
    check_n_floors_rounding,
    check_infiltration_rate_converted,
    check_glazing_material_created,
    check_shading_surfaces_created,
    check_no_shading_when_zero,
    check_rejects_missing_params,
    check_idf_saveable,
    check_single_floor,
    check_max_floors,
]


def main():
    if not Path(IDD_PATH).exists():
        print(f"IDD not found: {IDD_PATH}\nSet EPLUS_IDD or fix energyplus_path.txt.")
        return 2
    print(f"IDD: {IDD_PATH}")
    total = failed = 0
    print("=" * 72)
    print("GENERATOR MODULE - INTEGRATION TESTS")
    print("=" * 72)
    for chk in ALL_CHECKS:
        total += 1
        try:
            chk()
            print(f"  PASS  {chk.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {chk.__name__}: {type(e).__name__}: {e}")
    print("=" * 72)
    print(f"RESULTS: {total - failed}/{total} passed, {failed} failed")
    print("=" * 72)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
