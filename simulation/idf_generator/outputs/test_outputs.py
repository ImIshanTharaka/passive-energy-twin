"""
test_outputs.py - Structural checks for the outputs module.

Tests verify:
  1. Default OutputSet contains expected variables, meters, reports
  2. All target output variables for Phase 2 plan are requested
  3. Reporting frequency is propagated correctly
  4. SQLite is enabled by default
  5. No duplicate variables / meters / reports
  6. Bad frequency rejected
  7. Hourly Heating:/Cooling:EnergyTransfer meters for peak loads (Decision T4)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repository root on sys.path, so the idf_generator package imports
# correctly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import idf_generator.outputs.outputs as O
# ---------- Structural ----------

def check_default_build_returns_outputset():
    out = O.build_outputs()
    assert isinstance(out, O.OutputSet)
    assert len(out.variables) > 0
    assert len(out.meters) > 0
    assert len(out.summary_reports.report_names) > 0


def check_no_duplicate_variables():
    out = O.build_outputs()
    keys = [(v.variable_name, v.key_value) for v in out.variables]
    assert len(keys) == len(set(keys)), f"Duplicate variable requests: {keys}"


def check_no_duplicate_meters():
    out = O.build_outputs()
    keys = [(m.meter_name, m.reporting_frequency) for m in out.meters]
    assert len(keys) == len(set(keys)), f"Duplicate meter requests: {keys}"


def check_no_duplicate_summary_reports():
    out = O.build_outputs()
    names = out.summary_reports.report_names
    assert len(names) == len(set(names)), f"Duplicate summary reports: {names}"


# ---------- Target variables present ----------

def check_heating_energy_transfer_meter_present():
    out = O.build_outputs()
    meter_names = {m.meter_name for m in out.meters}
    assert "Heating:EnergyTransfer" in meter_names


def check_cooling_energy_transfer_meter_present():
    out = O.build_outputs()
    meter_names = {m.meter_name for m in out.meters}
    assert "Cooling:EnergyTransfer" in meter_names


def check_lighting_meter_present():
    out = O.build_outputs()
    meter_names = {m.meter_name for m in out.meters}
    assert "InteriorLights:Electricity" in meter_names


def check_equipment_meter_present():
    out = O.build_outputs()
    meter_names = {m.meter_name for m in out.meters}
    assert "InteriorEquipment:Electricity" in meter_names


def check_zone_cooling_variable_present():
    """Required for zone-level cooling breakdown (N/E/S/W)."""
    out = O.build_outputs()
    var_names = {v.variable_name for v in out.variables}
    assert "Zone Ideal Loads Zone Sensible Cooling Energy" in var_names


def check_zone_heating_variable_present():
    out = O.build_outputs()
    var_names = {v.variable_name for v in out.variables}
    assert "Zone Ideal Loads Zone Sensible Heating Energy" in var_names


def check_zone_air_temperature_present():
    """Needed for unmet-hours validation."""
    out = O.build_outputs()
    var_names = {v.variable_name for v in out.variables}
    assert "Zone Mean Air Temperature" in var_names


def check_peak_demand_report_present():
    """Gives peak_cooling_load and peak_heating_load."""
    out = O.build_outputs()
    assert "DemandEndUseComponentsSummary" in out.summary_reports.report_names


def check_unmet_hours_report_present():
    out = O.build_outputs()
    # 24.1: building-level unmet hours live inside ABUPS; per-zone in SystemSummary
    assert "AnnualBuildingUtilityPerformanceSummary" in out.summary_reports.report_names
    assert "SystemSummary" in out.summary_reports.report_names


def check_annual_summary_report_present():
    out = O.build_outputs()
    assert "AnnualBuildingUtilityPerformanceSummary" in out.summary_reports.report_names


# ---------- Configuration ----------

def check_sqlite_enabled_by_default():
    out = O.build_outputs()
    assert out.sqlite_mode == "SimpleAndTabular"


def check_default_table_style_html():
    out = O.build_outputs()
    assert out.table_style == "HTML"


def _non_peak(meters):
    return [m for m in meters if m not in O._PEAK_METERS]


def check_default_reporting_frequency_is_runperiod():
    out = O.build_outputs()
    for v in out.variables:
        assert v.reporting_frequency == "RunPeriod", \
            f"{v.variable_name}: frequency is {v.reporting_frequency}"
    for m in _non_peak(out.meters):
        assert m.reporting_frequency == "RunPeriod", \
            f"{m.meter_name}: frequency is {m.reporting_frequency}"


def check_custom_frequency_propagates():
    out = O.build_outputs(reporting_frequency="Monthly")
    for v in out.variables:
        assert v.reporting_frequency == "Monthly"
    for m in _non_peak(out.meters):
        assert m.reporting_frequency == "Monthly"


# ---------- Peak loads (Decision T4) ----------

def check_hourly_peak_meters_present():
    out = O.build_outputs()
    keys = {(m.meter_name, m.reporting_frequency) for m in out.meters}
    assert ("Heating:EnergyTransfer", "Hourly") in keys
    assert ("Cooling:EnergyTransfer", "Hourly") in keys


def check_annual_meters_unchanged_by_peak_meters():
    """The annual copies must still be there - annual targets come from them."""
    out = O.build_outputs()
    keys = {(m.meter_name, m.reporting_frequency) for m in out.meters}
    for name in ("Heating:EnergyTransfer", "Cooling:EnergyTransfer"):
        assert (name, "RunPeriod") in keys, name
    assert len(out.meters) == 6 + 2


def check_only_two_hourly_series_by_default():
    """Keep the SQL small: nothing else is hourly unless explicitly asked for."""
    out = O.build_outputs()
    hourly = [m for m in out.meters if m.reporting_frequency == "Hourly"]
    hourly += [v for v in out.variables if v.reporting_frequency == "Hourly"]
    assert len(hourly) == 2, hourly


def check_hourly_build_has_no_duplicate_peak_meters():
    out = O.build_outputs(reporting_frequency="Hourly")
    keys = [(m.meter_name, m.reporting_frequency) for m in out.meters]
    assert len(keys) == len(set(keys)) == 6, keys


def check_all_valid_frequencies_accepted():
    for freq in ["Hourly", "Daily", "Monthly", "RunPeriod", "Annual"]:
        out = O.build_outputs(reporting_frequency=freq)
        assert all(v.reporting_frequency == freq for v in out.variables)


# ---------- Bad inputs ----------

def check_rejects_invalid_frequency():
    try:
        O.build_outputs(reporting_frequency="Weekly")
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


# ---------- Runner ----------

ALL_CHECKS = [
    check_default_build_returns_outputset,
    check_no_duplicate_variables,
    check_no_duplicate_meters,
    check_no_duplicate_summary_reports,
    check_heating_energy_transfer_meter_present,
    check_cooling_energy_transfer_meter_present,
    check_lighting_meter_present,
    check_equipment_meter_present,
    check_zone_cooling_variable_present,
    check_zone_heating_variable_present,
    check_zone_air_temperature_present,
    check_peak_demand_report_present,
    check_unmet_hours_report_present,
    check_annual_summary_report_present,
    check_sqlite_enabled_by_default,
    check_default_table_style_html,
    check_default_reporting_frequency_is_runperiod,
    check_custom_frequency_propagates,
    check_all_valid_frequencies_accepted,
    check_rejects_invalid_frequency,
    check_hourly_peak_meters_present,
    check_annual_meters_unchanged_by_peak_meters,
    check_only_two_hourly_series_by_default,
    check_hourly_build_has_no_duplicate_peak_meters,
]


def main():
    total = failed = 0
    print("=" * 72)
    print("OUTPUTS MODULE - STRUCTURAL CHECKS")
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
