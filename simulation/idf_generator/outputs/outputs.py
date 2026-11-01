"""
outputs.py - EnergyPlus output requests.

This module declares WHICH EnergyPlus outputs to request from each
simulation. Every output corresponds to a target variable in the Phase 2
dataset (annual heating/cooling demand, peak loads, unmet hours, etc.)
or to a diagnostic variable (zone temperature, solar gains) useful for
sanity-checking during the pilot batch.

WHAT THIS MODULE PRODUCES
-------------------------
  OutputVariableRequest      - Output:Variable IDF objects
  OutputMeterRequest         - Output:Meter IDF objects
  OutputTableSummary         - Output:Table:SummaryReports object
  OutputSet                  - container for all of the above + SQLite
                               + table style settings

generator.py translates these into:
  Output:Variable                          (one per request)
  Output:Meter                             (one per request)
  Output:Table:SummaryReports              (1 total)
  Output:SQLite                            (1 total, "SimpleAndTabular")
  OutputControl:Table:Style                (1 total, "HTML")

MAPPING TO PHASE 2 TARGETS
--------------------------
DECISION (27 Sep 2026): heating/cooling DEMAND means ZONE LOAD - the
energy the ideal-loads system delivers to the zones to hold setpoint
(envelope + solar + internal gains). It EXCLUDES conditioning of outdoor
ventilation air, which ABUPS reports as "District Heating/Cooling".
Rationale: ventilation is fixed by occupancy rules and is not a design
parameter, so including it adds a large climate-driven term that passive
design cannot influence. Chicago smoke test: zone heating 76 MWh vs
supply heating 328 MWh. The exact zone-level variable (EnergyTransfer
meter vs Zone Ideal Loads Zone Sensible variables, which differ by ~3%)
is fixed in Sub-task 2.7. Unmet hours are ~0 with NoLimit ideal loads and
will not be used as learning targets.

  annual_heating_demand        <- Heating:EnergyTransfer meter, /area
  annual_cooling_demand        <- Cooling:EnergyTransfer meter, /area
  total_EUI                    <- sum of 4 meters / area
  peak_cooling_load            <- HOURLY Cooling:EnergyTransfer meter (see PEAK LOADS)
  peak_heating_load            <- HOURLY Heating:EnergyTransfer meter (see PEAK LOADS)
  unmet_hours_cooling          <- AnnualBuildingUtilityPerformanceSummary,
  unmet_hours_heating             "Comfort and Setpoint Not Met Summary" sub-table
                                  (per zone: SystemSummary)
  zone_cooling_N/E/S/W         <- Zone Ideal Loads Sensible Cooling
                                  Energy variable, per zone

PEAK LOADS (Decision T4, 8 Oct 2026, Sub-task 2.5)
--------------------------------------------------
The summary-table peaks (SensibleHeatGainSummary, DemandEndUseComponents)
are the highest single-TIMESTEP (10 min) rate of the year. With NoLimit
ideal loads and the DOE 06:00 setpoint step (15.6->21 C, 26.7->24 C) every
zone is pulled to setpoint within one timestep, so those peaks are
setback-recovery spikes (smoke run: ~620 W/m2 heating; 135/180 monthly
zone cooling peaks at 06:xx), not envelope/solar-driven loads.
Fix: Heating:/Cooling:EnergyTransfer are ALSO requested at Hourly
frequency (_PEAK_METERS, 2 x 8,760 values per run). The simulation still
runs at 6 timesteps/hour; only the REPORTING is hourly. The output parser
derives the hourly-mean peak and a steady-state peak that excludes the
recovery hour(s) after the setpoint change. Annual results are unaffected.

DESIGN NOTES
------------
- Reporting frequency defaults to "RunPeriod" (one value per simulation =
  annual total). This minimizes output file size for the 3,400-simulation
  batch. Override to "Hourly" or "Monthly" only for selected runs where
  time-series surrogates are being explored as future work.

- Output:SQLite is REQUIRED for programmatic dataset assembly. Without
  it, output extraction would need to parse text-based ESO/MTR/HTM files
  per simulation. SQLite provides a single queryable file per run.

- Sensible vs Total: with humidity control disabled in hvac.py the
  system never dehumidifies or humidifies, but Total is still NOT equal
  to Sensible: the zone's latent load (people + moisture carried in by
  outdoor air) is still reported. 2.5 smoke run, Colombo: cooling total
  98.0 vs sensible 78.9 kWh/m2; "total heating" 10.3 vs sensible 0.76,
  i.e. humid outdoor air booked as latent "heating". The dataset targets
  are therefore the SENSIBLE variables (Decision T1, 8 Oct 2026); Total is
  kept as a diagnostic column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# ============================================================
# Type aliases
# ============================================================

ReportingFrequency = Literal[
    "Detailed", "Timestep", "Hourly", "Daily", "Monthly", "RunPeriod", "Annual"
]

SQLiteMode = Literal["Simple", "SimpleAndTabular"]

TableStyle = Literal["Comma", "Tab", "Fixed", "HTML", "XML", "All"]


# ============================================================
# Data classes
# ============================================================

@dataclass(frozen=True)
class OutputVariableRequest:
    """
    Output:Variable IDF object. Requests reporting of a specific variable
    at a specific frequency for one or all matching instances.

    key_value:
        "*" requests all instances (e.g., all zones for zone variables).
        A specific name requests only that instance.
    """
    variable_name: str
    key_value: str = "*"
    reporting_frequency: ReportingFrequency = "RunPeriod"


@dataclass(frozen=True)
class OutputMeterRequest:
    """
    Output:Meter IDF object. Requests reporting of a building-level
    meter at a specific frequency.
    """
    meter_name: str
    reporting_frequency: ReportingFrequency = "RunPeriod"


@dataclass(frozen=True)
class OutputTableSummary:
    """
    Output:Table:SummaryReports IDF object. Requests a set of predefined
    tabular reports.
    """
    report_names: tuple[str, ...]


@dataclass(frozen=True)
class OutputSet:
    """Complete output configuration for one simulation."""
    variables: tuple[OutputVariableRequest, ...]
    meters: tuple[OutputMeterRequest, ...]
    summary_reports: OutputTableSummary
    sqlite_mode: SQLiteMode = "SimpleAndTabular"
    table_style: TableStyle = "HTML"


# ============================================================
# Standard output configuration
# ============================================================

# Per-zone time-resolved variables. Key Value "*" expands to all zones.
_STANDARD_VARIABLES: tuple[OutputVariableRequest, ...] = (
    # --- Zone-level heating/cooling demand (for zone-level breakdown) ---
    OutputVariableRequest("Zone Ideal Loads Zone Sensible Heating Energy"),
    OutputVariableRequest("Zone Ideal Loads Zone Sensible Cooling Energy"),
    OutputVariableRequest("Zone Ideal Loads Zone Total Heating Energy"),
    OutputVariableRequest("Zone Ideal Loads Zone Total Cooling Energy"),

    # --- Zone air state (for unmet-hours validation and diagnostics) ---
    OutputVariableRequest("Zone Mean Air Temperature"),
    OutputVariableRequest("Zone Air Relative Humidity"),

    # --- Solar gains (diagnostic: confirms WWR + SHGC are working) ---
    OutputVariableRequest("Zone Windows Total Transmitted Solar Radiation Energy"),

    # --- Outside conditions (sanity check on EPW reading) ---
    OutputVariableRequest("Site Outdoor Air Drybulb Temperature", key_value="Environment"),
    OutputVariableRequest("Site Direct Solar Radiation Rate per Area", key_value="Environment"),
)

# Whole-building energy meters. Sum of these / floor area = EUI.
_STANDARD_METERS: tuple[OutputMeterRequest, ...] = (
    # --- Thermal demand (from IdealLoads) ---
    OutputMeterRequest("Heating:EnergyTransfer"),
    OutputMeterRequest("Cooling:EnergyTransfer"),

    # --- Electrical loads (lights + plug loads) ---
    OutputMeterRequest("InteriorLights:Electricity"),
    OutputMeterRequest("InteriorEquipment:Electricity"),

    # --- Whole-building totals (cross-check) ---
    OutputMeterRequest("Electricity:Facility"),
    OutputMeterRequest("EnergyTransfer:Facility"),
)

# Hourly time series used ONLY for peak loads (Decision T4). Always added,
# in addition to the annual (RunPeriod) copies of the same meters above.
_PEAK_METERS: tuple[OutputMeterRequest, ...] = (
    OutputMeterRequest("Heating:EnergyTransfer", reporting_frequency="Hourly"),
    OutputMeterRequest("Cooling:EnergyTransfer", reporting_frequency="Hourly"),
)

# Summary reports: predefined tables. Avoid "AllSummary" (too verbose).
_STANDARD_SUMMARY = OutputTableSummary(
    report_names=(
        # End-use annual totals (cross-check vs. meters)
        "AnnualBuildingUtilityPerformanceSummary",
        # Peak demand by end use (gives peak_cooling_load, peak_heating_load)
        "DemandEndUseComponentsSummary",
        # Input verification + zone areas/loads
        "InputVerificationandResultsSummary",
        # Per-zone "Time Setpoint Not Met". Building-level unmet hours
        # ("Comfort and Setpoint Not Met Summary") are a sub-table of
        # AnnualBuildingUtilityPerformanceSummary in EnergyPlus 24.1 - there
        # is no standalone ComfortAndSetpointNotMetSummary report.
        "SystemSummary",
        # Heat-gain breakdown (diagnostic: shows envelope/solar/internal split)
        "SensibleHeatGainSummary",
        # Per-zone cooling/heating breakdown (monthly form is the only one in 24.1)
        "ZoneCoolingSummaryMonthly",
        "ZoneHeatingSummaryMonthly",
        # Envelope-specific reports (diagnostic)
        "EnvelopeSummary",
        "WindowEnergyZoneSummaryMonthly",
    ),
)


# ============================================================
# Top-level API
# ============================================================

def build_outputs(
    reporting_frequency: ReportingFrequency = "RunPeriod",
) -> OutputSet:
    """
    Build the standard output configuration.

    Parameters
    ----------
    reporting_frequency : EnergyPlus reporting frequency for Output:Variable
        and Output:Meter requests (the hourly peak-load meters, _PEAK_METERS,
        are always added on top). Default 'RunPeriod' (one annual value
        per simulation). Use 'Hourly' or 'Monthly' only for selected
        runs where time-series outputs are needed.

    Returns
    -------
    OutputSet with standard variables, meters, summary reports, and
    SQLite output enabled.
    """
    _validate_frequency(reporting_frequency)

    # Re-create variables/meters with the requested frequency (since
    # the constants above use the default "RunPeriod").
    variables = tuple(
        OutputVariableRequest(
            variable_name=v.variable_name,
            key_value=v.key_value,
            reporting_frequency=reporting_frequency,
        )
        for v in _STANDARD_VARIABLES
    )
    meters = tuple(
        OutputMeterRequest(
            meter_name=m.meter_name,
            reporting_frequency=reporting_frequency,
        )
        for m in _STANDARD_METERS
    )

    # Peak-load meters (T4): appended unless the same (meter, frequency)
    # is already requested, e.g. when reporting_frequency="Hourly".
    present = {(m.meter_name, m.reporting_frequency) for m in meters}
    meters = meters + tuple(m for m in _PEAK_METERS
                            if (m.meter_name, m.reporting_frequency) not in present)

    return OutputSet(
        variables=variables,
        meters=meters,
        summary_reports=_STANDARD_SUMMARY,
        sqlite_mode="SimpleAndTabular",
        table_style="HTML",
    )


# ============================================================
# Validation
# ============================================================

_VALID_FREQUENCIES = {
    "Detailed", "Timestep", "Hourly", "Daily", "Monthly",
    "RunPeriod", "Annual",
}


def _validate_frequency(freq: str) -> None:
    if freq not in _VALID_FREQUENCIES:
        raise ValueError(
            f"reporting_frequency must be one of {sorted(_VALID_FREQUENCIES)}, "
            f"got {freq!r}"
        )
