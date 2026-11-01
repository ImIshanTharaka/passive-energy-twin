"""
hvac.py - Ideal Loads HVAC specifications.

This module produces HVAC system specifications for a passive-design
study using EnergyPlus's IdealLoadsAirSystem. The Ideal Loads approach
strips away HVAC equipment efficiency, control complexity, and sizing
decisions so that the reported heating/cooling energy is purely the
thermal load the envelope imposes on the building.

WHY IDEAL LOADS
---------------
The research question concerns passive (envelope) design, NOT HVAC
selection. A real HVAC system would introduce three confounders:
    1. Equipment efficiency (chiller COP, boiler efficiency)
    2. Control strategy (supply reset, economizer, etc.)
    3. Autosizing decisions (over- or under-sized systems)
These would mix with the envelope effect we want to isolate. Ideal
Loads removes all three: infinite capacity, perfect efficiency, no
controls. The output 'Zone Ideal Loads Heating/Cooling Energy' is the
pure envelope load.

This is the universal approach in surrogate-modeling literature on
passive design (Shirzadi 2025, Stevanovic 2024, BESOS framework, etc.).

WHAT THIS MODULE PRODUCES
-------------------------
  ThermostatSpec           - one shared thermostat (references heating
                             and cooling setpoint schedules from
                             internal_loads.py)
  IdealLoadsZoneSpec       - one per zone, references the thermostat,
                             plus OA, availability, and control settings

generator.py translates these into:
  HVACTemplate:Thermostat                  (1 total)
  HVACTemplate:Zone:IdealLoadsAirSystem    (one per zone)

EnergyPlus's ExpandObjects preprocessor expands these into the dozen
underlying objects (Sizing:Zone, ZoneControl:Thermostat,
ZoneHVAC:IdealLoadsAirSystem, etc.).

DELIBERATE SIMPLIFICATIONS
---------------------------
The following IdealLoads features are turned OFF. Each is a deliberate
choice to keep the analysis focused on envelope demand:
  Economizer:              NoEconomizer     (active control out of scope)
  Heat recovery:           None             (equipment-level efficiency)
  Dehumidification:        None             (latent loads out of scope)
  Humidification:          None             (latent loads out of scope)
  Demand-controlled vent.: None             (active control out of scope)
  Heating/cooling limits:  NoLimit          (infinite capacity, the
                                              whole point of IdealLoads)

KEY LIMITATION: No latent (humidity) handling. For very humid climates
(Colombo!), this is a known simplification that future work could
address by adding sensible+latent surrogate outputs. The cooling demand
we report is sensible-only (Decision T1). Note the "Zone Ideal Loads Zone
Total" variables are NOT equal to Sensible even with humidity control off:
they still carry the latent load of people and outdoor air (2.5 smoke run,
Colombo: latent = 20% of total cooling; "total heating" 10.3 kWh/m2 vs
0.76 sensible). See outputs.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..geometry.geometry import BuildingGeometry
    from ..internal_loads.internal_loads import InternalLoadsSet


# ============================================================
# Default supply air conditions (EnergyPlus defaults; unchanged)
# ============================================================
# These affect supply-air mass flow rate but NOT total annual heating/
# cooling energy. The zone heat balance is set by setpoints and
# envelope, regardless of supply temperature.
DEFAULT_MAX_HEATING_SUPPLY_TEMP = 50.0           # C
DEFAULT_MIN_COOLING_SUPPLY_TEMP = 13.0           # C
DEFAULT_MAX_HEATING_SUPPLY_HR = 0.0156           # kg water / kg dry air
DEFAULT_MIN_COOLING_SUPPLY_HR = 0.0077


# ============================================================
# Data classes
# ============================================================

@dataclass(frozen=True)
class ThermostatSpec:
    """
    HVACTemplate:Thermostat object. References the two setpoint schedules
    defined in internal_loads.py. One thermostat shared by all zones.
    """
    name: str
    heating_setpoint_schedule_name: str
    cooling_setpoint_schedule_name: str


@dataclass(frozen=True)
class IdealLoadsZoneSpec:
    """
    HVACTemplate:Zone:IdealLoadsAirSystem object. One per zone.

    Most fields are EnergyPlus defaults (kept explicit here so they
    show up in the IDF rather than relying on EnergyPlus's defaults
    silently). The overrides from default are:
      outdoor_air_method = "Sum"
      oa_per_person, oa_per_area from internal_loads
      heat_recovery_type = "None"
      economizer_type = "NoEconomizer"
      dehumidification, humidification = "None"
    """
    name: str
    zone_name: str
    thermostat_name: str
    system_availability_schedule_name: str

    # Supply air conditions
    max_heating_supply_temp: float = DEFAULT_MAX_HEATING_SUPPLY_TEMP
    min_cooling_supply_temp: float = DEFAULT_MIN_COOLING_SUPPLY_TEMP
    max_heating_supply_humidity_ratio: float = DEFAULT_MAX_HEATING_SUPPLY_HR
    min_cooling_supply_humidity_ratio: float = DEFAULT_MIN_COOLING_SUPPLY_HR

    # Capacity limits ("NoLimit" = infinite capacity, the IdealLoads point)
    heating_limit: str = "NoLimit"
    cooling_limit: str = "NoLimit"

    # Humidity control: OFF (sensible-only analysis)
    dehumidification_control_type: str = "None"
    humidification_control_type: str = "None"

    # Outdoor air: ASHRAE 62.1 Sum method
    outdoor_air_method: str = "Sum"
    oa_per_person: float = 0.0025                # m^3/s per person
    oa_per_area: float = 0.0003                  # m^3/s per m^2
    oa_per_zone: float = 0.0                     # m^3/s per zone (not used)

    # Active control features: OFF (envelope-only analysis)
    demand_controlled_ventilation_type: str = "None"
    economizer_type: str = "NoEconomizer"
    heat_recovery_type: str = "None"


@dataclass(frozen=True)
class HVACSet:
    """Complete HVAC specification for one simulation."""
    thermostat: ThermostatSpec
    zone_systems: tuple[IdealLoadsZoneSpec, ...]


# ============================================================
# Top-level API
# ============================================================

def build_hvac(
    geometry: "BuildingGeometry",
    internal_loads: "InternalLoadsSet",
    thermostat_name: str = "OfficeThermostat",
) -> HVACSet:
    """
    Build the HVAC system spec: one thermostat + one IdealLoads system
    per zone.

    Parameters
    ----------
    geometry : BuildingGeometry, used to enumerate zones
    internal_loads : InternalLoadsSet, source of schedule names and OA
    thermostat_name : optional override (default 'OfficeThermostat')

    Returns
    -------
    HVACSet with one ThermostatSpec and one IdealLoadsZoneSpec per zone.
    """
    if not geometry.zones:
        raise ValueError("Cannot build HVAC: geometry has no zones")

    # Single shared thermostat
    thermostat = ThermostatSpec(
        name=thermostat_name,
        heating_setpoint_schedule_name=internal_loads.heating_setpoint_schedule_name,
        cooling_setpoint_schedule_name=internal_loads.cooling_setpoint_schedule_name,
    )

    # One IdealLoads per zone
    zone_systems = tuple(
        IdealLoadsZoneSpec(
            name=f"{zone.name}_IdealLoads",
            zone_name=zone.name,
            thermostat_name=thermostat.name,
            system_availability_schedule_name=internal_loads.always_on_schedule_name,
            oa_per_person=internal_loads.oa_per_person,
            oa_per_area=internal_loads.oa_per_area,
        )
        for zone in geometry.zones
    )

    return HVACSet(thermostat=thermostat, zone_systems=zone_systems)
