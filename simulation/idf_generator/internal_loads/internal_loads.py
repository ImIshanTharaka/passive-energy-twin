"""
internal_loads.py - Fixed internal loads, schedules, and operations.

This module declares ALL internal-load parameters and operational
schedules that are held constant across simulations. None of these are
sampled by LHS - they are anchored at DOE Medium Office prototype
values so the design space focuses on passive envelope, geometry, and
fenestration parameters only.

The single sampled parameter related to operations (infiltration rate,
in ACH at 50 Pa) is NOT in this module. Its rate is applied to per-zone
ZoneInfiltration:DesignFlowRate objects in generator.py, multiplied by
the infiltration *schedule* defined here.

ANCHORED VALUES
---------------
Source: ASHRAE 90.1-2019 Medium Office prototype (DOE Commercial
Reference Buildings)

  Occupant density:          0.0538 people/m^2   (1 per 18.6 m^2)
  Lighting power density:    6.9    W/m^2        (ASHRAE 90.1-2019)
  Equipment power density:   6.78   W/m^2        (DOE prototype)
  Activity level:            120    W/person     (ASHRAE 55 seated work)

  Outdoor air per person:    0.0025 m^3/s        (ASHRAE 62.1, office)
  Outdoor air per area:      0.0003 m^3/s/m^2    (ASHRAE 62.1, office)

  Setpoints (occupied / unoccupied):
    Heating:                 21.0  / 15.6  C
    Cooling:                 24.0  / 26.7  C

  HVAC operation:
    Weekdays:                06:00 - 22:00 occupied
    Saturday:                06:00 - 18:00 occupied
    Sunday/holidays:         setback all day
    HVAC always available (24/7); setbacks throttle the system, they
    do not turn it off. This is Strategy A in the module documentation.

SCHEDULES PRODUCED
------------------
  OfficeOccupancy        - 0..1 fraction of design occupancy
  OfficeLighting         - 0..1 fraction of design LPD
  OfficeEquipment        - 0..1 fraction of design EPD
  OfficeActivity         - W/person (constant 120)
  HeatingSetpoint        - temperature schedule (21 / 15.6 C)
  CoolingSetpoint        - temperature schedule (24 / 26.7 C)
  InfiltrationSchedule   - 0..1 multiplier (1.0 unoccupied, 0.25 occ.)
  AlwaysOn               - constant 1.0 (HVAC system availability)

Each schedule covers Weekdays + SummerDesignDay + WinterDesignDay (peak
pattern), Saturday, and AllOtherDays (catch-all for Sunday, holidays,
custom days). The two design days share the peak weekday pattern so
that EnergyPlus's sizing run reflects realistic peak loads.

DOWNSTREAM USE
--------------
hvac.py reads the heating/cooling setpoint and always_on schedule names.
generator.py creates one People, Lights, ElectricEquipment, and
ZoneInfiltration object per zone, all referencing the relevant schedule
by name and multiplying by the per-zone floor area.
"""

from __future__ import annotations

from dataclasses import dataclass


# ============================================================
# Fixed values (exported as module-level constants for reuse)
# ============================================================

# Per-area densities (applied uniformly to all zones)
OCCUPANT_DENSITY = 0.0538           # people/m^2
LIGHTING_POWER_DENSITY = 6.9        # W/m^2
EQUIPMENT_POWER_DENSITY = 6.78      # W/m^2

# Per-person values
ACTIVITY_LEVEL = 120.0              # W/person (ASHRAE 55, seated)

# Outdoor air requirements (ASHRAE 62.1, office)
OA_PER_PERSON = 0.0025              # m^3/s per person
OA_PER_AREA = 0.0003                # m^3/s per m^2

# Thermostat setpoints
HEATING_SETPOINT_OCCUPIED = 21.0    # C
HEATING_SETPOINT_UNOCCUPIED = 15.6  # C  (= 60 F, DOE setback)
COOLING_SETPOINT_OCCUPIED = 24.0    # C
COOLING_SETPOINT_UNOCCUPIED = 26.7  # C  (= 80 F, DOE setup)

# Infiltration multiplier
INFIL_FRACTION_OCCUPIED = 0.25      # HVAC pressurizes building
INFIL_FRACTION_UNOCCUPIED = 1.0     # full nominal infiltration


# ============================================================
# Data classes
# ============================================================

@dataclass(frozen=True)
class ScheduleTypeLimitsSpec:
    """
    EnergyPlus ScheduleTypeLimits object. Validates that schedule values
    fall within allowed ranges. Required for every Schedule:Compact.

    numeric_type:  "Continuous" or "Discrete"
    unit_type:     "Dimensionless", "Temperature", "ActivityLevel", etc.
    """
    name: str
    lower_limit: float | None = None
    upper_limit: float | None = None
    numeric_type: str = "Continuous"
    unit_type: str = "Dimensionless"


@dataclass(frozen=True)
class ScheduleInterval:
    """
    One (until_time, value) entry in a day pattern. Interpreted as:
    'this value applies from the previous Until time up to until_time'.
    """
    until_time: str      # "HH:MM" 24-hour
    value: float


@dataclass(frozen=True)
class ScheduleDayPattern:
    """
    One 'For: <day_types>' block in a Schedule:Compact. The intervals
    must end at 24:00 to fully cover one day.

    day_types: tuple of EnergyPlus day-type strings, e.g.
        ("Weekdays", "SummerDesignDay", "WinterDesignDay")
        ("Saturday",)
        ("AllOtherDays",)   # catch-all for Sunday, Holiday, etc.
    """
    day_types: tuple[str, ...]
    intervals: tuple[ScheduleInterval, ...]


@dataclass(frozen=True)
class ScheduleSpec:
    """
    EnergyPlus Schedule:Compact object, valid for the full year.

    The current model has no seasonal variation, so through_date is
    "12/31" for all schedules.
    """
    name: str
    schedule_type_limits: str        # name of a ScheduleTypeLimitsSpec
    day_patterns: tuple[ScheduleDayPattern, ...]
    through_date: str = "12/31"


@dataclass(frozen=True)
class InternalLoadsSet:
    """
    Complete fixed loads + schedules for one simulation. Returned by
    build_internal_loads().
    """
    # Densities and rates
    occupant_density: float
    lighting_power_density: float
    equipment_power_density: float
    activity_level: float
    oa_per_person: float
    oa_per_area: float

    # Schedule definitions
    schedule_type_limits: tuple[ScheduleTypeLimitsSpec, ...]
    schedules: tuple[ScheduleSpec, ...]

    # Named references (consumed by hvac.py and generator.py)
    occupancy_schedule_name: str
    lighting_schedule_name: str
    equipment_schedule_name: str
    activity_schedule_name: str
    heating_setpoint_schedule_name: str
    cooling_setpoint_schedule_name: str
    infiltration_schedule_name: str
    always_on_schedule_name: str

    def schedule_by_name(self, name: str) -> ScheduleSpec:
        for s in self.schedules:
            if s.name == name:
                return s
        raise KeyError(f"Schedule {name!r} not in InternalLoadsSet")

    def type_limits_by_name(self, name: str) -> ScheduleTypeLimitsSpec:
        for t in self.schedule_type_limits:
            if t.name == name:
                return t
        raise KeyError(f"ScheduleTypeLimits {name!r} not in InternalLoadsSet")


# ============================================================
# Schedule type limits (4 distinct types used)
# ============================================================

_FRACTION_LIMITS = ScheduleTypeLimitsSpec(
    name="Fraction",
    lower_limit=0.0, upper_limit=1.0,
    numeric_type="Continuous", unit_type="Dimensionless",
)

_TEMPERATURE_LIMITS = ScheduleTypeLimitsSpec(
    name="Temperature",
    lower_limit=-60.0, upper_limit=100.0,
    numeric_type="Continuous", unit_type="Temperature",
)

_ACTIVITY_LIMITS = ScheduleTypeLimitsSpec(
    name="ActivityLevel",
    lower_limit=0.0,
    numeric_type="Continuous", unit_type="ActivityLevel",
)

_ONOFF_LIMITS = ScheduleTypeLimitsSpec(
    name="OnOff",
    lower_limit=0.0, upper_limit=1.0,
    numeric_type="Discrete", unit_type="Dimensionless",
)


# ============================================================
# Common day-type groups
# ============================================================

_WEEKDAYS_AND_DESIGN = ("Weekdays", "SummerDesignDay", "WinterDesignDay")
_SATURDAY = ("Saturday",)
_ALL_OTHER_DAYS = ("AllOtherDays",)
_ALL_DAYS = ("AllDays",)


# ============================================================
# Schedules (DOE Medium Office reference patterns)
# ============================================================

# --- Occupancy: peak 0.95, lunch dip to 0.50 ---
_OCCUPANCY_SCHEDULE = ScheduleSpec(
    name="OfficeOccupancy",
    schedule_type_limits="Fraction",
    day_patterns=(
        ScheduleDayPattern(
            day_types=_WEEKDAYS_AND_DESIGN,
            intervals=(
                ScheduleInterval("06:00", 0.00),
                ScheduleInterval("07:00", 0.10),
                ScheduleInterval("08:00", 0.20),
                ScheduleInterval("12:00", 0.95),
                ScheduleInterval("13:00", 0.50),   # lunch
                ScheduleInterval("17:00", 0.95),
                ScheduleInterval("18:00", 0.30),
                ScheduleInterval("20:00", 0.10),
                ScheduleInterval("22:00", 0.05),
                ScheduleInterval("24:00", 0.00),
            ),
        ),
        ScheduleDayPattern(
            day_types=_SATURDAY,
            intervals=(
                ScheduleInterval("06:00", 0.00),
                ScheduleInterval("08:00", 0.10),
                ScheduleInterval("14:00", 0.50),
                ScheduleInterval("17:00", 0.10),
                ScheduleInterval("24:00", 0.00),
            ),
        ),
        ScheduleDayPattern(
            day_types=_ALL_OTHER_DAYS,
            intervals=(
                ScheduleInterval("24:00", 0.00),
            ),
        ),
    ),
)

# --- Lighting: peak 0.90, longer ramp, 0.05 baseline overnight ---
_LIGHTING_SCHEDULE = ScheduleSpec(
    name="OfficeLighting",
    schedule_type_limits="Fraction",
    day_patterns=(
        ScheduleDayPattern(
            day_types=_WEEKDAYS_AND_DESIGN,
            intervals=(
                ScheduleInterval("05:00", 0.05),
                ScheduleInterval("07:00", 0.10),
                ScheduleInterval("08:00", 0.30),
                ScheduleInterval("17:00", 0.90),
                ScheduleInterval("18:00", 0.50),
                ScheduleInterval("20:00", 0.30),
                ScheduleInterval("22:00", 0.20),
                ScheduleInterval("24:00", 0.05),
            ),
        ),
        ScheduleDayPattern(
            day_types=_SATURDAY,
            intervals=(
                ScheduleInterval("06:00", 0.05),
                ScheduleInterval("08:00", 0.10),
                ScheduleInterval("14:00", 0.30),
                ScheduleInterval("17:00", 0.15),
                ScheduleInterval("24:00", 0.05),
            ),
        ),
        ScheduleDayPattern(
            day_types=_ALL_OTHER_DAYS,
            intervals=(
                ScheduleInterval("24:00", 0.05),
            ),
        ),
    ),
)

# --- Equipment: higher overnight baseline 0.30 (computers stay on) ---
_EQUIPMENT_SCHEDULE = ScheduleSpec(
    name="OfficeEquipment",
    schedule_type_limits="Fraction",
    day_patterns=(
        ScheduleDayPattern(
            day_types=_WEEKDAYS_AND_DESIGN,
            intervals=(
                ScheduleInterval("07:00", 0.30),
                ScheduleInterval("08:00", 0.40),
                ScheduleInterval("12:00", 0.90),
                ScheduleInterval("13:00", 0.80),   # slight lunch dip
                ScheduleInterval("17:00", 0.90),
                ScheduleInterval("18:00", 0.50),
                ScheduleInterval("20:00", 0.40),
                ScheduleInterval("24:00", 0.30),
            ),
        ),
        ScheduleDayPattern(
            day_types=_SATURDAY,
            intervals=(
                ScheduleInterval("08:00", 0.30),
                ScheduleInterval("14:00", 0.50),
                ScheduleInterval("17:00", 0.40),
                ScheduleInterval("24:00", 0.30),
            ),
        ),
        ScheduleDayPattern(
            day_types=_ALL_OTHER_DAYS,
            intervals=(
                ScheduleInterval("24:00", 0.30),
            ),
        ),
    ),
)

# --- Activity level: constant 120 W/person ---
_ACTIVITY_SCHEDULE = ScheduleSpec(
    name="OfficeActivity",
    schedule_type_limits="ActivityLevel",
    day_patterns=(
        ScheduleDayPattern(
            day_types=_ALL_DAYS,
            intervals=(
                ScheduleInterval("24:00", ACTIVITY_LEVEL),
            ),
        ),
    ),
)

# --- Heating setpoint: 21 occupied, 15.6 setback ---
_HEATING_SETPOINT_SCHEDULE = ScheduleSpec(
    name="HeatingSetpoint",
    schedule_type_limits="Temperature",
    day_patterns=(
        ScheduleDayPattern(
            day_types=_WEEKDAYS_AND_DESIGN,
            intervals=(
                ScheduleInterval("06:00", HEATING_SETPOINT_UNOCCUPIED),
                ScheduleInterval("22:00", HEATING_SETPOINT_OCCUPIED),
                ScheduleInterval("24:00", HEATING_SETPOINT_UNOCCUPIED),
            ),
        ),
        ScheduleDayPattern(
            day_types=_SATURDAY,
            intervals=(
                ScheduleInterval("06:00", HEATING_SETPOINT_UNOCCUPIED),
                ScheduleInterval("18:00", HEATING_SETPOINT_OCCUPIED),
                ScheduleInterval("24:00", HEATING_SETPOINT_UNOCCUPIED),
            ),
        ),
        ScheduleDayPattern(
            day_types=_ALL_OTHER_DAYS,
            intervals=(
                ScheduleInterval("24:00", HEATING_SETPOINT_UNOCCUPIED),
            ),
        ),
    ),
)

# --- Cooling setpoint: 24 occupied, 26.7 setup ---
_COOLING_SETPOINT_SCHEDULE = ScheduleSpec(
    name="CoolingSetpoint",
    schedule_type_limits="Temperature",
    day_patterns=(
        ScheduleDayPattern(
            day_types=_WEEKDAYS_AND_DESIGN,
            intervals=(
                ScheduleInterval("06:00", COOLING_SETPOINT_UNOCCUPIED),
                ScheduleInterval("22:00", COOLING_SETPOINT_OCCUPIED),
                ScheduleInterval("24:00", COOLING_SETPOINT_UNOCCUPIED),
            ),
        ),
        ScheduleDayPattern(
            day_types=_SATURDAY,
            intervals=(
                ScheduleInterval("06:00", COOLING_SETPOINT_UNOCCUPIED),
                ScheduleInterval("18:00", COOLING_SETPOINT_OCCUPIED),
                ScheduleInterval("24:00", COOLING_SETPOINT_UNOCCUPIED),
            ),
        ),
        ScheduleDayPattern(
            day_types=_ALL_OTHER_DAYS,
            intervals=(
                ScheduleInterval("24:00", COOLING_SETPOINT_UNOCCUPIED),
            ),
        ),
    ),
)

# --- Infiltration: inverse of HVAC operation (0.25 occ / 1.0 unocc) ---
_INFILTRATION_SCHEDULE = ScheduleSpec(
    name="InfiltrationSchedule",
    schedule_type_limits="Fraction",
    day_patterns=(
        ScheduleDayPattern(
            day_types=_WEEKDAYS_AND_DESIGN,
            intervals=(
                ScheduleInterval("06:00", INFIL_FRACTION_UNOCCUPIED),
                ScheduleInterval("22:00", INFIL_FRACTION_OCCUPIED),
                ScheduleInterval("24:00", INFIL_FRACTION_UNOCCUPIED),
            ),
        ),
        ScheduleDayPattern(
            day_types=_SATURDAY,
            intervals=(
                ScheduleInterval("06:00", INFIL_FRACTION_UNOCCUPIED),
                ScheduleInterval("18:00", INFIL_FRACTION_OCCUPIED),
                ScheduleInterval("24:00", INFIL_FRACTION_UNOCCUPIED),
            ),
        ),
        ScheduleDayPattern(
            day_types=_ALL_OTHER_DAYS,
            intervals=(
                ScheduleInterval("24:00", INFIL_FRACTION_UNOCCUPIED),
            ),
        ),
    ),
)

# --- Always-on: 1.0 constant (HVAC system availability) ---
_ALWAYS_ON_SCHEDULE = ScheduleSpec(
    name="AlwaysOn",
    schedule_type_limits="OnOff",
    day_patterns=(
        ScheduleDayPattern(
            day_types=_ALL_DAYS,
            intervals=(
                ScheduleInterval("24:00", 1.0),
            ),
        ),
    ),
)


# ============================================================
# Top-level API
# ============================================================

def build_internal_loads() -> InternalLoadsSet:
    """
    Build the InternalLoadsSet from fixed DOE prototype values.

    Takes no parameters - all values are constants. The function exists
    so the rest of the pipeline has a uniform 'build_X()' interface.
    """
    return InternalLoadsSet(
        # Densities and rates
        occupant_density=OCCUPANT_DENSITY,
        lighting_power_density=LIGHTING_POWER_DENSITY,
        equipment_power_density=EQUIPMENT_POWER_DENSITY,
        activity_level=ACTIVITY_LEVEL,
        oa_per_person=OA_PER_PERSON,
        oa_per_area=OA_PER_AREA,

        # Schedule type limits
        schedule_type_limits=(
            _FRACTION_LIMITS,
            _TEMPERATURE_LIMITS,
            _ACTIVITY_LIMITS,
            _ONOFF_LIMITS,
        ),

        # All schedules
        schedules=(
            _OCCUPANCY_SCHEDULE,
            _LIGHTING_SCHEDULE,
            _EQUIPMENT_SCHEDULE,
            _ACTIVITY_SCHEDULE,
            _HEATING_SETPOINT_SCHEDULE,
            _COOLING_SETPOINT_SCHEDULE,
            _INFILTRATION_SCHEDULE,
            _ALWAYS_ON_SCHEDULE,
        ),

        # Named references
        occupancy_schedule_name=_OCCUPANCY_SCHEDULE.name,
        lighting_schedule_name=_LIGHTING_SCHEDULE.name,
        equipment_schedule_name=_EQUIPMENT_SCHEDULE.name,
        activity_schedule_name=_ACTIVITY_SCHEDULE.name,
        heating_setpoint_schedule_name=_HEATING_SETPOINT_SCHEDULE.name,
        cooling_setpoint_schedule_name=_COOLING_SETPOINT_SCHEDULE.name,
        infiltration_schedule_name=_INFILTRATION_SCHEDULE.name,
        always_on_schedule_name=_ALWAYS_ON_SCHEDULE.name,
    )
