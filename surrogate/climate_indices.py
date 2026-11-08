"""
climate_indices.py - EPW-derived climate indices for climate generalisation (Sub-task 3.4, fix 1).

Plain idea: the 7 Phase 2 climate features are whole-year summaries. The energy the building
spends on outdoor (ventilation) air depends on the weather *while the office is occupied*, and
on humidity as much as temperature. These indices are computed from the hourly EPW for the
archetype's occupied hours, against the archetype's own setpoints. They contain no fitted
constants (Phase 3 plan Rev 2 §2.2; 3.4 plan F1), so a new site only needs its EPW file.

Occupied hours (generator schedules, `idf_generator.internal_loads`):
    Mon-Fri 06:00-22:00 and Sat 06:00-18:00 at the occupied setpoints (21 / 24 degC);
    Sunday set back all day. EPW hours are hour-ENDING, so the occupied hours are
    Mon-Fri hour 07..22 and Sat hour 07..18. The RunPeriod and the EPW both start on
    Sunday 1 January (`Day_of_Week_for_Start_Day="Sunday"`), and every EPW in this project
    declares no holidays and no daylight saving (checked here; anything else is refused,
    because EnergyPlus would then shift or replace day types and the mask would be wrong).

Indices (sums over occupied hours; h = moist-air enthalpy, kJ/kg dry air):
    occ_enth_excess    sum max(0, h_out - h_room)           kJ/kg.h   room = 24 degC, 50 % RH
    occ_latent_excess  sum max(0, W_out - W_room) * 2501    kJ/kg.h   latent part (moisture)
    occ_band_dh        sum clip(24 - T_out, 0, 3)           K.h       free-cooling band 21-24 degC
    occ_cool_dh        sum max(0, T_out - 24)               K.h
    occ_heat_dh        sum max(0, 21 - T_out)               K.h
    hdd15, hdd12       sum_d max(0, base - T_d), T_d = mean of 24 hourly values (2.3 convention)

Psychrometrics: ASHRAE Handbook - Fundamentals (2017), ch. 1: saturation pressure eqs. (5)-(6)
(Hyland-Wexler), humidity ratio eq. (20) with the hourly EPW station pressure, enthalpy eq. (32).
Outdoor humidity from the EPW dew point; room air at 24 degC / 50 % RH at the same pressure.

Holdout guard: `training_indices()` and `indices_for_key()` refuse the holdout climate (London)
unless `allow_holdout=True`, which only `final_test.py` (3.7) passes.

Run:  ./venv/bin/python surrogate/climate_indices.py      (prints the table for the 10 training climates)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from climate.climate_features import (daily_mean_temperature, degree_days, holdout_keys,
                                      load_climate_registry, training_keys)
from climate.epw_parser import EPWData, parse_epw
from idf_generator.internal_loads.internal_loads import (COOLING_SETPOINT_OCCUPIED,
                                                         HEATING_SETPOINT_OCCUPIED)
from surrogate.paths import EPW_DIR

# ----------------------------------------------------------------------------
# Fixed constants (archetype physics; not tuned - 3.4 plan F1, master plan M6)
# ----------------------------------------------------------------------------
T_HEAT = HEATING_SETPOINT_OCCUPIED        # 21.0 degC (generator)
T_COOL = COOLING_SETPOINT_OCCUPIED        # 24.0 degC (generator)
RH_ROOM = 0.50                            # room relative humidity for the enthalpy reference
BAND_WIDTH = T_COOL - T_HEAT              # 3 K: free-cooling band 21-24 degC
HDD_BASES = (15.0, 12.0)                  # balance-point degree-days (internal gains)
H_FG0 = 2501.0                            # kJ/kg, latent heat of vaporisation at 0 degC
CP_DA, CP_WV = 1.006, 1.86                # kJ/kg.K, dry air and water vapour
MW_RATIO = 0.621945                       # molar-mass ratio water / dry air

# Occupied hours, hour-ending (weekday 0 = Sunday, as the RunPeriod start day)
START_WEEKDAY = 0                         # 1 January = Sunday
OCC_HOURS = {1: range(7, 23), 2: range(7, 23), 3: range(7, 23), 4: range(7, 23),
             5: range(7, 23), 6: range(7, 19)}   # Mon..Fri 07..22, Sat 07..18; Sun none

INDEX_COLUMNS = ["occ_enth_excess", "occ_latent_excess", "occ_band_dh", "occ_cool_dh",
                 "occ_heat_dh", "hdd15", "hdd12"]
OA_LEVEL_INDICES = ["occ_enth_excess", "occ_band_dh", "occ_heat_dh"]   # F3 straight line


class HoldoutClimateError(RuntimeError):
    """Raised when the holdout climate is requested outside the final test (3.7)."""


# ----------------------------------------------------------------------------
# Psychrometrics (vectorised, SI)
# ----------------------------------------------------------------------------

def saturation_pressure(t_c) -> np.ndarray:
    """Saturation vapour pressure (Pa) over ice (t < 0 degC) or liquid water (t >= 0 degC).

    ASHRAE Fundamentals 2017 ch. 1, eqs. (5) and (6) (Hyland and Wexler 1983).
    """
    t = np.asarray(t_c, dtype=float) + 273.15
    ice = (-5.6745359e3 / t + 6.3925247 - 9.677843e-3 * t + 6.2215701e-7 * t ** 2
           + 2.0747825e-9 * t ** 3 - 9.484024e-13 * t ** 4 + 4.1635019 * np.log(t))
    water = (-5.8002206e3 / t + 1.3914993 - 4.8640239e-2 * t + 4.1764768e-5 * t ** 2
             - 1.4452093e-8 * t ** 3 + 6.5459673 * np.log(t))
    return np.exp(np.where(t < 273.15, ice, water))


def humidity_ratio(p_w, p_atm) -> np.ndarray:
    """Humidity ratio W (kg water / kg dry air) from vapour and total pressure (Pa); eq. (20)."""
    p_w, p_atm = np.asarray(p_w, dtype=float), np.asarray(p_atm, dtype=float)
    return MW_RATIO * p_w / (p_atm - p_w)


def enthalpy(t_c, w) -> np.ndarray:
    """Moist-air specific enthalpy (kJ/kg dry air); eq. (32)."""
    t_c, w = np.asarray(t_c, dtype=float), np.asarray(w, dtype=float)
    return CP_DA * t_c + w * (H_FG0 + CP_WV * t_c)


# ----------------------------------------------------------------------------
# Occupied-hour mask and indices
# ----------------------------------------------------------------------------

def occupied_mask(n_hours: int, hour_ending: np.ndarray | None = None) -> np.ndarray:
    """Boolean mask of occupied hours for a year starting on Sunday 1 January.

    `hour_ending` (EPW hour field, 1..24) is checked against the row position when given.
    """
    if n_hours % 24:
        raise ValueError(f"{n_hours} hours is not a whole number of days")
    rows = np.arange(n_hours)
    hour = rows % 24 + 1
    if hour_ending is not None and not np.array_equal(np.asarray(hour_ending, dtype=int), hour):
        raise ValueError("EPW hour column is not 1..24 in row order")
    weekday = (rows // 24 + START_WEEKDAY) % 7
    mask = np.zeros(n_hours, dtype=bool)
    for wd, hours in OCC_HOURS.items():
        mask |= (weekday == wd) & np.isin(hour, list(hours))
    return mask


def _check_calendar(epw_path: Path) -> None:
    """Refuse EPWs whose holidays or daylight saving would change EnergyPlus's day types."""
    with open(epw_path, encoding="latin-1") as fh:
        for line in fh:
            if line.upper().startswith("HOLIDAYS/DAYLIGHT SAVINGS"):
                f = [x.strip() for x in line.split(",")]
                if f[1].upper() != "NO" or any(x not in ("0", "") for x in f[2:5]):
                    raise ValueError(f"{epw_path.name}: holidays/daylight saving declared ({line.strip()}); "
                                     "the occupied-hour mask assumes none")
                return
    raise ValueError(f"{epw_path.name}: no HOLIDAYS/DAYLIGHT SAVINGS header line")


def indices_from_epw(epw: EPWData) -> dict[str, float]:
    """The 7 indices (INDEX_COLUMNS) for one parsed EPW. No site key, no holdout guard:
    this is the path a new site takes in Phases 4-5."""
    _check_calendar(Path(epw.path))
    occ = occupied_mask(epw.n_hours, epw.hour)
    t, p = epw.dry_bulb, epw.atm_pressure

    w_out = humidity_ratio(saturation_pressure(epw.dew_point), p)
    w_room = humidity_ratio(RH_ROOM * saturation_pressure(np.full_like(t, T_COOL)), p)
    h_out, h_room = enthalpy(t, w_out), enthalpy(np.full_like(t, T_COOL), w_room)

    daily = daily_mean_temperature(t)
    out = {
        "occ_enth_excess": float(np.clip(h_out - h_room, 0, None)[occ].sum()),
        "occ_latent_excess": float((np.clip(w_out - w_room, 0, None) * H_FG0)[occ].sum()),
        "occ_band_dh": float(np.clip(T_COOL - t, 0, BAND_WIDTH)[occ].sum()),
        "occ_cool_dh": float(np.clip(t - T_COOL, 0, None)[occ].sum()),
        "occ_heat_dh": float(np.clip(T_HEAT - t, 0, None)[occ].sum()),
        "hdd15": degree_days(daily, HDD_BASES[0], "heating"),
        "hdd12": degree_days(daily, HDD_BASES[1], "heating"),
    }
    if not all(math.isfinite(v) for v in out.values()):
        raise ValueError(f"{epw.path}: non-finite climate index {out}")
    return out


def indices_for_key(key: str, allow_holdout: bool = False) -> dict[str, float]:
    """Indices for one registered climate key, read from `EPW_DIR/<key>.epw`."""
    registry = load_climate_registry(check_files=False)   # only the EPW actually read is needed
    if key not in registry:
        raise KeyError(f"unknown climate key {key!r}")
    if key in holdout_keys(registry) and not allow_holdout:
        raise HoldoutClimateError(f"{key!r} is the holdout climate: its indices are computed "
                                  "only inside final_test.py (3.7)")
    return indices_from_epw(parse_epw(EPW_DIR / registry[key].epw_filename))


def training_indices() -> pd.DataFrame:
    """Index table for the 10 training climates (index = climate_key, columns = INDEX_COLUMNS)."""
    keys = training_keys(load_climate_registry(check_files=False))
    table = pd.DataFrame([indices_for_key(k) for k in keys], index=pd.Index(keys, name="climate_key"))
    return table[INDEX_COLUMNS]


def add_indices(df: pd.DataFrame, table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return `df` with the 7 index columns joined on `climate_key` (same index and row order).

    Raises if any row's climate has no indices (e.g. the holdout climate in a training table).
    """
    table = training_indices() if table is None else table
    missing = sorted(set(df["climate_key"]) - set(table.index))
    if missing:
        raise KeyError(f"no climate indices for {missing}")
    joined = table.loc[df["climate_key"].to_numpy(), INDEX_COLUMNS].to_numpy()
    out = df.copy()
    for j, col in enumerate(INDEX_COLUMNS):
        out[col] = joined[:, j]
    return out


def main() -> int:
    table = training_indices()
    pd.set_option("display.width", 140)
    print("EPW-derived climate indices - 10 training climates (occupied hours, archetype setpoints)\n")
    print(table.round(1).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
