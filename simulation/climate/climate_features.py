"""
climate_features.py - Climate feature computation and the climate registry.

Two jobs:

  1. COMPRESS a year of hourly weather (8760 hours x several fields) into
     the eight climate features used as surrogate-model inputs.

  2. BE THE REGISTRY: the single answer to "which weather file is
     'chicago', what role does it play, and what are its eight numbers?"
     Every downstream component asks that question through this module.

WHAT THIS MODULE PRODUCES
-------------------------
  ClimateFeatures      - the 8 ML input features (7 continuous + zone label)
  ClimateDiagnostics   - reporting/QA quantities, NEVER model inputs
  ClimateRecord        - one climate: identity + role + path + both of above
  climate_metadata.csv - the stored artefact (one row per climate)

THE TWO OPERATIONS
------------------
  BUILD (once, Sub-task 2.3.5)                         build_climate_registry()
      climate_config.yaml + EPW files -> parse -> compute -> records
      write_climate_metadata() then saves them to climate_metadata.csv

  LOAD (Sub-tasks 2.4, 2.6, 2.7)                        load_climate_registry()
      climate_metadata.csv -> validate -> the same records
      Touches no EPW contents. Checks each EPW FILE EXISTS, because the
      batch runner passes that path to EnergyPlus on the command line.

THE FEATURES
------------
  Plain means over 8760 hours:
      mean_dbt   = mean(T_h)                         degC
      mean_rh    = mean(RH_h)                        %
      mean_wind  = mean(v_h)                         m/s
  Plain sum:
      annual_ghi = sum(I_h) / 1000                   kWh/m2 per year
  Conditional sums over 365 daily mean temperatures T_d:
      hdd18      = sum_d max(0, 18 - T_d)            degC-day
      cdd10      = sum_d max(0, T_d - 10)            degC-day
  Not computed:
      latitude   <- EPW LOCATION header
      ashrae_zone<- climate_config.yaml (validated against the .stat file)

  The max(0, .) clipping is what makes degree-days informative: without it
  warm days would cancel cold ones and HDD would collapse into the annual
  mean temperature. A building that needs heating in January and cooling
  in July needs both; clipping counts them separately.

DEGREE-DAY CONVENTION - SETTLED EMPIRICALLY
-------------------------------------------
T_d is the MEAN OF THE 24 HOURLY VALUES of each day.

This was chosen by testing, not by preference. Three candidate
definitions were compared against the "(wthr file)" degree-days that
EnergyPlus writes into each station's .stat file:

    24-hour mean           11/11 stations exact (max diff 0.5 degC-day)
    (Tmax + Tmin) / 2       0/11 exact (max diff 213)
    hourly / 24             2/11 exact (max diff 171)

The 24-hour mean reproduces EnergyPlus's own figures exactly, so it is
used. validate_registry() re-runs that comparison on every build.

NOTE: each .stat file also contains "(standard)" degree-days. Those come
from ASHRAE long-term climate normals on an 18.3 degC base - a different
period of record - and will NOT match. Only the "(wthr file)" section is
the correct comparison.

ROUNDING AND TRAIN/SERVE PARITY
-------------------------------
compute_features() rounds each feature to a fixed precision
(FEATURE_DECIMALS) BEFORE returning. The CSV stores exactly those values.
So the numbers the surrogate is trained on (read from the CSV) and the
numbers a Phase 5 user's uploaded EPW produces (computed live) are
identical by construction, rather than differing in the 6th decimal.

FILE NAMING
-----------
Every EPW is named after its config key: colombo.epw, london.epw, ...
_load_config() enforces this. Earlier names carried the ASHRAE zone
(1A_Colombo.epw) and the role (HOLDOUT_4A_London.epw); when ASHRAE
169-2021 relabelled Colombo as 0A, the filename became false. A key never
changes, so a key-based name cannot drift. Zone and role live only in
the config, where they are validated.

NO SCALING HERE
---------------
Features are returned in natural units. Standardisation belongs to the
Phase 3 model pipeline, where the scaler is fitted on training data only
and saved with the model. Doing it here would bake an irreversible
transformation into the dataset.

DEPENDENCIES
------------
numpy and the standard library. PyYAML is imported only inside the BUILD
path, so the LOAD path - which the Phase 5 web backend uses - does not
need it. No eppy, no EnergyPlus.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

try:                                            # package import (repo root on path)
    from .epw_parser import EPWData, EPWParseError, parse_epw
except ImportError:                             # script / test execution
    from epw_parser import EPWData, EPWParseError, parse_epw  # type: ignore

__all__ = [
    "ClimateRegistryError",
    "ClimateFeatures", "ClimateDiagnostics", "ClimateRecord", "StatReference",
    "CONTINUOUS_FEATURES", "FEATURE_DECIMALS", "DEGREE_DAY_METHOD",
    "daily_mean_temperature", "degree_days",
    "compute_features", "compute_diagnostics",
    "build_climate_registry", "write_climate_metadata", "load_climate_registry",
    "training_keys", "holdout_keys",
    "read_stat_reference", "validate_registry",
]


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "climate_config.yaml"
DEFAULT_EPW_DIR = REPO_ROOT / "climate" / "epw_files"
DEFAULT_STAT_DIR = REPO_ROOT / "climate" / "epw_stat"
DEFAULT_METADATA_CSV = REPO_ROOT / "climate" / "climate_metadata.csv"

CSV_SCHEMA_VERSION = 1
HOURS_PER_DAY = 24

HDD_BASE_C = 18.0
CDD_BASE_C = 10.0
DEGREE_DAY_METHOD = "daily_mean_of_24_hourly_values"

# The seven continuous surrogate inputs, in the canonical column order.
# ashrae_zone is the eighth feature but is categorical; see Decision D2.
CONTINUOUS_FEATURES: tuple[str, ...] = (
    "hdd18", "cdd10", "annual_ghi", "mean_dbt", "mean_rh", "latitude", "mean_wind",
)

FEATURE_DECIMALS: dict[str, int] = {
    "hdd18": 1, "cdd10": 1, "annual_ghi": 1,
    "mean_dbt": 3, "mean_rh": 2, "latitude": 4, "mean_wind": 3,
}

DIAGNOSTIC_DECIMALS: dict[str, int] = {
    "longitude": 4, "elevation": 1,
    "dbt_min": 1, "dbt_max": 1, "dbt_std": 3,
    "hdd10": 1, "cdd18": 1, "ghi_summer": 1, "ghi_winter": 1,
}

VALID_ROLES = ("training", "holdout", "excluded")
REQUIRED_CONFIG_KEYS = ("city", "country", "role", "koppen", "ashrae_zone",
                        "wmo", "epw_filename")

# .stat tolerance: .stat values are integers, ours carry one decimal.
STAT_TOLERANCE = 1.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ClimateRegistryError(Exception):
    """Raised when config, EPW files or the metadata CSV are inconsistent.

    Every condition that raises here would otherwise pair the wrong
    weather with the wrong label somewhere downstream - a failure that
    produces no error of its own and a silently false dataset.
    """


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClimateFeatures:
    """The eight climate features that accompany every dataset row."""
    hdd18: float              # degC-day, base 18 degC
    cdd10: float              # degC-day, base 10 degC
    annual_ghi: float         # kWh/m2 per year
    mean_dbt: float           # degC
    mean_rh: float            # %
    latitude: float           # degrees north
    mean_wind: float          # m/s
    ashrae_zone: str          # ASHRAE 169-2021 label, categorical

    def continuous_vector(self) -> np.ndarray:
        """The seven continuous features in CONTINUOUS_FEATURES order."""
        return np.array([getattr(self, name) for name in CONTINUOUS_FEATURES], dtype=float)

    def as_dict(self) -> dict[str, float | str]:
        return {**{n: getattr(self, n) for n in CONTINUOUS_FEATURES},
                "ashrae_zone": self.ashrae_zone}


@dataclass(frozen=True)
class ClimateDiagnostics:
    """Quantities for QA and the methodology chapter. NOT model inputs.

    Kept in a separate class so that "which columns were model inputs" can
    never be muddled: model inputs are exactly ClimateFeatures.
    """
    longitude: float
    elevation: float          # m
    dbt_min: float            # degC, hourly minimum
    dbt_max: float            # degC, hourly maximum
    dbt_std: float            # degC, hourly standard deviation
    hdd10: float              # degC-day, base 10
    cdd18: float              # degC-day, base 18
    ghi_summer: float         # kWh/m2, warm half-year (hemisphere-aware)
    ghi_winter: float         # kWh/m2, cool half-year
    n_hours: int
    n_substituted: int        # values the parser had to fill; 0 for TMYx
    data_source: str          # e.g. "SRC-TMYx"
    period_of_record: str     # e.g. "2011-2025"


@dataclass(frozen=True)
class ClimateRecord:
    """Everything known about one climate. What the registry maps keys to."""
    key: str
    city: str
    country: str
    role: Literal["training", "holdout"]
    koppen: str
    wmo: str                  # string, always - "037760" must keep its zero
    epw_filename: str
    epw_path: Path            # resolved against the EPW directory
    features: ClimateFeatures
    diagnostics: ClimateDiagnostics

    @property
    def is_training(self) -> bool:
        return self.role == "training"

    @property
    def is_holdout(self) -> bool:
        return self.role == "holdout"


@dataclass(frozen=True)
class StatReference:
    """Independent reference values read from a station's .stat file."""
    hdd18: float              # "(wthr file)" heating degree-days, 18 degC base
    cdd10: float              # "(wthr file)" cooling degree-days, 10 degC base
    ashrae_zone: str          # "Climate Zone" line, ASHRAE 169-2021


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------

def daily_mean_temperature(dry_bulb: np.ndarray) -> np.ndarray:
    """Collapse hourly dry-bulb into one mean temperature per day.

    Mean of the 24 hourly values - the convention that reproduces
    EnergyPlus's own weather-file degree-days (see module docstring).
    """
    if len(dry_bulb) % HOURS_PER_DAY:
        raise ValueError(f"{len(dry_bulb)} hours is not a whole number of days")
    return dry_bulb.reshape(-1, HOURS_PER_DAY).mean(axis=1)


def degree_days(daily_temps: np.ndarray, base: float,
                mode: Literal["heating", "cooling"]) -> float:
    """Sum of daily departures beyond a base temperature, clipped at zero.

        heating:  sum_d max(0, base - T_d)
        cooling:  sum_d max(0, T_d - base)
    """
    if mode == "heating":
        departures = base - daily_temps
    elif mode == "cooling":
        departures = daily_temps - base
    else:
        raise ValueError(f"mode must be 'heating' or 'cooling', got {mode!r}")
    return float(np.clip(departures, 0.0, None).sum())


def _round(value: float, decimals: int) -> float:
    return float(round(value, decimals))


def compute_features(epw: EPWData, ashrae_zone: str) -> ClimateFeatures:
    """Compress one parsed EPW into the eight climate features.

    Values are rounded to FEATURE_DECIMALS here, so that live computation
    (Phase 5) and CSV-loaded values (training) are identical.
    """
    daily = daily_mean_temperature(epw.dry_bulb)
    raw = {
        "hdd18": degree_days(daily, HDD_BASE_C, "heating"),
        "cdd10": degree_days(daily, CDD_BASE_C, "cooling"),
        "annual_ghi": float(epw.glob_horiz_rad.sum()) / 1000.0,
        "mean_dbt": float(epw.dry_bulb.mean()),
        "mean_rh": float(epw.rel_humidity.mean()),
        "latitude": float(epw.location.latitude),
        "mean_wind": float(epw.wind_speed.mean()),
    }
    return ClimateFeatures(
        **{name: _round(raw[name], FEATURE_DECIMALS[name]) for name in CONTINUOUS_FEATURES},
        ashrae_zone=ashrae_zone,
    )


def compute_diagnostics(epw: EPWData) -> ClimateDiagnostics:
    """Secondary quantities for QA and reporting."""
    daily = daily_mean_temperature(epw.dry_bulb)

    # Warm half-year: Apr-Sep in the northern hemisphere, Oct-Mar in the south.
    northern = epw.location.latitude >= 0
    warm = (epw.month >= 4) & (epw.month <= 9)
    if not northern:
        warm = ~warm
    ghi_kwh = epw.glob_horiz_rad / 1000.0

    raw = {
        "longitude": float(epw.location.longitude),
        "elevation": float(epw.location.elevation),
        "dbt_min": float(epw.dry_bulb.min()),
        "dbt_max": float(epw.dry_bulb.max()),
        "dbt_std": float(epw.dry_bulb.std()),
        "hdd10": degree_days(daily, 10.0, "heating"),
        "cdd18": degree_days(daily, 18.0, "cooling"),
        "ghi_summer": float(ghi_kwh[warm].sum()),
        "ghi_winter": float(ghi_kwh[~warm].sum()),
    }
    return ClimateDiagnostics(
        **{name: _round(value, DIAGNOSTIC_DECIMALS[name]) for name, value in raw.items()},
        n_hours=epw.n_hours,
        n_substituted=int(sum(epw.missing_counts.values())
                          + sum(epw.out_of_range_counts.values())),
        data_source=epw.location.data_source,
        period_of_record=epw.period_of_record,
    )


# ---------------------------------------------------------------------------
# BUILD: config + EPW files -> records
# ---------------------------------------------------------------------------

def _load_config(config_path: Path) -> dict[str, dict]:
    """Read climate_config.yaml, returning non-excluded climates."""
    import yaml                                 # BUILD path only - see docstring

    if not config_path.exists():
        raise ClimateRegistryError(f"config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text())
    climates = config.get("climates") or {}
    if not climates:
        raise ClimateRegistryError(f"{config_path.name}: no 'climates' section")

    selected = {}
    for key, entry in climates.items():
        missing = [k for k in REQUIRED_CONFIG_KEYS if k not in entry]
        if missing:
            raise ClimateRegistryError(f"config entry {key!r} missing {missing}")
        expected_name = f"{key}.epw"
        if entry["epw_filename"] != expected_name:
            raise ClimateRegistryError(
                f"config entry {key!r}: epw_filename is {entry['epw_filename']!r}, "
                f"expected {expected_name!r}. EPW files are named after their config "
                f"key so that no fact that can change (zone, role) lives in a filename."
            )
        if entry["role"] not in VALID_ROLES:
            raise ClimateRegistryError(f"config entry {key!r}: invalid role {entry['role']!r}")
        if entry["role"] != "excluded":
            selected[key] = entry
    return selected


def build_climate_registry(config_path: str | Path = DEFAULT_CONFIG,
                           epw_dir: str | Path = DEFAULT_EPW_DIR,
                           ) -> dict[str, ClimateRecord]:
    """Parse every configured EPW and compute its features. The slow path.

    Also checks that each file's own WMO number matches the config entry
    that points at it - so a file copied under the wrong name (Denver's
    weather saved as chicago.epw) is caught here, rather than producing
    a dataset in which Chicago's buildings are simulated in Denver.
    """
    config_path, epw_dir = Path(config_path), Path(epw_dir)
    entries = _load_config(config_path)

    registry: dict[str, ClimateRecord] = {}
    for key, entry in entries.items():
        path = epw_dir / entry["epw_filename"]
        try:
            epw = parse_epw(path)
        except EPWParseError as exc:
            raise ClimateRegistryError(f"climate {key!r}: {exc}") from exc

        config_wmo = str(entry["wmo"]).strip()
        if epw.location.wmo.lstrip("0") != config_wmo.lstrip("0"):
            raise ClimateRegistryError(
                f"climate {key!r}: config says WMO {config_wmo} but "
                f"{path.name} contains WMO {epw.location.wmo} "
                f"({epw.location.city}). The file does not belong to this entry."
            )

        registry[key] = ClimateRecord(
            key=key,
            city=str(entry["city"]),
            country=str(entry["country"]),
            role=entry["role"],
            koppen=str(entry["koppen"]),
            wmo=config_wmo,
            epw_filename=entry["epw_filename"],
            epw_path=path,
            features=compute_features(epw, str(entry["ashrae_zone"])),
            diagnostics=compute_diagnostics(epw),
        )

    if not any(r.is_training for r in registry.values()):
        raise ClimateRegistryError("no training climates in config")
    return registry


# ---------------------------------------------------------------------------
# CSV: the stored artefact
# ---------------------------------------------------------------------------

IDENTITY_COLUMNS = ("key", "city", "country", "role", "koppen", "wmo", "epw_filename")
FEATURE_COLUMNS = (*CONTINUOUS_FEATURES, "ashrae_zone")
DIAGNOSTIC_FLOAT_COLUMNS = tuple(DIAGNOSTIC_DECIMALS)
DIAGNOSTIC_OTHER_COLUMNS = ("n_hours", "n_substituted", "data_source", "period_of_record")
PROVENANCE_COLUMNS = ("degree_day_method", "csv_schema_version")

CSV_COLUMNS = (*IDENTITY_COLUMNS, *FEATURE_COLUMNS, *DIAGNOSTIC_FLOAT_COLUMNS,
               *DIAGNOSTIC_OTHER_COLUMNS, *PROVENANCE_COLUMNS)


def _fmt(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def write_climate_metadata(registry: dict[str, ClimateRecord],
                           csv_path: str | Path = DEFAULT_METADATA_CSV) -> Path:
    """Write the registry to CSV. Deterministic: same input, same bytes.

    Rows are sorted by key and every float has a fixed number of decimals,
    so regenerating the file produces a byte-identical result and any
    real change shows up in a diff.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        for key in sorted(registry):
            r, f, d = registry[key], registry[key].features, registry[key].diagnostics
            writer.writerow([
                r.key, r.city, r.country, r.role, r.koppen, r.wmo, r.epw_filename,
                *(_fmt(getattr(f, n), FEATURE_DECIMALS[n]) for n in CONTINUOUS_FEATURES),
                f.ashrae_zone,
                *(_fmt(getattr(d, n), DIAGNOSTIC_DECIMALS[n]) for n in DIAGNOSTIC_FLOAT_COLUMNS),
                d.n_hours, d.n_substituted, d.data_source, d.period_of_record,
                DEGREE_DAY_METHOD, CSV_SCHEMA_VERSION,
            ])
    return csv_path


# ---------------------------------------------------------------------------
# LOAD: CSV -> records. The fast path used by 2.4, 2.6, 2.7 and Phase 5.
# ---------------------------------------------------------------------------

def _to_float(row: dict, column: str, key: str) -> float:
    text = row[column].strip()
    try:
        value = float(text)
    except ValueError as exc:
        raise ClimateRegistryError(f"{key}: column {column!r} is not a number: {text!r}") from exc
    if not math.isfinite(value):
        raise ClimateRegistryError(f"{key}: column {column!r} is {text!r}")
    return value


def load_climate_registry(csv_path: str | Path = DEFAULT_METADATA_CSV,
                          epw_dir: str | Path = DEFAULT_EPW_DIR,
                          check_files: bool = True,
                          ) -> dict[str, ClimateRecord]:
    """Read climate_metadata.csv into records, validating as it goes.

    Uses the stdlib csv reader, which returns every value as a string, and
    converts each column explicitly. That is deliberate: pandas would infer
    London's WMO "037760" as the integer 37760 and silently drop the zero.

    check_files: verify every EPW exists. Keep True for the batch runner,
    which hands these paths to EnergyPlus. It may be False for consumers
    that only need the numbers (e.g. Phase 5 serving without the EPWs).
    """
    csv_path, epw_dir = Path(csv_path), Path(epw_dir)
    if not csv_path.exists():
        raise ClimateRegistryError(
            f"{csv_path} not found - run: python climate/climate_features.py --build"
        )

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if header != CSV_COLUMNS:
            missing = [c for c in CSV_COLUMNS if c not in header]
            extra = [c for c in header if c not in CSV_COLUMNS]
            raise ClimateRegistryError(
                f"{csv_path.name}: columns do not match this module "
                f"(missing {missing}, unexpected {extra}). Rebuild with --build."
            )
        rows = list(reader)

    registry: dict[str, ClimateRecord] = {}
    for row in rows:
        key = row["key"].strip()
        if key in registry:
            raise ClimateRegistryError(f"{csv_path.name}: duplicate key {key!r}")
        if row["role"] not in ("training", "holdout"):
            raise ClimateRegistryError(f"{key}: invalid role {row['role']!r}")
        if row["degree_day_method"] != DEGREE_DAY_METHOD:
            raise ClimateRegistryError(
                f"{key}: CSV built with degree-day method {row['degree_day_method']!r}, "
                f"this module uses {DEGREE_DAY_METHOD!r}. Rebuild with --build."
            )

        epw_path = epw_dir / row["epw_filename"]
        if check_files and not epw_path.exists():
            raise ClimateRegistryError(
                f"{key}: EPW file missing: {epw_path}. EnergyPlus needs this file "
                f"for every {key} simulation."
            )

        registry[key] = ClimateRecord(
            key=key,
            city=row["city"],
            country=row["country"],
            role=row["role"],                   # type: ignore[arg-type]
            koppen=row["koppen"],
            wmo=row["wmo"],
            epw_filename=row["epw_filename"],
            epw_path=epw_path,
            features=ClimateFeatures(
                **{n: _to_float(row, n, key) for n in CONTINUOUS_FEATURES},
                ashrae_zone=row["ashrae_zone"],
            ),
            diagnostics=ClimateDiagnostics(
                **{n: _to_float(row, n, key) for n in DIAGNOSTIC_FLOAT_COLUMNS},
                n_hours=int(row["n_hours"]),
                n_substituted=int(row["n_substituted"]),
                data_source=row["data_source"],
                period_of_record=row["period_of_record"],
            ),
        )

    if not any(r.is_training for r in registry.values()):
        raise ClimateRegistryError(f"{csv_path.name}: no training climates")
    return registry


def training_keys(registry: dict[str, ClimateRecord]) -> list[str]:
    """Keys of training climates, sorted. What the sampler in 2.4 iterates."""
    return sorted(k for k, r in registry.items() if r.is_training)


def holdout_keys(registry: dict[str, ClimateRecord]) -> list[str]:
    """Keys of held-out climates, sorted. Must never reach training."""
    return sorted(k for k, r in registry.items() if r.is_holdout)


# ---------------------------------------------------------------------------
# Validation against independent reference data
# ---------------------------------------------------------------------------

_STAT_HDD = re.compile(r"-\s*(\d+)\s+annual \(wthr file\) heating degree-days \(18C")
_STAT_CDD = re.compile(r"-\s*(\d+)\s+annual \(wthr file\) cooling degree-days \(10C")
_STAT_ZONE = re.compile(r'Climate Zone "([^"]+)"')


def read_stat_reference(stat_path: str | Path) -> StatReference | None:
    """Read the "(wthr file)" degree-days and ASHRAE zone from a .stat file.

    Returns None if the file is absent or any value cannot be found.
    Deliberately ignores the "(standard)" section - see module docstring.
    """
    stat_path = Path(stat_path)
    if not stat_path.exists():
        return None
    text = stat_path.read_text(encoding="latin-1", errors="replace")
    hdd, cdd, zone = _STAT_HDD.search(text), _STAT_CDD.search(text), _STAT_ZONE.search(text)
    if not (hdd and cdd and zone):
        return None
    return StatReference(hdd18=float(hdd.group(1)), cdd10=float(cdd.group(1)),
                         ashrae_zone=zone.group(1))


def validate_registry(registry: dict[str, ClimateRecord],
                      stat_dir: str | Path = DEFAULT_STAT_DIR,
                      ) -> list[tuple[str, bool, str]]:
    """Compare every climate against its .stat file.

    Returns (label, passed, detail) tuples. Checks per climate:
      - HDD18 within STAT_TOLERANCE of EnergyPlus's weather-file value
      - CDD10 within STAT_TOLERANCE of EnergyPlus's weather-file value
      - config ashrae_zone equals the zone stated in the .stat file
    """
    stat_dir = Path(stat_dir)
    results: list[tuple[str, bool, str]] = []
    for key in sorted(registry):
        record = registry[key]
        ref = read_stat_reference(stat_dir / (Path(record.epw_filename).stem + ".stat"))
        if ref is None:
            results.append((f"{key}: .stat reference", False, "missing or unreadable"))
            continue

        f = record.features
        for name, ours, theirs in (("HDD18", f.hdd18, ref.hdd18), ("CDD10", f.cdd10, ref.cdd10)):
            diff = ours - theirs
            results.append((f"{key}: {name}", abs(diff) <= STAT_TOLERANCE,
                            f"computed {ours:.1f}, EnergyPlus {theirs:.0f}, diff {diff:+.1f}"))

        results.append((f"{key}: ASHRAE zone", f.ashrae_zone == ref.ashrae_zone,
                        f"config {f.ashrae_zone}, .stat {ref.ashrae_zone}"))
    return results


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _print_table(registry: dict[str, ClimateRecord]) -> None:
    head = (f"{'key':15}{'role':9}{'zone':6}{'HDD18':>8}{'CDD10':>8}{'GHI':>8}"
            f"{'DBT':>8}{'RH':>7}{'lat':>9}{'wind':>7}")
    print(head)
    print("-" * len(head))
    for key in sorted(registry, key=lambda k: (registry[k].role != "training", k)):
        r, f = registry[key], registry[key].features
        print(f"{key:15}{r.role:9}{f.ashrae_zone:6}{f.hdd18:8.0f}{f.cdd10:8.0f}"
              f"{f.annual_ghi:8.0f}{f.mean_dbt:8.2f}{f.mean_rh:7.1f}"
              f"{f.latitude:9.3f}{f.mean_wind:7.2f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Climate features and registry (Sub-task 2.3).")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true",
                      help="parse EPWs, compute features, write climate_metadata.csv, validate")
    mode.add_argument("--validate", action="store_true",
                      help="check the CSV against each station's .stat file")
    mode.add_argument("--show", action="store_true", help="print the registry from the CSV")
    args = parser.parse_args(argv)

    try:
        if args.build:
            registry = build_climate_registry()
            out = write_climate_metadata(registry)
            print(f"\nBuilt {len(registry)} climates -> {out.relative_to(REPO_ROOT)}")
            print(f"Degree-day method: {DEGREE_DAY_METHOD}\n")
            _print_table(registry)
            registry = load_climate_registry()          # prove the file round-trips
        else:
            registry = load_climate_registry()

        if args.show:
            _print_table(registry)
            print(f"\ntraining: {', '.join(training_keys(registry))}")
            print(f"holdout:  {', '.join(holdout_keys(registry))}")
            return 0

        results = validate_registry(registry)
        failures = [r for r in results if not r[1]]
        print(f"\nValidation against EnergyPlus .stat files")
        for label, passed, detail in results:
            if not passed or args.validate:
                print(f"  {'PASS' if passed else 'FAIL'}  {label:32} {detail}")
        print(f"\n{len(results) - len(failures)}/{len(results)} checks passed")
        return 1 if failures else 0

    except ClimateRegistryError as exc:
        print(f"\nERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
