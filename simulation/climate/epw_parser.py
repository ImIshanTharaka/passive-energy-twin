"""
epw_parser.py - EnergyPlus Weather (EPW) file reader.

This module does ONE thing: turn an EPW file into clean, labelled,
validated numeric arrays. It performs NO building-science computation.
Degree-days, annual irradiation and every other derived quantity belong
to climate_features.py.

WHAT THIS MODULE PRODUCES
-------------------------
  EPWLocation    - station identity parsed from the LOCATION header
  EPWData        - hourly arrays + location + data-quality record
  parse_epw()    - the single public entry point

WHY IT IS SEPARATE FROM climate_features.py
-------------------------------------------
Two different kinds of knowledge:
  - this module knows the FILE FORMAT   (which column is which,
    which numbers mean "missing", how many rows a year has)
  - climate_features.py knows the SCIENCE (what a degree-day is,
    which base temperature, how to aggregate irradiation)

Keeping them apart means the degree-day convention is settled in exactly
one place, and an EPW format change is handled in exactly the other.
It also mirrors idf_generator/, where only generator.py touches eppy.

THE COLUMN-INDEX PROBLEM
------------------------
An EPW data row is 35 comma-separated numbers with no column names
anywhere in the file. Field positions are fixed by the EnergyPlus
Auxiliary Programs specification and must simply be known. A wrong
index produces a dataset that is quietly, confidently incorrect -
no error, no warning, just wrong physics for 3,400 simulations.

Every field position therefore lives in ONE table (FIELD_SPECS) rather
than scattered through the code, and test_epw_parser.py verifies each
index individually against a synthetic file in which every field holds
a distinct known value.

MISSING DATA
------------
EPW does not leave missing values blank; it writes sentinel numbers that
look like real data (99.9 degC, 999 %, 9999 Wh/m2). Undetected, a few
missing hours quietly bias annual means with nothing appearing broken.

Policy, in order:
  1. Sentinel values      -> NaN  (counted as 'missing')
  2. Physically impossible values -> NaN  (counted as 'out_of_range')
     A dry-bulb of -80 degC is not a sentinel but is not data either.
  3. NaN runs <= MAX_GAP_HOURS -> filled by circular linear interpolation
  4. NaN runs >  MAX_GAP_HOURS -> EPWParseError

Step 4 is deliberate. A long gap is a corrupt or inappropriate file, and
interpolating across it would invent weather. Better to stop than to
produce a plausible-looking dataset built on invented data.

Interpolation is CIRCULAR: hour 8760 is adjacent to hour 1, so a gap
spanning New Year is filled from both sides rather than left dangling.

WHAT IS DELIBERATELY NOT PARSED
-------------------------------
Wind direction (field index 20) is excluded. It is an angular quantity,
so linear interpolation across the 0/360 boundary is wrong (the mean of
350 deg and 10 deg is 0 deg, not 180 deg). It is not among the eight
climate features, so parsing it would add a correctness trap for no
benefit. If a later phase needs it, handle it with circular statistics.

The DESIGN CONDITIONS, TYPICAL/EXTREME PERIODS and GROUND TEMPERATURES
header lines are not parsed either. Published degree-days used for
cross-validation come from the companion .stat file, which is easier to
parse and is what EnergyPlus itself reports.

STRUCTURAL ASSUMPTIONS (all asserted, not trusted)
--------------------------------------------------
  - 8 header lines, the first beginning with LOCATION
  - exactly 8760 data rows: TMY/TMYx files never contain 29 February,
    so an 8784-row file is an AMY or a leap year and is rejected rather
    than silently reshaped
  - a single full-year DATA PERIODS declaration

THE YEAR COLUMN IS MEANINGLESS
------------------------------
A TMY file is assembled from different real years - January from one,
February from another - so the year column jumps around and carries no
information. It is not exposed. Month/day/hour are exposed for
diagnostics and for the daily reshaping climate_features.py performs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "EPWParseError",
    "EPWLocation",
    "EPWData",
    "FieldSpec",
    "FIELD_SPECS",
    "parse_epw",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_HEADER_LINES = 8
EXPECTED_ROWS = 8760          # 365 x 24; TMY files exclude 29 February
HOURS_PER_DAY = 24
MAX_GAP_HOURS = 6             # longest NaN run that may be interpolated
MIN_EPW_FIELDS = 22           # highest index used + 1 (wind speed at 21)
ENCODING = "latin-1"          # EPW comment lines are not reliably UTF-8


# ---------------------------------------------------------------------------
# Field specification - the single source of truth for column positions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldSpec:
    """One numeric column of an EPW data row.

    missing_at: values >= this are the EPW 'missing' sentinel. All EPW
        sentinels are large positive numbers (99.9, 999, 9999), so a
        single >= comparison covers every field uniformly.
    valid_min / valid_max: physically plausible bounds. Values outside
        them are treated as missing even when they are not sentinels.
    """
    name: str
    index: int
    units: str
    missing_at: float
    valid_min: float
    valid_max: float


# Positions per the EnergyPlus Auxiliary Programs EPW specification.
# Row layout (0-based): 0 year, 1 month, 2 day, 3 hour, 4 minute,
# 5 data-source flags, then the numeric fields below.
FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec("dry_bulb",        6, "degC",   99.9,   -70.0,   70.0),
    FieldSpec("dew_point",       7, "degC",   99.9,   -70.0,   70.0),
    FieldSpec("rel_humidity",    8, "%",     999.0,     0.0,  110.0),
    FieldSpec("atm_pressure",    9, "Pa",  999999.0, 31000.0, 120000.0),
    FieldSpec("glob_horiz_rad", 13, "Wh/m2", 9999.0,    0.0, 1600.0),
    FieldSpec("dir_normal_rad", 14, "Wh/m2", 9999.0,    0.0, 1600.0),
    FieldSpec("diff_horiz_rad", 15, "Wh/m2", 9999.0,    0.0, 1600.0),
    FieldSpec("wind_speed",     21, "m/s",    999.0,    0.0,   40.0),
)

DATE_INDICES = {"month": 1, "day": 2, "hour": 3}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class EPWParseError(Exception):
    """Raised when an EPW file is malformed or its data is unusable.

    Deliberately not a warning. Every condition that raises here would
    otherwise produce a silently incorrect dataset.
    """


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EPWLocation:
    """Station identity, parsed from the LOCATION header line.

    This - not climate_config.yaml - is the authoritative source for
    coordinates. The config deliberately stores no coordinate fields,
    because duplicated station details drift away from the files they
    describe.
    """
    city: str
    state: str
    country: str
    data_source: str          # e.g. "TMY3", "SRC-TMYx"
    wmo: str
    latitude: float           # degrees, north positive
    longitude: float          # degrees, east positive
    tz_offset: float          # hours from UTC
    elevation: float          # metres above sea level

    def __str__(self) -> str:
        return (f"{self.city}, {self.country} ({self.data_source} {self.wmo}) "
                f"{self.latitude:.3f}, {self.longitude:.3f}, {self.elevation:.0f} m")


@dataclass(frozen=True)
class EPWData:
    """One parsed EPW file: 8760 hourly values per field, plus provenance.

    Array fields are float64 of length EXPECTED_ROWS, fully populated
    (any NaN remaining after gap-filling would have raised).
    """
    path: Path
    location: EPWLocation

    dry_bulb: np.ndarray
    dew_point: np.ndarray
    rel_humidity: np.ndarray
    atm_pressure: np.ndarray
    glob_horiz_rad: np.ndarray
    dir_normal_rad: np.ndarray
    diff_horiz_rad: np.ndarray
    wind_speed: np.ndarray

    month: np.ndarray
    day: np.ndarray
    hour: np.ndarray

    period_of_record: str = ""
    missing_counts: dict[str, int] = field(default_factory=dict)
    out_of_range_counts: dict[str, int] = field(default_factory=dict)
    interpolated_counts: dict[str, int] = field(default_factory=dict)

    @property
    def n_hours(self) -> int:
        return len(self.dry_bulb)

    @property
    def n_days(self) -> int:
        return self.n_hours // HOURS_PER_DAY

    @property
    def is_pristine(self) -> bool:
        """True when no value needed substituting - expected for TMY/TMYx."""
        return not (sum(self.missing_counts.values())
                    or sum(self.out_of_range_counts.values()))

    def field_array(self, name: str) -> np.ndarray:
        """Look up an hourly array by its FIELD_SPECS name."""
        try:
            return getattr(self, name)
        except AttributeError as exc:
            raise KeyError(f"unknown field {name!r}") from exc

    def quality_summary(self) -> str:
        if self.is_pristine:
            return "pristine (no missing or out-of-range values)"
        parts = []
        for spec in FIELD_SPECS:
            miss = self.missing_counts.get(spec.name, 0)
            oor = self.out_of_range_counts.get(spec.name, 0)
            if miss or oor:
                parts.append(f"{spec.name}: {miss} missing, {oor} out-of-range")
        return "; ".join(parts)


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

def _parse_location(line: str, path: Path) -> EPWLocation:
    """Parse the LOCATION header.

    Layout: LOCATION,city,state,country,source,WMO,lat,lon,tz,elevation
    """
    parts = [p.strip() for p in line.split(",")]
    if not parts or parts[0].upper() != "LOCATION":
        raise EPWParseError(f"{path.name}: first line is not a LOCATION header")
    if len(parts) < 10:
        raise EPWParseError(
            f"{path.name}: LOCATION header has {len(parts)} fields, expected at least 10"
        )
    try:
        return EPWLocation(
            city=parts[1], state=parts[2], country=parts[3],
            data_source=parts[4], wmo=parts[5],
            latitude=float(parts[6]), longitude=float(parts[7]),
            tz_offset=float(parts[8]), elevation=float(parts[9]),
        )
    except ValueError as exc:
        raise EPWParseError(f"{path.name}: non-numeric value in LOCATION header: {exc}") from exc


def _parse_period_of_record(header: list[str]) -> str:
    """Extract 'Period of Record=YYYY-YYYY' from COMMENTS 1, when present.

    TMYx files record it; TMY3 files generally do not. Absence is normal
    and is not an error - the field is provenance metadata, not data.
    """
    for line in header:
        match = re.search(r"Period of Record\s*=\s*([0-9]{4}\s*-\s*[0-9]{4})", line, re.I)
        if match:
            return match.group(1).replace(" ", "")
    return ""


def _validate_data_periods(line: str, path: Path) -> None:
    """Check DATA PERIODS declares exactly one period.

    Layout: DATA PERIODS,n_periods,records_per_hour,name,start_day,start,end
    A file with multiple periods or sub-hourly records would break the
    8760-row and 24-hour-per-day assumptions everywhere downstream.
    """
    parts = [p.strip() for p in line.split(",")]
    if not parts or not parts[0].upper().startswith("DATA PERIODS"):
        raise EPWParseError(f"{path.name}: line 8 is not a DATA PERIODS header")
    try:
        n_periods, records_per_hour = int(parts[1]), int(parts[2])
    except (IndexError, ValueError) as exc:
        raise EPWParseError(f"{path.name}: malformed DATA PERIODS header") from exc
    if n_periods != 1:
        raise EPWParseError(
            f"{path.name}: {n_periods} data periods declared, expected 1 full year"
        )
    if records_per_hour != 1:
        raise EPWParseError(
            f"{path.name}: {records_per_hour} records per hour, expected hourly data"
        )


# ---------------------------------------------------------------------------
# Gap handling
# ---------------------------------------------------------------------------

def _nan_run_lengths(is_nan: np.ndarray) -> list[int]:
    """Lengths of consecutive NaN runs, treating the year as circular.

    A gap spanning 31 December into 1 January is one run, not two.
    """
    n = len(is_nan)
    if not is_nan.any():
        return []
    if is_nan.all():
        return [n]

    runs, i = [], 0
    while i < n:
        if is_nan[i]:
            j = i
            while j < n and is_nan[j]:
                j += 1
            runs.append([i, j - 1, j - i])          # start, end, length
            i = j
        else:
            i += 1

    # Merge a run ending at the last hour with one starting at the first.
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == n - 1:
        merged = runs[-1][2] + runs[0][2]
        runs = runs[1:-1] + [[runs[-1][0], runs[0][1], merged]]

    return [r[2] for r in runs]


def _fill_short_gaps(values: np.ndarray, name: str, path: Path) -> tuple[np.ndarray, int]:
    """Interpolate NaN runs of at most MAX_GAP_HOURS; raise on longer ones.

    Interpolation is circular. The valid sample points are tiled across
    [-n, 0), [0, n) and [n, 2n) before interpolating, so hours near
    either end of the year are filled from both sides rather than being
    held flat at the first or last valid value.
    """
    is_nan = np.isnan(values)
    n_missing = int(is_nan.sum())
    if n_missing == 0:
        return values, 0

    if is_nan.all():
        raise EPWParseError(f"{path.name}: field {name!r} is entirely missing")

    runs = _nan_run_lengths(is_nan)
    longest = max(runs)
    if longest > MAX_GAP_HOURS:
        raise EPWParseError(
            f"{path.name}: field {name!r} has a {longest}-hour gap "
            f"(limit {MAX_GAP_HOURS}). Interpolating this would invent weather; "
            f"replace the file instead. Total missing: {n_missing} hours "
            f"across {len(runs)} gap(s)."
        )

    n = len(values)
    idx = np.arange(n, dtype=float)
    valid = ~is_nan
    xp = np.concatenate([idx[valid] - n, idx[valid], idx[valid] + n])
    fp = np.tile(values[valid], 3)
    return np.interp(idx, xp, fp), n_missing


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_epw(path: str | Path) -> EPWData:
    """Read an EPW file into validated hourly arrays.

    Raises EPWParseError on any structural problem or unfillable gap.
    Every failure mode here would otherwise yield a silently wrong
    dataset, so none of them are warnings.
    """
    path = Path(path)
    if not path.exists():
        raise EPWParseError(f"file not found: {path}")

    lines = [ln for ln in path.read_text(encoding=ENCODING).splitlines() if ln.strip()]
    if len(lines) <= N_HEADER_LINES:
        raise EPWParseError(f"{path.name}: {len(lines)} lines, expected header plus data")

    header, body = lines[:N_HEADER_LINES], lines[N_HEADER_LINES:]
    location = _parse_location(header[0], path)
    _validate_data_periods(header[7], path)
    period_of_record = _parse_period_of_record(header)

    if len(body) != EXPECTED_ROWS:
        hint = ""
        if len(body) == 8784:
            hint = (" This is a leap year (8784 hours). TMY/TMYx files exclude "
                    "29 February; an actual-year file cannot be used here.")
        raise EPWParseError(
            f"{path.name}: {len(body)} data rows, expected {EXPECTED_ROWS}.{hint}"
        )

    rows = [ln.split(",") for ln in body]
    short = next((i for i, r in enumerate(rows) if len(r) < MIN_EPW_FIELDS), None)
    if short is not None:
        raise EPWParseError(
            f"{path.name}: data row {short + 1} has {len(rows[short])} fields, "
            f"expected at least {MIN_EPW_FIELDS}"
        )

    # Date columns. The year column is deliberately ignored - see docstring.
    dates: dict[str, np.ndarray] = {}
    for name, index in DATE_INDICES.items():
        try:
            dates[name] = np.array([int(float(r[index])) for r in rows], dtype=int)
        except ValueError as exc:
            raise EPWParseError(f"{path.name}: non-numeric {name} column: {exc}") from exc

    # Numeric fields, one FieldSpec at a time.
    arrays: dict[str, np.ndarray] = {}
    missing_counts: dict[str, int] = {}
    out_of_range_counts: dict[str, int] = {}
    interpolated_counts: dict[str, int] = {}

    for spec in FIELD_SPECS:
        try:
            raw = np.array([float(r[spec.index]) for r in rows], dtype=float)
        except ValueError as exc:
            raise EPWParseError(
                f"{path.name}: non-numeric value in field {spec.name!r} "
                f"(column {spec.index}): {exc}"
            ) from exc

        sentinel = raw >= spec.missing_at
        missing_counts[spec.name] = int(sentinel.sum())
        cleaned = np.where(sentinel, np.nan, raw)

        # Out-of-range values are not sentinels but are not data either.
        with np.errstate(invalid="ignore"):
            bad = (~np.isnan(cleaned)) & ((cleaned < spec.valid_min) | (cleaned > spec.valid_max))
        out_of_range_counts[spec.name] = int(bad.sum())
        cleaned = np.where(bad, np.nan, cleaned)

        filled, n_filled = _fill_short_gaps(cleaned, spec.name, path)
        arrays[spec.name] = filled
        interpolated_counts[spec.name] = n_filled

    return EPWData(
        path=path,
        location=location,
        month=dates["month"], day=dates["day"], hour=dates["hour"],
        period_of_record=period_of_record,
        missing_counts=missing_counts,
        out_of_range_counts=out_of_range_counts,
        interpolated_counts=interpolated_counts,
        **arrays,
    )
