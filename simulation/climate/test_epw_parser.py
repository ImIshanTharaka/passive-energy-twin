"""
test_epw_parser.py - Structural and data-quality checks for the EPW reader.

Tests verify:
  1. A well-formed synthetic file parses, with correct header values
  2. EVERY field reads its own column (the highest-value test here)
  3. Malformed files are rejected rather than silently accepted
  4. Missing-value sentinels are detected and short gaps interpolated
  5. Long gaps raise instead of inventing weather
  6. Interpolation is circular across the year boundary
  7. The 11 real project EPW files parse and are pristine

Test 2 deserves emphasis. A wrong column index is the one defect in this
module that produces no error at all - just physically wrong weather fed
to 3,400 simulations. The synthetic file gives every field a distinct
constant, so a transposed index fails loudly here instead of quietly in
the dataset.

Run:  python climate/test_epw_parser.py
"""

from __future__ import annotations

import datetime as dt
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import epw_parser as P  # type: ignore


# ---------------------------------------------------------------------------
# Synthetic EPW construction
# ---------------------------------------------------------------------------

HEADER_FILLER = [
    "DESIGN CONDITIONS,0",
    "TYPICAL/EXTREME PERIODS,0",
    "GROUND TEMPERATURES,0",
    "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
    'COMMENTS 1,"Synthetic test file. Period of Record=2011-2025"',
    "COMMENTS 2,test",
]

LOCATION_LINE = "LOCATION,Testville,TS,TST,SRC-TMYx,123456,7.25,-80.50,5.5,42.0"
DATA_PERIODS_LINE = "DATA PERIODS,1,1,Data,Sunday, 1/ 1,12/31"

# In-range defaults, one per FIELD_SPECS entry.
DEFAULTS = {6: "20.0", 7: "10.0", 8: "50", 9: "101325",
            13: "100", 14: "200", 15: "50", 21: "3.0"}


def make_row(month: int, day: int, hour: int,
             overrides: dict[int, object] | None = None) -> str:
    fields = ["0"] * 35
    fields[0], fields[1], fields[2], fields[3], fields[4] = "1999", str(month), str(day), str(hour), "0"
    fields[5] = "?9?9?9?9E0"
    for idx, value in DEFAULTS.items():
        fields[idx] = value
    for idx, value in (overrides or {}).items():
        fields[idx] = str(value)
    return ",".join(fields)


def make_epw(n_days: int = 365,
             location: str = LOCATION_LINE,
             data_periods: str = DATA_PERIODS_LINE,
             per_hour: dict[int, dict[int, object]] | None = None,
             drop_header: bool = False,
             truncate_row: int | None = None) -> str:
    """Build a complete synthetic EPW.

    per_hour maps absolute hour index (0-based) -> {field index: value}.
    """
    header = [] if drop_header else [location, *HEADER_FILLER, data_periods]
    rows, h = [], 0
    start = dt.date(1999, 1, 1)
    for d in range(n_days):
        date = start + dt.timedelta(days=d)
        for hour in range(1, 25):
            row = make_row(date.month, date.day, hour, (per_hour or {}).get(h))
            if truncate_row == h:
                row = ",".join(row.split(",")[:10])
            rows.append(row)
            h += 1
    return "\n".join([*header, *rows]) + "\n"


def write_epw(tmp: Path, name: str, content: str) -> Path:
    path = tmp / name
    path.write_text(content, encoding=P.ENCODING)
    return path


def parse_text(content: str, name: str = "synthetic.epw") -> P.EPWData:
    with tempfile.TemporaryDirectory() as td:
        return P.parse_epw(write_epw(Path(td), name, content))


def expect_error(content: str, fragment: str, name: str = "synthetic.epw") -> None:
    try:
        parse_text(content, name)
    except P.EPWParseError as exc:
        assert fragment.lower() in str(exc).lower(), \
            f"expected {fragment!r} in error, got: {exc}"
        return
    raise AssertionError(f"expected EPWParseError mentioning {fragment!r}, none raised")


# ---------------------------------------------------------------------------
# 1. Well-formed file
# ---------------------------------------------------------------------------

def check_valid_file_parses():
    data = parse_text(make_epw())
    assert isinstance(data, P.EPWData)
    assert data.n_hours == P.EXPECTED_ROWS, data.n_hours
    assert data.n_days == 365, data.n_days


def check_location_header_parsed():
    loc = parse_text(make_epw()).location
    assert loc.city == "Testville", loc.city
    assert loc.country == "TST", loc.country
    assert loc.data_source == "SRC-TMYx", loc.data_source
    assert loc.wmo == "123456", loc.wmo
    assert loc.latitude == 7.25, loc.latitude
    assert loc.longitude == -80.50, loc.longitude
    assert loc.tz_offset == 5.5, loc.tz_offset
    assert loc.elevation == 42.0, loc.elevation


def check_period_of_record_extracted():
    assert parse_text(make_epw()).period_of_record == "2011-2025"


def check_period_of_record_absent_is_not_an_error():
    content = make_epw().replace("Period of Record=2011-2025", "no record given")
    assert parse_text(content).period_of_record == ""


def check_date_columns_parsed():
    data = parse_text(make_epw())
    assert data.month[0] == 1 and data.day[0] == 1 and data.hour[0] == 1
    assert data.month[-1] == 12 and data.day[-1] == 31 and data.hour[-1] == 24
    assert set(np.unique(data.hour)) == set(range(1, 25))


def check_pristine_flag_true_for_clean_file():
    data = parse_text(make_epw())
    assert data.is_pristine, data.quality_summary()
    assert sum(data.interpolated_counts.values()) == 0


# ---------------------------------------------------------------------------
# 2. Column mapping - every field must read its own column
# ---------------------------------------------------------------------------

def check_each_field_reads_its_own_column():
    """Give every field a distinct constant and confirm it lands correctly.

    This is what catches a transposed index, which produces no error of
    its own anywhere else in the pipeline.
    """
    distinct = {6: 21.5, 7: 11.25, 8: 55.0, 9: 99000.0,
                13: 111.0, 14: 222.0, 15: 333.0, 21: 7.5}
    rows = {h: dict(distinct) for h in range(P.EXPECTED_ROWS)}
    data = parse_text(make_epw(per_hour=rows))

    for spec in P.FIELD_SPECS:
        expected = distinct[spec.index]
        actual = data.field_array(spec.name)
        assert np.allclose(actual, expected), (
            f"{spec.name} (column {spec.index}) read {actual[0]}, expected {expected} "
            f"- column index is wrong"
        )


def check_field_specs_are_internally_consistent():
    indices = [s.index for s in P.FIELD_SPECS]
    assert len(indices) == len(set(indices)), f"duplicate column index: {indices}"
    assert max(indices) < P.MIN_EPW_FIELDS, "MIN_EPW_FIELDS is below the highest index used"
    names = [s.name for s in P.FIELD_SPECS]
    assert len(names) == len(set(names)), f"duplicate field name: {names}"
    for spec in P.FIELD_SPECS:
        assert spec.valid_min < spec.valid_max, spec.name
        assert spec.missing_at > spec.valid_min, spec.name


def check_wind_direction_is_not_exposed():
    """Excluded deliberately - linear interpolation of angles is wrong."""
    data = parse_text(make_epw())
    assert not hasattr(data, "wind_direction")
    assert 20 not in [s.index for s in P.FIELD_SPECS]


# ---------------------------------------------------------------------------
# 3. Malformed files are rejected
# ---------------------------------------------------------------------------

def check_rejects_missing_location_header():
    content = make_epw().replace("LOCATION,Testville", "NOTLOCATION,Testville", 1)
    expect_error(content, "not a LOCATION header")


def check_rejects_truncated_location_header():
    expect_error(make_epw(location="LOCATION,Testville,TS,TST"), "LOCATION header has")


def check_rejects_non_numeric_location_field():
    bad = LOCATION_LINE.replace(",7.25,", ",north,")
    expect_error(make_epw(location=bad), "non-numeric")


def check_rejects_leap_year_file():
    expect_error(make_epw(n_days=366), "8784")


def check_rejects_short_file():
    expect_error(make_epw(n_days=364), "data rows")


def check_rejects_multiple_data_periods():
    expect_error(make_epw(data_periods="DATA PERIODS,2,1,Data,Sunday, 1/ 1,12/31"),
                 "data periods")


def check_rejects_subhourly_records():
    expect_error(make_epw(data_periods="DATA PERIODS,1,4,Data,Sunday, 1/ 1,12/31"),
                 "records per hour")


def check_rejects_malformed_data_periods_line():
    expect_error(make_epw(data_periods="SOMETHING ELSE,1,1"), "DATA PERIODS")


def check_rejects_short_data_row():
    expect_error(make_epw(truncate_row=500), "fields")


def check_rejects_non_numeric_data_value():
    expect_error(make_epw(per_hour={10: {6: "warm"}}), "non-numeric")


def check_rejects_missing_file():
    try:
        P.parse_epw("/nonexistent/path/to/nowhere.epw")
    except P.EPWParseError as exc:
        assert "not found" in str(exc).lower()
        return
    raise AssertionError("expected EPWParseError for missing file")


# ---------------------------------------------------------------------------
# 4-5. Missing values and gap handling
# ---------------------------------------------------------------------------

def check_sentinel_values_are_detected_and_filled():
    gap = {h: {6: "99.9"} for h in range(100, 103)}      # 3-hour gap
    data = parse_text(make_epw(per_hour=gap))
    assert data.missing_counts["dry_bulb"] == 3, data.missing_counts
    assert data.interpolated_counts["dry_bulb"] == 3
    assert not np.isnan(data.dry_bulb).any(), "NaN survived gap filling"
    assert not data.is_pristine


def check_each_field_sentinel_is_recognised():
    """One field at a time, so a wrong sentinel constant cannot hide."""
    for spec in P.FIELD_SPECS:
        gap = {h: {spec.index: spec.missing_at} for h in range(50, 52)}
        data = parse_text(make_epw(per_hour=gap))
        assert data.missing_counts[spec.name] == 2, (
            f"{spec.name}: sentinel {spec.missing_at} not detected "
            f"({data.missing_counts[spec.name]} found)"
        )


def check_out_of_range_values_are_flagged_separately():
    bad = {h: {6: "-95.0"} for h in range(200, 202)}     # cold, but not a sentinel
    data = parse_text(make_epw(per_hour=bad))
    assert data.out_of_range_counts["dry_bulb"] == 2, data.out_of_range_counts
    assert data.missing_counts["dry_bulb"] == 0, "out-of-range miscounted as sentinel"
    assert not np.isnan(data.dry_bulb).any()


def check_interpolated_values_are_physically_sensible():
    per_hour = {100: {6: "10.0"}, 101: {6: "99.9"}, 102: {6: "99.9"}, 103: {6: "22.0"}}
    data = parse_text(make_epw(per_hour=per_hour))
    filled = data.dry_bulb[101:103]
    assert np.all((filled > 10.0) & (filled < 22.0)), filled
    assert filled[0] < filled[1], "interpolation is not monotonic across the gap"
    assert np.isclose(filled[0], 14.0) and np.isclose(filled[1], 18.0), filled


def check_gap_at_the_limit_is_accepted():
    gap = {h: {6: "99.9"} for h in range(300, 300 + P.MAX_GAP_HOURS)}
    data = parse_text(make_epw(per_hour=gap))
    assert data.interpolated_counts["dry_bulb"] == P.MAX_GAP_HOURS


def check_gap_beyond_the_limit_raises():
    gap = {h: {6: "99.9"} for h in range(300, 300 + P.MAX_GAP_HOURS + 1)}
    expect_error(make_epw(per_hour=gap), "gap")


def check_many_short_gaps_are_allowed():
    """Total missing can be large provided no single run is too long."""
    gap = {h: {6: "99.9"} for h in range(0, 8760, 100)}
    data = parse_text(make_epw(per_hour=gap))
    assert data.interpolated_counts["dry_bulb"] == len(gap)


def check_entirely_missing_field_raises():
    gap = {h: {21: "999"} for h in range(P.EXPECTED_ROWS)}
    expect_error(make_epw(per_hour=gap), "entirely missing")


def check_gap_across_year_boundary_is_filled_circularly():
    """Hour 8760 is adjacent to hour 1; a New Year gap uses both sides."""
    per_hour = {8757: {6: "99.9"}, 8758: {6: "99.9"}, 8759: {6: "99.9"},
                0: {6: "99.9"}, 1: {6: "99.9"}}
    per_hour[8756] = {6: "0.0"}
    per_hour[2] = {6: "12.0"}
    data = parse_text(make_epw(per_hour=per_hour))

    assert not np.isnan(data.dry_bulb).any()
    wrapped = np.concatenate([data.dry_bulb[8757:], data.dry_bulb[:2]])
    assert np.all(np.diff(wrapped) > 0), f"not monotonic across boundary: {wrapped}"
    assert np.all((wrapped > 0.0) & (wrapped < 12.0)), wrapped


def check_nan_run_lengths_merges_across_boundary():
    is_nan = np.zeros(24, dtype=bool)
    is_nan[[0, 1, 22, 23]] = True
    assert P._nan_run_lengths(is_nan) == [4], P._nan_run_lengths(is_nan)

    is_nan = np.zeros(24, dtype=bool)
    is_nan[[5, 6, 15]] = True
    assert sorted(P._nan_run_lengths(is_nan)) == [1, 2]

    assert P._nan_run_lengths(np.zeros(24, dtype=bool)) == []
    assert P._nan_run_lengths(np.ones(24, dtype=bool)) == [24]


# ---------------------------------------------------------------------------
# 6. The real project files
# ---------------------------------------------------------------------------

EPW_DIR = Path(__file__).parent / "epw_files"


def _real_files() -> list[Path]:
    return sorted(EPW_DIR.glob("*.epw")) if EPW_DIR.exists() else []


def check_real_files_present():
    files = _real_files()
    if not files:
        print("      (skipped - no EPW files; run climate/download_epw.py)")
        return
    assert len(files) == 11, f"expected 11 EPW files, found {len(files)}"


def check_real_files_parse():
    for path in _real_files():
        data = P.parse_epw(path)
        assert data.n_hours == P.EXPECTED_ROWS, f"{path.name}: {data.n_hours} hours"
        assert not np.isnan(data.dry_bulb).any(), f"{path.name}: NaN in dry_bulb"


def check_real_files_are_pristine():
    """TMYx files should need no substitution at all."""
    for path in _real_files():
        data = P.parse_epw(path)
        assert data.is_pristine, f"{path.name}: {data.quality_summary()}"


def check_real_files_have_plausible_locations():
    for path in _real_files():
        loc = P.parse_epw(path).location
        assert -90 <= loc.latitude <= 90, f"{path.name}: latitude {loc.latitude}"
        assert -180 <= loc.longitude <= 180, f"{path.name}: longitude {loc.longitude}"
        assert -500 <= loc.elevation <= 5000, f"{path.name}: elevation {loc.elevation}"
        assert -12 <= loc.tz_offset <= 14, f"{path.name}: tz {loc.tz_offset}"


def check_real_files_share_one_vintage():
    """Mixed vintages are the confound the set was rebuilt to remove."""
    sources = {p.name: P.parse_epw(p).location.data_source for p in _real_files()}
    if not sources:
        return
    assert len(set(sources.values())) == 1, f"mixed data sources: {sources}"


def check_colombo_is_tropical():
    """End-to-end sanity: a known file must carry recognisably its own weather."""
    path = EPW_DIR / "colombo.epw"
    if not path.exists():
        return
    data = P.parse_epw(path)
    assert data.dry_bulb.mean() > 25, f"Colombo mean DBT {data.dry_bulb.mean():.1f}"
    assert data.dry_bulb.min() > 10, f"Colombo min DBT {data.dry_bulb.min():.1f}"
    assert data.rel_humidity.mean() > 60, f"Colombo mean RH {data.rel_humidity.mean():.1f}"
    assert 6.0 < data.location.latitude < 8.0, data.location.latitude


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_CHECKS = [
    check_valid_file_parses,
    check_location_header_parsed,
    check_period_of_record_extracted,
    check_period_of_record_absent_is_not_an_error,
    check_date_columns_parsed,
    check_pristine_flag_true_for_clean_file,

    check_each_field_reads_its_own_column,
    check_field_specs_are_internally_consistent,
    check_wind_direction_is_not_exposed,

    check_rejects_missing_location_header,
    check_rejects_truncated_location_header,
    check_rejects_non_numeric_location_field,
    check_rejects_leap_year_file,
    check_rejects_short_file,
    check_rejects_multiple_data_periods,
    check_rejects_subhourly_records,
    check_rejects_malformed_data_periods_line,
    check_rejects_short_data_row,
    check_rejects_non_numeric_data_value,
    check_rejects_missing_file,

    check_sentinel_values_are_detected_and_filled,
    check_each_field_sentinel_is_recognised,
    check_out_of_range_values_are_flagged_separately,
    check_interpolated_values_are_physically_sensible,
    check_gap_at_the_limit_is_accepted,
    check_gap_beyond_the_limit_raises,
    check_many_short_gaps_are_allowed,
    check_entirely_missing_field_raises,
    check_gap_across_year_boundary_is_filled_circularly,
    check_nan_run_lengths_merges_across_boundary,

    check_real_files_present,
    check_real_files_parse,
    check_real_files_are_pristine,
    check_real_files_have_plausible_locations,
    check_real_files_share_one_vintage,
    check_colombo_is_tropical,
]


def main():
    total = failed = 0
    print("=" * 72)
    print("EPW PARSER - STRUCTURAL AND DATA-QUALITY CHECKS")
    print("=" * 72)
    for chk in ALL_CHECKS:
        total += 1
        try:
            chk()
            print(f"  PASS  {chk.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {chk.__name__}: {e}")
        except P.EPWParseError as e:
            failed += 1
            print(f"  ERROR {chk.__name__}: unexpected EPWParseError: {e}")
    print("=" * 72)
    print(f"RESULTS: {total - failed}/{total} passed, {failed} failed")
    print("=" * 72)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
