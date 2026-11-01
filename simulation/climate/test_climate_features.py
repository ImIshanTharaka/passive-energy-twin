"""
test_climate_features.py - Checks for climate feature computation and the registry.

Tests verify:
  1. The feature equations, on synthetic weather with hand-computable answers
  2. The 24-hour-mean degree-day convention (not the max/min midpoint)
  3. Seasons do not cancel: a hot summer and a cold winter give both HDD and CDD
  4. Diagnostics stay separate from model features
  5. BUILD rejects a weather file saved under the wrong climate's name
  6. The CSV round-trips exactly and is byte-identical when regenerated
  7. LOAD rejects missing files, wrong columns, NaNs and method mismatches
  8. London's WMO "037760" keeps its leading zero
  9. Train/serve parity: live computation == values stored in the CSV
 10. Every climate matches EnergyPlus's own .stat degree-days and zone
 11. Physical sanity on the real 11-climate set

Run:  python climate/test_climate_features.py
"""

from __future__ import annotations

import datetime as dt
import sys
import tempfile
from dataclasses import fields
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import climate_features as C  # type: ignore
import epw_parser as P        # type: ignore


# ---------------------------------------------------------------------------
# Synthetic weather
# ---------------------------------------------------------------------------

N = 8760
_DATES = [dt.date(1999, 1, 1) + dt.timedelta(days=d) for d in range(365)]
MONTHS = np.repeat([d.month for d in _DATES], 24)
DAYS = np.repeat([d.day for d in _DATES], 24)
HOURS = np.tile(np.arange(1, 25), 365)


def make_epw(dry_bulb=20.0, rh=50.0, ghi=100.0, wind=3.0,
             latitude=7.0, wmo="000001") -> P.EPWData:
    """Build an EPWData directly. Scalars broadcast to a full year."""
    def arr(v):
        return np.full(N, float(v)) if np.isscalar(v) else np.asarray(v, dtype=float)

    zeros = {s.name: 0 for s in P.FIELD_SPECS}
    return P.EPWData(
        path=Path("synthetic.epw"),
        location=P.EPWLocation(city="Synth", state="", country="TST", data_source="SYN",
                               wmo=wmo, latitude=latitude, longitude=80.0,
                               tz_offset=5.5, elevation=10.0),
        dry_bulb=arr(dry_bulb), dew_point=arr(5.0), rel_humidity=arr(rh),
        atm_pressure=arr(101325.0), glob_horiz_rad=arr(ghi),
        dir_normal_rad=arr(0.0), diff_horiz_rad=arr(0.0), wind_speed=arr(wind),
        month=MONTHS, day=DAYS, hour=HOURS,
        period_of_record="2011-2025",
        missing_counts=dict(zeros), out_of_range_counts=dict(zeros),
        interpolated_counts=dict(zeros),
    )


def features(**kw) -> C.ClimateFeatures:
    return C.compute_features(make_epw(**kw), "0A")


def expect_registry_error(fn, fragment: str) -> None:
    try:
        fn()
    except C.ClimateRegistryError as exc:
        assert fragment.lower() in str(exc).lower(), f"expected {fragment!r} in: {exc}"
        return
    raise AssertionError(f"expected ClimateRegistryError mentioning {fragment!r}")


# ---------------------------------------------------------------------------
# 1. Feature equations
# ---------------------------------------------------------------------------

def check_constant_cold_year():
    f = features(dry_bulb=10.0)
    assert f.hdd18 == 8 * 365, f.hdd18            # 8 degrees below 18, every day
    assert f.cdd10 == 0.0, f.cdd10                # never above 10


def check_constant_warm_year():
    f = features(dry_bulb=25.0)
    assert f.hdd18 == 0.0, f.hdd18
    assert f.cdd10 == 15 * 365, f.cdd10           # 15 degrees above 10, every day


def check_base_temperatures_are_boundaries():
    assert features(dry_bulb=18.0).hdd18 == 0.0   # exactly at base -> no heating
    assert features(dry_bulb=10.0).cdd10 == 0.0   # exactly at base -> no cooling


def check_annual_ghi_is_a_sum_in_kwh():
    assert features(ghi=100.0).annual_ghi == 876.0   # 100 Wh x 8760 h / 1000


def check_plain_means():
    f = features(dry_bulb=21.5, rh=63.0, wind=4.25)
    assert f.mean_dbt == 21.5 and f.mean_rh == 63.0 and f.mean_wind == 4.25, f


def check_latitude_comes_from_header():
    assert features(latitude=44.8831).latitude == 44.8831


def check_zone_label_passed_through():
    assert C.compute_features(make_epw(), "5B").ashrae_zone == "5B"


def check_features_are_rounded_to_declared_precision():
    f = features(dry_bulb=17.123456789, rh=61.987654321, wind=3.14159265)
    for name in C.CONTINUOUS_FEATURES:
        value = getattr(f, name)
        assert value == round(value, C.FEATURE_DECIMALS[name]), f"{name} = {value!r}"


def check_continuous_vector_order():
    f = features(dry_bulb=10.0)
    vec = f.continuous_vector()
    assert vec.shape == (7,)
    for i, name in enumerate(C.CONTINUOUS_FEATURES):
        assert vec[i] == getattr(f, name), name


# ---------------------------------------------------------------------------
# 2-3. Degree-day convention and the no-cancellation property
# ---------------------------------------------------------------------------

def check_uses_24_hour_mean_not_midpoint():
    """Each day: 23 hours at 0 degC, one hour at 24 degC.

    24-hour mean = 1.0  -> HDD18 = 17/day -> 6205
    midpoint     = 12.0 -> HDD18 =  6/day -> 2190
    Only the 24-hour mean reproduces EnergyPlus's own figures.
    """
    day = np.zeros(24)
    day[14] = 24.0
    f = features(dry_bulb=np.tile(day, 365))
    assert f.hdd18 == 17 * 365, f"{f.hdd18} - midpoint convention would give {6 * 365}"


def check_hot_summer_and_cold_winter_do_not_cancel():
    """Annual mean ~18 degC, yet the building needs both heating and cooling."""
    temps = np.concatenate([np.full(182 * 24, 30.0), np.full(183 * 24, 6.0)])
    f = features(dry_bulb=temps)
    assert abs(f.mean_dbt - 18.0) < 0.1, f.mean_dbt
    assert f.hdd18 == 183 * 12, f.hdd18
    assert f.cdd10 == 182 * 20, f.cdd10


def check_degree_days_rejects_bad_mode():
    try:
        C.degree_days(np.zeros(365), 18.0, "warming")        # type: ignore[arg-type]
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid mode")


def check_daily_mean_rejects_partial_day():
    try:
        C.daily_mean_temperature(np.zeros(8761))
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-whole number of days")


# ---------------------------------------------------------------------------
# 4. Diagnostics
# ---------------------------------------------------------------------------

def check_diagnostics_are_separate_from_features():
    feature_fields = {f.name for f in fields(C.ClimateFeatures)}
    diag_fields = {f.name for f in fields(C.ClimateDiagnostics)}
    assert feature_fields == set(C.CONTINUOUS_FEATURES) | {"ashrae_zone"}, feature_fields
    assert not feature_fields & diag_fields, feature_fields & diag_fields


def check_seasonal_ghi_is_hemisphere_aware():
    sunny_apr_sep = np.where((MONTHS >= 4) & (MONTHS <= 9), 100.0, 0.0)
    north = C.compute_diagnostics(make_epw(ghi=sunny_apr_sep, latitude=40.0))
    south = C.compute_diagnostics(make_epw(ghi=sunny_apr_sep, latitude=-33.0))
    assert north.ghi_summer > 0 and north.ghi_winter == 0.0, north
    assert south.ghi_summer == 0.0 and south.ghi_winter > 0, south


def check_diagnostic_temperature_range():
    temps = np.tile(np.linspace(5.0, 25.0, 24), 365)
    d = C.compute_diagnostics(make_epw(dry_bulb=temps))
    assert d.dbt_min == 5.0 and d.dbt_max == 25.0, (d.dbt_min, d.dbt_max)
    assert d.n_hours == N and d.n_substituted == 0


# ---------------------------------------------------------------------------
# Real-data helpers
# ---------------------------------------------------------------------------

_REAL: dict | None = None


def real_registry() -> dict | None:
    """Build the real registry once; None if EPW files are absent."""
    global _REAL
    if _REAL is None:
        if not C.DEFAULT_EPW_DIR.exists() or not any(C.DEFAULT_EPW_DIR.glob("*.epw")):
            print("      (skipped - no EPW files)")
            return None
        _REAL = C.build_climate_registry()
    return _REAL


def write_temp_config(tmp: Path, climates: dict) -> Path:
    import yaml
    path = tmp / "climate_config.yaml"
    path.write_text(yaml.safe_dump({"climates": climates}))
    return path


CHICAGO_ENTRY = {"city": "Chicago", "country": "USA", "role": "training", "koppen": "Dfa",
                 "ashrae_zone": "5A", "wmo": "725300", "epw_filename": "chicago.epw"}


# ---------------------------------------------------------------------------
# 5. BUILD
# ---------------------------------------------------------------------------

def check_build_real_registry():
    reg = real_registry()
    if reg is None:
        return
    assert len(reg) == 11, f"{len(reg)} climates"
    assert len(C.training_keys(reg)) == 10, C.training_keys(reg)


def check_holdout_is_london_only():
    reg = real_registry()
    if reg is None:
        return
    assert C.holdout_keys(reg) == ["london"], C.holdout_keys(reg)
    assert "london" not in C.training_keys(reg)


def check_build_rejects_weather_file_under_wrong_name():
    """Denver's weather copied in as chicago.epw must fail, not silently build.

    The filename is correct for the key, so the naming rule passes - only
    the WMO check inside the file can catch this.
    """
    if real_registry() is None:
        return
    with tempfile.TemporaryDirectory() as td:
        epw_dir = Path(td) / "epw"
        epw_dir.mkdir()
        (epw_dir / "chicago.epw").write_bytes((C.DEFAULT_EPW_DIR / "denver.epw").read_bytes())
        cfg = write_temp_config(Path(td), {"chicago": CHICAGO_ENTRY})
        expect_registry_error(lambda: C.build_climate_registry(cfg, epw_dir), "WMO")


def check_build_rejects_filename_not_matching_key():
    """A filename carrying zone or role (the old 5A_Chicago.epw style) is refused."""
    with tempfile.TemporaryDirectory() as td:
        old_style = dict(CHICAGO_ENTRY, epw_filename="5A_Chicago.epw")
        cfg = write_temp_config(Path(td), {"chicago": old_style})
        expect_registry_error(lambda: C.build_climate_registry(cfg, Path(td)), "chicago.epw")


def check_build_rejects_missing_epw():
    with tempfile.TemporaryDirectory() as td:
        cfg = write_temp_config(Path(td), {"chicago": CHICAGO_ENTRY})
        empty = Path(td) / "empty"
        empty.mkdir()
        expect_registry_error(lambda: C.build_climate_registry(cfg, empty), "not found")


def check_build_skips_excluded_climates():
    if real_registry() is None:
        return
    with tempfile.TemporaryDirectory() as td:
        cfg = write_temp_config(Path(td), {
            "chicago": CHICAGO_ENTRY,
            "denver": dict(CHICAGO_ENTRY, role="excluded", wmo="725650",
                           epw_filename="denver.epw"),
        })
        reg = C.build_climate_registry(cfg, C.DEFAULT_EPW_DIR)
        assert list(reg) == ["chicago"], list(reg)


def check_build_rejects_invalid_role():
    with tempfile.TemporaryDirectory() as td:
        cfg = write_temp_config(Path(td), {"chicago": dict(CHICAGO_ENTRY, role="training_set")})
        expect_registry_error(lambda: C.build_climate_registry(cfg, Path(td)), "invalid role")


# ---------------------------------------------------------------------------
# 6-8. CSV round trip and LOAD
# ---------------------------------------------------------------------------

def check_csv_round_trip_is_exact():
    reg = real_registry()
    if reg is None:
        return
    with tempfile.TemporaryDirectory() as td:
        path = C.write_climate_metadata(reg, Path(td) / "m.csv")
        loaded = C.load_climate_registry(path, C.DEFAULT_EPW_DIR)
        assert set(loaded) == set(reg)
        for key in reg:
            assert loaded[key].features == reg[key].features, key
            assert loaded[key].diagnostics == reg[key].diagnostics, key
            assert loaded[key].epw_path == reg[key].epw_path, key


def check_csv_is_byte_identical_on_regeneration():
    reg = real_registry()
    if reg is None:
        return
    with tempfile.TemporaryDirectory() as td:
        a = C.write_climate_metadata(reg, Path(td) / "a.csv").read_bytes()
        b = C.write_climate_metadata(C.build_climate_registry(), Path(td) / "b.csv").read_bytes()
        assert a == b, "regenerated CSV differs"


def check_wmo_keeps_leading_zero():
    reg = real_registry()
    if reg is None:
        return
    with tempfile.TemporaryDirectory() as td:
        path = C.write_climate_metadata(reg, Path(td) / "m.csv")
        loaded = C.load_climate_registry(path, C.DEFAULT_EPW_DIR)
        assert loaded["london"].wmo == "037760", repr(loaded["london"].wmo)


def _temp_csv(td: str, mutate=None) -> Path:
    reg = real_registry()
    path = C.write_climate_metadata(reg, Path(td) / "m.csv")
    if mutate:
        path.write_text(mutate(path.read_text()))
    return path


def check_loader_rejects_missing_epw_file():
    if real_registry() is None:
        return
    with tempfile.TemporaryDirectory() as td:
        path = _temp_csv(td)
        empty = Path(td) / "no_epws"
        empty.mkdir()
        expect_registry_error(lambda: C.load_climate_registry(path, empty), "EPW file missing")
        assert len(C.load_climate_registry(path, empty, check_files=False)) == 11


def check_loader_rejects_wrong_columns():
    if real_registry() is None:
        return
    with tempfile.TemporaryDirectory() as td:
        path = _temp_csv(td, lambda t: t.replace("hdd18", "HDD_18", 1))
        expect_registry_error(lambda: C.load_climate_registry(path), "columns")


def check_loader_rejects_nan_feature():
    if real_registry() is None:
        return
    with tempfile.TemporaryDirectory() as td:
        def poison(text: str) -> str:
            lines = text.splitlines()
            cols = lines[1].split(",")
            cols[C.CSV_COLUMNS.index("hdd18")] = "nan"
            lines[1] = ",".join(cols)
            return "\n".join(lines) + "\n"
        path = _temp_csv(td, poison)
        expect_registry_error(lambda: C.load_climate_registry(path), "hdd18")


def check_loader_rejects_method_mismatch():
    if real_registry() is None:
        return
    with tempfile.TemporaryDirectory() as td:
        path = _temp_csv(td, lambda t: t.replace(C.DEGREE_DAY_METHOD, "midpoint"))
        expect_registry_error(lambda: C.load_climate_registry(path), "degree-day method")


def check_loader_rejects_missing_csv():
    expect_registry_error(lambda: C.load_climate_registry("/nonexistent/m.csv"), "not found")


# ---------------------------------------------------------------------------
# 9. Train/serve parity
# ---------------------------------------------------------------------------

def check_live_computation_equals_stored_values():
    """What Phase 5 computes from an uploaded EPW == what training read from CSV."""
    reg = real_registry()
    if reg is None:
        return
    with tempfile.TemporaryDirectory() as td:
        stored = C.load_climate_registry(C.write_climate_metadata(reg, Path(td) / "m.csv"))
        for key, record in stored.items():
            live = C.compute_features(P.parse_epw(record.epw_path),
                                      record.features.ashrae_zone)
            assert live == record.features, f"{key}: live {live} != stored {record.features}"


# ---------------------------------------------------------------------------
# 10. Agreement with EnergyPlus
# ---------------------------------------------------------------------------

def check_stat_reader_uses_weather_file_section():
    """Chicago: (wthr file) HDD18 = 3124. The (standard) section says 3346."""
    ref = C.read_stat_reference(C.DEFAULT_STAT_DIR / "chicago.stat")
    if ref is None:
        return
    assert ref.hdd18 == 3124.0, f"read {ref.hdd18} - wrong .stat section?"
    assert ref.cdd10 == 1938.0, ref.cdd10
    assert ref.ashrae_zone == "5A", ref.ashrae_zone


def check_every_climate_matches_energyplus():
    reg = real_registry()
    if reg is None:
        return
    failures = [f"{label}: {detail}" for label, ok, detail in C.validate_registry(reg) if not ok]
    assert not failures, "; ".join(failures)


# ---------------------------------------------------------------------------
# 11. Physical sanity on the real set
# ---------------------------------------------------------------------------

def check_physical_orderings():
    reg = real_registry()
    if reg is None:
        return
    f = {k: r.features for k, r in reg.items()}
    argmax = lambda attr: max(f, key=lambda k: getattr(f[k], attr))
    argmin = lambda attr: min(f, key=lambda k: getattr(f[k], attr))

    assert f["colombo"].hdd18 == 0.0 and f["chennai"].hdd18 == 0.0, "tropics need heating?"
    assert argmax("hdd18") == "minneapolis", argmax("hdd18")
    assert argmax("cdd10") == "chennai", argmax("cdd10")
    assert argmin("mean_rh") == "phoenix", argmin("mean_rh")
    assert argmin("annual_ghi") == "london", argmin("annual_ghi")
    assert argmax("annual_ghi") == "phoenix", argmax("annual_ghi")
    elev = {k: r.diagnostics.elevation for k, r in reg.items()}
    assert max(elev, key=elev.get) == "denver", elev


def check_all_real_files_were_pristine():
    reg = real_registry()
    if reg is None:
        return
    dirty = {k: r.diagnostics.n_substituted for k, r in reg.items() if r.diagnostics.n_substituted}
    assert not dirty, f"substituted values: {dirty}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_CHECKS = [
    check_constant_cold_year,
    check_constant_warm_year,
    check_base_temperatures_are_boundaries,
    check_annual_ghi_is_a_sum_in_kwh,
    check_plain_means,
    check_latitude_comes_from_header,
    check_zone_label_passed_through,
    check_features_are_rounded_to_declared_precision,
    check_continuous_vector_order,

    check_uses_24_hour_mean_not_midpoint,
    check_hot_summer_and_cold_winter_do_not_cancel,
    check_degree_days_rejects_bad_mode,
    check_daily_mean_rejects_partial_day,

    check_diagnostics_are_separate_from_features,
    check_seasonal_ghi_is_hemisphere_aware,
    check_diagnostic_temperature_range,

    check_build_real_registry,
    check_holdout_is_london_only,
    check_build_rejects_weather_file_under_wrong_name,
    check_build_rejects_filename_not_matching_key,
    check_build_rejects_missing_epw,
    check_build_skips_excluded_climates,
    check_build_rejects_invalid_role,

    check_csv_round_trip_is_exact,
    check_csv_is_byte_identical_on_regeneration,
    check_wmo_keeps_leading_zero,
    check_loader_rejects_missing_epw_file,
    check_loader_rejects_wrong_columns,
    check_loader_rejects_nan_feature,
    check_loader_rejects_method_mismatch,
    check_loader_rejects_missing_csv,

    check_live_computation_equals_stored_values,

    check_stat_reader_uses_weather_file_section,
    check_every_climate_matches_energyplus,

    check_physical_orderings,
    check_all_real_files_were_pristine,
]


def main():
    total = failed = 0
    print("=" * 72)
    print("CLIMATE FEATURES + REGISTRY - CHECKS")
    print("=" * 72)
    for chk in ALL_CHECKS:
        total += 1
        try:
            chk()
            print(f"  PASS  {chk.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {chk.__name__}: {e}")
        except (C.ClimateRegistryError, P.EPWParseError) as e:
            failed += 1
            print(f"  ERROR {chk.__name__}: {type(e).__name__}: {e}")
    print("=" * 72)
    print(f"RESULTS: {total - failed}/{total} passed, {failed} failed")
    print("=" * 72)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
