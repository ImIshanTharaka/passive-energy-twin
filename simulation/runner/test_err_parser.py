"""
test_err_parser.py - Checks for runner/err_parser.py (Sub-task 2.5.2).

Fixtures are verbatim EnergyPlus 24.1 .err files from this project's own runs
(clean smoke run, a run with 2 warnings, the early run that died on invalid
summary-report names), plus a recurring-error block in EnergyPlus's format.

Run:  python runner/test_err_parser.py
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner import err_parser as E          # noqa: E402

CLEAN = """Program Version,EnergyPlus, Version 24.1.0-9d7789a3ac, YMD=2026.09.27 19:54,
   ************* Testing Individual Branch Integrity
   ************* All Branches passed integrity testing
   ************* No node connection errors were found.
   ************* Beginning Simulation
   ************* Simulation Error Summary *************
   ************* EnergyPlus Warmup Error Summary. During Warmup: 0 Warning; 0 Severe Errors.
   ************* EnergyPlus Sizing Error Summary. During Sizing: 0 Warning; 0 Severe Errors.
   ************* EnergyPlus Completed Successfully-- 0 Warning; 0 Severe Errors; Elapsed Time=00hr 00min  9.62sec
"""

TWO_WARNINGS = """Program Version,EnergyPlus, Version 24.1.0-9d7789a3ac, YMD=2026.09.27 19:22,
   ** Warning ** GetSurfaceData: World Coordinate System selected.  Any non-zero Building/Zone North Axes or non-zero Zone Origins are ignored.
   **   ~~~   ** These may be used in daylighting reference point coordinate calculations but not in normal geometry inputs.
   ** Warning ** GetSurfaceData: Construction INTERIORCEILINGFLOOR of interzone surface ZONE_F0_CORE_CEILING does not have the same materials in the reverse order as the construction INTERIORCEILINGFLOOR of adjacent surface ZONE_F1_CORE_FLOOR
   **   ~~~   ** or the properties of the reversed layers are not correct due to differing layer front and back side values
   **   ~~~   ** ...but Nominal U values are similar, diff=[0.0000] ... simulation proceeds.
   **   ~~~   ** ...if the two zones are expected to have significantly different temperatures, the proper "reverse" construction should be created.
   **   ~~~   ** ...this problem for this pair will not be reported again.
   ************* Testing Individual Branch Integrity
   ************* All Branches passed integrity testing
   ************* Beginning Simulation
   ************* Simulation Error Summary *************
   ************* EnergyPlus Warmup Error Summary. During Warmup: 0 Warning; 0 Severe Errors.
   ************* EnergyPlus Sizing Error Summary. During Sizing: 0 Warning; 0 Severe Errors.
   ************* EnergyPlus Completed Successfully-- 2 Warning; 0 Severe Errors; Elapsed Time=00hr 00min  9.69sec
"""

FATAL = """Program Version,EnergyPlus, Version 24.1.0-9d7789a3ac, YMD=2026.09.27 19:09,
   ** Severe  ** <root>[Output:Table:SummaryReports][Output:Table:SummaryReports 1][reports][3][report_name] - "ComfortAndSetpointNotMetSummary" - Failed to match against any enum values.
   ** Severe  ** <root>[Output:Table:SummaryReports][Output:Table:SummaryReports 1][reports][5][report_name] - "ZoneCoolingSummary" - Failed to match against any enum values.
   ** Severe  ** <root>[Output:Table:SummaryReports][Output:Table:SummaryReports 1][reports][6][report_name] - "ZoneHeatingSummary" - Failed to match against any enum values.
   ** Severe  ** <root>[Output:Table:SummaryReports][Output:Table:SummaryReports 1][reports][8][report_name] - "WindowEnergyTransferSummary" - Failed to match against any enum values.
   **  Fatal  ** Errors occurred on processing input file. Preceding condition(s) cause termination.
   ...Summary of Errors that led to program termination:
   ..... Reference severe error count=4
   ************* Warning:  Node connection errors not checked - most system input has not been read (see previous warning).
   ************* Fatal error -- final processing.  Program exited before simulations began.  See previous error messages.
   ************* EnergyPlus Warmup Error Summary. During Warmup: 0 Warning; 0 Severe Errors.
   ************* EnergyPlus Sizing Error Summary. During Sizing: 0 Warning; 0 Severe Errors.
   ************* EnergyPlus Terminated--Fatal Error Detected. 0 Warning; 4 Severe Errors; Elapsed Time=00hr 00min  0.08sec
"""

RECURRING = """Program Version,EnergyPlus, Version 24.1.0-9d7789a3ac, YMD=2026.10.07 10:00,
   ** Warning ** CalcHeatBalanceInsideSurf: Zone="ZONE_F4_PERIMETER_S" has view factor enforced reciprocity
   **   ~~~   **  and/or completeness errors.
   ************* Beginning Simulation
   ** Warning ** CalcHeatBalanceInsideSurf: Temperature (high) out of bounds (201.4] for zone="ZONE_F2_CORE", for surface="ZONE_F2_CORE_ROOF"
   **   ~~~   **  Environment=RUN PERIOD 1, at Simulation time=07/14 13:00 - 13:10
   ************* Simulation Error Summary *************
   ************* ===== Recurring Error Summary =====
   ************* The following recurring error messages occurred.
   *************
   *************  ** Warning ** CalcHeatBalanceInsideSurf: Temperature (high) out of bounds for zone="ZONE_F2_CORE"
   *************  **   ~~~   **   This error occurred 37 total times;
   *************  **   ~~~   **   during Warmup 0 times;
   *************  **   ~~~   **   during Sizing 0 times.
   *************  **   ~~~   **   Max=201.4 C  Min=100.2 C
   *************
   ************* EnergyPlus Warmup Error Summary. During Warmup: 0 Warning; 0 Severe Errors.
   ************* EnergyPlus Completed Successfully-- 3 Warning; 0 Severe Errors; Elapsed Time=00hr 01min 2.50sec
"""


def check_clean_run():
    r = E.parse_err_text(CLEAN)
    assert r.version == "24.1.0" and r.completed and not r.terminated_fatal
    assert (r.reported_warnings, r.reported_severe) == (0, 0) and r.messages == []
    assert abs(r.elapsed_s - 9.62) < 1e-9 and r.clean


def check_warnings_with_continuations():
    r = E.parse_err_text(TWO_WARNINGS)
    assert r.completed and r.reported_warnings == 2
    ws = r.by_level("Warning")
    assert len(ws) == r.reported_warnings, "every warning counted by EnergyPlus is parsed"
    assert len(ws[0].details) == 1 and len(ws[1].details) == 4
    assert "diff=[0.0000]" in ws[1].details[1]
    assert [m.category for m in ws] == ["world_coordinate_system", "interzone_construction_reverse"]
    assert all(m.benign for m in ws) and r.clean


def check_fatal_run():
    r = E.parse_err_text(FATAL)
    assert not r.completed and r.terminated_fatal and r.reported_severe == 4
    assert len(r.by_level("Severe")) == 4 and r.n_fatal == 1 and not r.clean
    assert {m.category for m in r.by_level("Severe")} == {"input_enum"}
    assert r.by_level("Fatal")[0].category == E.FATAL_CATEGORY
    assert r.by_level("Fatal")[0].details == [], "'...Summary' lines are not ~~~ continuations"
    assert abs(r.elapsed_s - 0.08) < 1e-9


def check_banner_warning_is_not_a_message():
    r = E.parse_err_text(FATAL)
    assert not any("Node connection" in m.text for m in r.messages)


def check_recurring_block():
    r = E.parse_err_text(RECURRING)
    rec = [m for m in r.messages if m.recurring]
    assert len(rec) == 1 and rec[0].count == 37
    assert rec[0].details == ["during Warmup 0 times;", "during Sizing 0 times.",
                              "Max=201.4 C  Min=100.2 C"]
    assert rec[0].category == "inside_surface_temperature" and not rec[0].benign
    first = [m for m in r.messages if not m.recurring]
    assert len(first) == 2 and all(m.count == 1 for m in first)
    assert "Environment=RUN PERIOD 1" in first[1].details[0]
    assert not r.clean and abs(r.elapsed_s - 62.5) < 1e-9


def check_signature_groups_across_names_and_numbers():
    a = E.signature('CalcHeatBalanceInsideSurf: Temperature (high) out of bounds (201.4] '
                    'for zone="ZONE_F2_CORE", for surface="ZONE_F2_CORE_ROOF"')
    b = E.signature('CalcHeatBalanceInsideSurf: Temperature (high) out of bounds (187.9] '
                    'for zone="ZONE_F4_PERIMETER_N", for surface="ZONE_F4_PERIMETER_N_WALL"')
    assert a == b, (a, b)
    c = E.signature("GetSurfaceData: Construction INTERIORCEILINGFLOOR of interzone surface "
                    "ZONE_F0_CORE_CEILING does not have the same materials")
    d = E.signature("GetSurfaceData: Construction INTERIORCEILINGFLOOR of interzone surface "
                    "ZONE_F3_PERIMETER_E_CEILING does not have the same materials")
    assert c == d and "<NAME>" in c and "GetSurfaceData" in c


def check_unused_construction_is_benign():
    """Verbatim from pilot_E01 (1-storey): interior floor/ceiling constructions unused."""
    r = E.parse_err_text(
        "   ** Warning ** CheckUsedConstructions: There are 2 nominally unused constructions in input.\n"
        "   **   ~~~   ** For explicit details on each unused construction, use Output:Diagnostics,DisplayExtraWarnings;\n"
        "   ************* EnergyPlus Completed Successfully-- 1 Warning; 0 Severe Errors; Elapsed Time=00hr 00min  5.00sec\n")
    assert r.messages[0].category == "unused_construction" and r.messages[0].benign and r.clean


def check_unknown_message_is_other_and_not_benign():
    r = E.parse_err_text("   ** Warning ** Something entirely new happened\n")
    assert r.messages[0].category == "other" and not r.messages[0].benign


DENVER_ERR = """   ************* Beginning Simulation
   ** Warning ** Temperature out of range [-100. to 200.] (PsyPsatFnTemp)
   **   ~~~   **  Routine=PsyTwbFnTdbWPb, Environment=ANNUALRUN, at Simulation time=10/14 01:00 - 01:10
   **   ~~~   **  Input Temperature=-2572.62
   ************* Simulation Error Summary *************
   *************
   ************* ===== Recurring Error Summary =====
   ************* The following recurring error messages occurred.
   *************
   *************  ** Warning ** Temperature out of range [-100. to 200.] (PsyPsatFnTemp)
   *************  **   ~~~   **   This error occurred 3 total times;
   *************  **   ~~~   **   during Warmup 0 times;
   *************  **   ~~~   **   during Sizing 0 times.
   *************  **   ~~~   **   Max=-119.748493 C  Min=-2572.621948 C
   *************
   ************* EnergyPlus Completed Successfully-- 2 Warning; 0 Severe Errors; Elapsed Time=00hr 00min 40.00sec
"""


def check_outdoor_wetbulb_warning_is_benign():
    """Verbatim from main_A001_denver (2.6 smoke): both entries benign, run clean."""
    r = E.parse_err_text(DENVER_ERR)
    assert [m.category for m in r.messages] == ["outdoor_wetbulb_psychrometric"] * 2, \
        [m.category for m in r.messages]
    assert r.messages[1].recurring and r.messages[1].count == 3
    assert r.clean


def check_recurring_boilerplate_does_not_categorise():
    """'during Sizing 0 times.' must not turn an unknown recurring message into 'sizing'."""
    text = DENVER_ERR.replace("(PsyPsatFnTemp)", "(SomethingNew)").replace("Routine=PsyTwbFnTdbWPb", "Routine=X")
    r = E.parse_err_text(text)
    assert [m.category for m in r.messages] == ["other", "other"], [m.category for m in r.messages]
    assert not r.clean


def check_every_category_regex_matches_something_plausible():
    samples = {
        "shading_geometry": "DetermineShadowingCombinations: There are 12 surfaces which are casting shadows",
        "convergence": "SimHVAC: Maximum iterations (20) exceeded for all HVAC loops",
        "subsurface_geometry": "Window ZONE_F1_PERIMETER_S_WIN extends outside base surface",
        "surface_geometry": "CalculateZoneVolume: 1 zone is not fully enclosed",
        "weather_location_mismatch": "Weather file location will be used rather than entered Location object.",
    }
    for cat, text in samples.items():
        got = E.parse_err_text(f"   ** Warning ** {text}\n").messages[0].category
        assert got == cat, (text, got)


def check_real_files_on_disk():
    """Every .err already in outputs/ parses, and parsed counts match EnergyPlus's totals."""
    files = list((ROOT / "outputs").rglob("eplusout.err"))
    for f in files:
        r = E.parse_err(f)
        assert r.version and r.reported_warnings is not None, f
        nonrec = [m for m in r.messages if not m.recurring]
        n_sev = sum(m.level == "Severe" for m in nonrec)
        # EnergyPlus may group several severe errors into ONE message ("There are 4
        # degenerate surfaces" -> count 4, 1 message; pilot P15), so only require that
        # severe messages exist exactly when EnergyPlus counted any, and never exceed it.
        assert (n_sev > 0) == (r.reported_severe > 0) and n_sev <= r.reported_severe, f


def check_aggregate_batch_folder():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        for rid, text in (("r_clean", CLEAN), ("r_warn", TWO_WARNINGS),
                          ("r_fatal", FATAL), ("r_rec", RECURRING)):
            (t / rid).mkdir()
            (t / rid / "eplusout.err").write_text(text)
        summaries, messages = E.aggregate(t)
        assert [s["run_id"] for s in summaries] == ["r_clean", "r_fatal", "r_rec", "r_warn"]
        by = {s["run_id"]: s for s in summaries}
        assert by["r_clean"]["clean"] == 1 and by["r_warn"]["clean"] == 1
        assert by["r_fatal"]["terminated_fatal"] == 1 and by["r_fatal"]["n_fatal"] == 1
        assert by["r_rec"]["n_nonbenign_warnings"] == 3
        assert "input_enum:4" in by["r_fatal"]["categories"]
        with (t / "err_summary.csv").open() as fh:
            assert len(list(csv.DictReader(fh))) == 4
        with (t / "err_messages.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == len(messages) == 2 + 5 + 3
        table = E.category_table(messages, 4)
        assert "input_enum" in table and "1/4" in table


ALL_CHECKS = [
    check_clean_run,
    check_warnings_with_continuations,
    check_fatal_run,
    check_banner_warning_is_not_a_message,
    check_recurring_block,
    check_signature_groups_across_names_and_numbers,
    check_unused_construction_is_benign,
    check_unknown_message_is_other_and_not_benign,
    check_outdoor_wetbulb_warning_is_benign,
    check_recurring_boilerplate_does_not_categorise,
    check_every_category_regex_matches_something_plausible,
    check_real_files_on_disk,
    check_aggregate_batch_folder,
]


def main():
    total = failed = 0
    print("=" * 72)
    print("ERR PARSER (2.5.2) - CHECKS")
    print("=" * 72)
    for chk in ALL_CHECKS:
        total += 1
        try:
            chk()
            print(f"  PASS  {chk.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {chk.__name__}: {e}")
        except Exception as e:                       # noqa: BLE001
            failed += 1
            print(f"  ERROR {chk.__name__}: {type(e).__name__}: {e}")
    print("=" * 72)
    print(f"RESULTS: {total - failed}/{total} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
