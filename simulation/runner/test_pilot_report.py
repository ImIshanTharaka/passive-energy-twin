"""
test_pilot_report.py - Checks for runner/pilot_report.py (Sub-task 2.5.6).

Unit checks on the decision logic with small synthetic frames, plus an
end-to-end build on outputs/pilot when that folder exists.

Run:  python runner/test_pilot_report.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner import pilot_report as P        # noqa: E402


def check_spearman():
    a = pd.Series([1, 2, 3, 4, 5])
    assert abs(P.spearman(a, a ** 3) - 1) < 1e-12
    assert abs(P.spearman(a, -a) + 1) < 1e-12


def _colombo(heat, floor_r):
    return {"data": pd.DataFrame({"climate_key": ["colombo"] * len(heat),
                                  "heat_sens_kwh_m2": heat, "floor_r_value": floor_r})}


def check_ground_diagnostic_logic():
    assert P.check_ground(_colombo([4.6, 3.0, 2.0, 1.0, 0.5], [0.5, 1, 2, 3, 5]))[0] == "FAIL"
    assert P.check_ground(_colombo([0.6, 0.5, 0.4, 0.3, 0.2], [0.5, 1, 2, 3, 5]))[0] == "PASS", \
        "tiny heating is not a failure even if it trends"
    assert P.check_ground(_colombo([0.1, 0.3, 0.2, 0.4, 0.1], [0.5, 1, 2, 3, 5]))[0] == "PASS"


def check_vtpair_detects_difference():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / "run_log.csv").write_text("run_id,status\n")
        pd.DataFrame({"run_id": ["vtpair_X_vtA_c", "vtpair_X_vtB_c"], "heat": [1.0, 1.0],
                      "cool": [5.0, 5.0]}).to_csv(t / "results.csv", index=False)
        assert P.check_vtpair(t)[0] == "PASS"
        pd.DataFrame({"run_id": ["vtpair_X_vtA_c", "vtpair_X_vtB_c"], "heat": [1.0, 1.0],
                      "cool": [5.0, 5.1]}).to_csv(t / "results.csv", index=False)
        assert P.check_vtpair(t)[0] == "FAIL"
    assert P.check_vtpair(None)[0] == "INFO"


def check_end_to_end_on_pilot():
    batch = ROOT / "outputs" / "pilot"
    if not (batch / "run_log.csv").exists():
        return
    with tempfile.TemporaryDirectory() as t:
        copy = Path(t) / "pilot"
        shutil.copytree(batch, copy, ignore=shutil.ignore_patterns("report_figures", "pilot_report.md"))
        text, checks = P.build_report(ROOT / "sampling" / "samples_pilot.csv", copy, None)
        titles = [c[1][0] for c in checks]
        assert titles == list("RPDEVKMGT"), titles
        assert all(c[0] in ("PASS", "WARN", "FAIL", "INFO") for c in checks)
        assert len(list((copy / "report_figures").glob("*.png"))) == 4
        assert "## R. Reliability" in text


ALL_CHECKS = [check_spearman, check_ground_diagnostic_logic, check_vtpair_detects_difference,
              check_end_to_end_on_pilot]


def main():
    total = failed = 0
    print("=" * 72)
    print("PILOT REPORT (2.5.6) - CHECKS")
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
