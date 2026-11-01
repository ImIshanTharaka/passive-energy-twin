"""
test_campaign_report.py - Checks for runner/campaign_report.py (Sub-task 2.6).

Decision logic on small synthetic frames, plus an end-to-end build on a copy of
outputs/pilot when that folder exists (its D-check values must equal the pilot
report's, which used the same LHS designs).

Run:  python runner/test_campaign_report.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner import campaign_report as C        # noqa: E402

T = list(C.TARGETS)
META = pd.read_csv(C.METADATA).set_index("key")


def _D(**kw):
    base = {"samples": pd.DataFrame(), "log": pd.DataFrame(), "data": pd.DataFrame(),
            "err_sum": pd.DataFrame(), "err_msg": pd.DataFrame(), "res_err": pd.DataFrame(),
            "meta": META, "order": ["minneapolis", "colombo"], "training": ["minneapolis", "colombo"],
            "holdout_climates": [], "batches": [], "partial": False}
    base.update(kw)
    return base


def _completeness_case(statuses, bad_target=False, partial=False):
    ids = [f"main_A00{i}_colombo" for i in range(len(statuses))]
    S = pd.DataFrame({"run_id": ids, "batch": "main", "climate_key": "colombo", "design_set": "anchor"})
    log = pd.DataFrame({"run_id": ids, "status": statuses, "message": ""})
    ok = [r for r, s in zip(ids, statuses) if s == "ok"]
    d = pd.DataFrame({"run_id": ok, **{t: 1.0 for t in T}})
    if bad_target:
        d.loc[0, T[0]] = np.nan
    return C.check_completeness(_D(samples=S, log=log, data=d, partial=partial))[0]


def check_completeness_logic():
    assert _completeness_case(["ok"] * 4) == "PASS"
    assert _completeness_case(["ok"] * 3 + ["failed"]) == "FAIL", "75 % parsed is below the 99 % WARN band"
    assert _completeness_case(["ok"] * 199 + ["failed"]) == "WARN"
    assert _completeness_case(["ok"] * 4, bad_target=True) == "FAIL"
    assert _completeness_case(["ok"] * 3 + ["failed"], partial=True) == "INFO"


def check_anchor_logic():
    S = pd.DataFrame({"run_id": ["a", "b", "c", "d"], "design_set": "anchor", "design_id": ["A1", "A1", "A2", "A2"],
                      "climate_key": ["colombo", "chicago", "colombo", "chicago"]})
    d = S.copy()
    assert C.check_anchors(_D(samples=S, data=d))[0] == "PASS"
    st, _, txt = C.check_anchors(_D(samples=S, data=d.iloc[:3]))
    assert st == "WARN" and "A2" in txt and "chicago" in txt


def _direction_data(seed=0, n=200, wrong_weak=False, break_cooling=False):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({x: r.uniform(0, 1, n) for x in C.INPUTS})
    d["shgc_x_wwr"] = r.uniform(0, 1, n)
    d["mean_overhang"] = r.uniform(0, 1, n)
    d["cool_sens_kwh_m2"] = r.uniform(0, 1, n) if break_cooling else d.shgc_x_wwr + 0.2 * r.uniform(0, 1, n)
    d["peak_cool_steady_w_m2"] = d.shgc_x_wwr + 0.2 * r.uniform(0, 1, n)
    d["heat_sens_kwh_m2"] = d.glazing_u - 0.05 * d.wall_r_value + (0.3 * d.wall_r_value if wrong_weak else 0)
    d["heat_sens_kwh_m2"] -= 0.05 * d.roof_r_value - 0.05 * d.infiltration_ach
    d["peak_heat_steady_w_m2"] = d.glazing_u
    d["eui_site_kwh_m2"] = d.cool_sens_kwh_m2 + d.heat_sens_kwh_m2
    d = d.assign(climate_key="minneapolis", design_set="specific", batch="main")
    return _D(data=d, order=["minneapolis"], training=["minneapolis"])


def check_direction_logic():
    assert C.check_directions(_direction_data())[0] == "PASS"
    assert C.check_directions(_direction_data(wrong_weak=True))[0] == "WARN", "significant wrong-sign weak effect"
    assert C.check_directions(_direction_data(break_cooling=True))[0] == "FAIL", "missing tier-1 effect"
    small = _direction_data(n=10)
    assert C.check_directions(small)[0] == "INFO"


def check_edge_designs_excluded():
    D = _direction_data()
    D["data"].loc[:9, "design_set"] = "edge"
    assert len(C._direction_frame(D)) == 190


def check_london_extrapolation_flagged():
    st, _, txt = C.check_holdout(_D(holdout_climates=["london"],
                                    training=[k for k in META.index if META.loc[k, "role"] == "training"]))
    row = [ln for ln in txt.splitlines() if ln.startswith("| annual_ghi")][0]
    assert "below" in row, row
    assert st == "INFO"


def check_gate():
    assert C.gate([("PASS", "", ""), ("INFO", "", "")]) == "GO"
    assert C.gate([("PASS", "", ""), ("WARN", "", "")]) == "REVIEW"
    assert C.gate([("WARN", "", ""), ("FAIL", "", "")]) == "NO-GO"


def check_end_to_end_on_pilot():
    batch = ROOT / "outputs" / "pilot"
    if not (batch / "run_log.csv").exists():
        return
    with tempfile.TemporaryDirectory() as t:
        copy = Path(t) / "pilot"
        shutil.copytree(batch, copy, ignore=shutil.ignore_patterns("report_figures", "*.md"))
        text, checks = C.build_report([("pilot", ROOT / "sampling" / "samples_pilot.csv", copy)], Path(t) / "rep")
        assert [c[1][0] for c in checks] == list("CARPDHKGMT"), [c[1] for c in checks]
        assert C.gate(checks) == "GO", [(c[0], c[1]) for c in checks]
        assert "**+0.68**" in text and "**+0.64**" in text, "D values must match the pilot report"
        assert len(list((Path(t) / "rep" / "campaign_figures").glob("*.png"))) == 3   # no anchors -> no fig 2


def check_accepted_exceptions():
    """A severe message listed (run_id, category) in accepted_exceptions does not FAIL R."""
    es = pd.DataFrame({"run_id": ["r1", "r2"], "terminated_fatal": [0, 0], "reported_severe": [1, 0],
                       "clean": [0, 1]})
    em = pd.DataFrame({"run_id": ["r1"], "level": ["Severe"], "category": ["convergence"], "benign": [0],
                       "signature": ["CheckWarmupConvergence ..."]})
    S = pd.DataFrame({"run_id": ["r1", "r2"], "climate_key": ["chicago", "chicago"]})
    log = pd.DataFrame({"run_id": ["r1", "r2"], "attempts": [1, 1]})
    D = _D(err_sum=es, err_msg=em, samples=S, log=log, order=["chicago"])
    assert C.check_reliability(D, accepted=set())[0] == "FAIL"
    st, _, txt = C.check_reliability(D, accepted={("r1", "convergence")})
    assert st == "PASS" and "accepted" in txt, (st, txt)
    assert C.check_reliability(D, accepted={("r1", "other")})[0] == "FAIL", "category must match"


ALL_CHECKS = [check_completeness_logic, check_anchor_logic, check_direction_logic, check_edge_designs_excluded,
              check_london_extrapolation_flagged, check_gate, check_accepted_exceptions,
              check_end_to_end_on_pilot]


def main():
    total = failed = 0
    print("=" * 72)
    print("CAMPAIGN REPORT (2.6) - CHECKS")
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
