"""
test_oat.py - Checks for runner/oat.py (Sub-task 2.5, step 7.4).

Run:  python runner/test_oat.py
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner import oat as O                       # noqa: E402
from runner import output_parser as OP            # noqa: E402
from sampling.transform import load_design_space  # noqa: E402

SPACE = load_design_space()


def check_design_count_and_one_change_each():
    ds = O.designs(SPACE)
    assert len(ds) == 1 + 2 * len(SPACE.params) == 43
    base = ds[0][2]
    for did, varied, params in ds[1:]:
        changed = {k for k in SPACE.sampled_names if params[k] != base[k]}
        assert changed <= {varied}, (did, changed)
        assert params["glazing_vt"] == params["glazing_shgc"], "VT stays derived (S2)"


def check_levels_are_bounds_except_orientation():
    ds = {d: p for d, _, p in O.designs(SPACE)}
    for p in SPACE.params:
        lo, hi = ds[f"{p.name}__lo"][p.name], ds[f"{p.name}__hi"][p.name]
        if p.name == "orientation":
            assert (lo, hi) == (0.0, 90.0)
        else:
            assert (lo, hi) == (p.lo, p.hi), (p.name, lo, hi)


def check_expectations_reference_real_columns():
    for name, items in O.EXPECT.items():
        assert name in SPACE.sampled_names, name
        for col, climate, sign in items:
            assert col in OP.COLUMNS and climate in O.CLIMATES and sign in (-1, 1), (name, col)


def check_analysis_on_synthetic_results():
    """Fake results where every parameter raises cooling by 1% and nothing else:
    all respond; expectations with sign -1 on cooling must be flagged."""
    with tempfile.TemporaryDirectory() as t:
        b = Path(t)
        rows = []
        for climate in O.CLIMATES:
            for did, varied, _ in O.designs(SPACE):
                r = {c: 10.0 for c in OP.COLUMNS if c.endswith(("_kwh_m2", "_w_m2"))}
                if did.endswith("__hi"):
                    for c in r:
                        if "cool" in c:
                            r[c] = 10.1
                rows.append({"run_id": f"oat_{did}_{climate}", **r})
        with (b / "results.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        (b / "run_log.csv").write_text("run_id,status\n")
        import os
        os.utime(b / "results.csv", None)
        out, s = O.analyse(b)
        assert s["responds"] == s["n_params"] == 21
        bad = {(r["param"], e[0]) for r in out for e in r["expect"] if not e[4]}
        assert ("glazing_u", "heat_sens_kwh_m2") in bad          # heat did not change -> not confirmed
        assert ("glazing_shgc", "cool_sens_kwh_m2") not in bad   # cooling rose, as expected


def check_samples_file_round_trip():
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "s.csv"
        orig = O.SAMPLES
        try:
            O.SAMPLES = p
            n = O.build(p)
        finally:
            O.SAMPLES = orig
        assert n == 86
        with p.open() as fh:
            rdr = csv.DictReader(fh)
            assert rdr.fieldnames == list(O.ID_COLUMNS) + SPACE.idf_names
            ids = [r["run_id"] for r in rdr]
        assert len(ids) == len(set(ids)) == 86


ALL_CHECKS = [check_design_count_and_one_change_each, check_levels_are_bounds_except_orientation,
              check_expectations_reference_real_columns, check_analysis_on_synthetic_results,
              check_samples_file_round_trip]


def main():
    total = failed = 0
    print("=" * 72)
    print("OAT (2.5.7.4) - CHECKS")
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
