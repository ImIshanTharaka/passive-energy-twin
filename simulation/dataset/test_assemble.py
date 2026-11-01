"""
test_assemble.py - Checks for dataset/assemble.py (Sub-task 2.7).

Join and check logic on small synthetic sources (every FAIL path is provoked once),
the Phase 3 loader's tamper detection, plus an end-to-end build of the real dataset
when outputs/main and outputs/holdout exist (written to a temp folder, never to
outputs/dataset).

Run:  python dataset/test_assemble.py
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset import assemble as A        # noqa: E402

INPUTS = ["floor_area", "n_floors", "wwr_south"]
BOUNDS = {"floor_area": (1500.0, 10000.0), "n_floors": (1.0, 5.0), "wwr_south": (0.1, 0.8)}
CF = list(A.CONTINUOUS_FEATURES)


def _climate():
    rows = []
    for i, (k, role, z) in enumerate([("colombo", "training", "0A"), ("minneapolis", "training", "6A"),
                                      ("london", "holdout", "4A")]):
        rows.append({"key": k, "role": role, "ashrae_zone": z, **{f: float(i * 10 + j) for j, f in enumerate(CF)}})
    return pd.DataFrame(rows)


def _results(samples, heat=None):
    n = len(samples)
    r = pd.DataFrame({"run_id": samples.run_id.values})
    for c in A.RESULT_COLUMNS[1:]:
        r[c] = 1.0
    r["n_floors"] = samples.n_floors.values
    r["n_zones"] = 5 * samples.n_floors.values
    r["cond_area_m2"] = samples.floor_area.values * samples.n_floors.values
    hot = samples.climate_key.isin(["colombo"]).values
    r["heat_sens_kwh_m2"] = np.where(hot, 0.1, 40.0) if heat is None else heat
    r["cool_sens_kwh_m2"] = np.where(hot, 80.0, 20.0)
    r["ep_version"] = "24.1.0"
    for c in [c for c in A.RESULT_COLUMNS if c.endswith("_at")]:
        r[c] = "01-02 07"
    return r.reset_index(drop=True) if n else r


def _samples(batch, design_set, ids, climates):
    rows = []
    for i, (d, k) in enumerate((d, k) for d in ids for k in climates):
        rows.append({"run_id": f"{batch}_{d}_{k}", "batch": batch, "design_set": design_set, "design_id": d,
                     "climate_key": k, "floor_area": 2000.0 + 100 * i, "n_floors": 1 + i % 5,
                     "wwr_south": 0.3, "glazing_vt": 0.5})
    return pd.DataFrame(rows)


def _split(name, samples, sha="s"):
    log = pd.DataFrame({"run_id": samples.run_id, "status": "ok"})
    err = pd.DataFrame({"run_id": samples.run_id, "clean": 1, "reported_warnings": 0, "reported_severe": 0,
                        "terminated_fatal": 0, "categories": np.nan})
    sf, bd, _ = A.SPLITS[name]
    return A.SplitSource(name=name, samples_file=sf, batch_dir=bd, samples=samples, results=_results(samples),
                         run_log=log, err_summary=err, run_meta=[{"samples_sha256": sha}], samples_sha=sha,
                         expected_samples_sha=sha, results_fresh=True)


def _src():
    tr = _samples("main", "anchor", ["A001", "A002", "A003", "A004"], ["colombo", "minneapolis"])
    hid = _samples("holdout_id", "holdout", ["C001", "C002"], ["colombo", "minneapolis"])
    ood = _samples("holdout_ood", "anchor", ["A001", "A002"], ["london"])
    ho = pd.concat([hid, ood], ignore_index=True)
    return A.Sources(splits={"training": _split("training", tr), "holdout": _split("holdout", ho, "h")},
                     climate=_climate(), inputs=INPUTS, bounds=BOUNDS,
                     input_units={n: "u" for n in INPUTS}, integer_inputs={"n_floors"}, accepted={},
                     climate_sha="c", expected_climate_sha="c")


def _status(b, cid):
    return next(c for c in b.checks if c.id == cid).status


def check_column_order():
    cols = A.column_order(INPUTS)
    assert cols[:6] == A.ID_COLUMNS + ["ashrae_zone"]
    assert cols[6:9] == INPUTS and cols[9:16] == CF and cols[16:] == A.TARGET_NAMES
    assert len(A.TARGET_NAMES) == 10 and "glazing_vt" not in cols


def check_clean_build_is_go():
    b = A.build(_src())
    bad = [c for c in b.checks if c.status in (A.FAIL, A.WARN)]
    assert b.gate == "GO", bad
    assert len(b.data["training"]) == 8 and len(b.data["holdout"]) == 6
    assert len(b.diagnostics) == 14 and "sim_n_floors" in b.diagnostics.columns


def check_join_values():
    src = _src()
    b = A.build(src)
    tr = b.data["training"].set_index("run_id")
    row = tr.loc["main_A001_minneapolis"]
    assert row.heating_demand_kwh_m2 == 40.0 and row.ashrae_zone == "6A"
    assert row.hdd18 == 10.0 and row.mean_wind == 16.0, "climate features joined on climate_key"
    assert "glazing_vt" not in tr.columns and "heat_sens_kwh_m2" not in tr.columns


def check_missing_result_fails():
    src = _src()
    src.splits["training"].results = src.splits["training"].results.iloc[1:]
    b = A.build(src)
    assert _status(b, "C1") == A.FAIL and _status(b, "S1") == A.FAIL and b.gate == "NO-GO"


def check_extra_and_duplicate_results_fail():
    src = _src()
    r = src.splits["training"].results
    extra = r.iloc[[0]].assign(run_id="main_Z999_colombo")
    src.splits["training"].results = pd.concat([r, r.iloc[[1]], extra], ignore_index=True)
    c = A.check_completeness(src)
    assert c.status == A.FAIL and "duplicate" in c.detail and "not in samples" in c.detail, c.detail


def check_failed_run_fails():
    src = _src()
    src.splits["holdout"].run_log.loc[0, "status"] = "failed"
    assert A.check_completeness(src).status == A.FAIL


def check_provenance_fails():
    src = _src()
    src.splits["training"].run_meta.append({"samples_sha256": "other"})
    assert A.check_provenance(src).status == A.FAIL
    src = _src()
    src.splits["holdout"].results_fresh = False
    assert "older" in A.check_provenance(src).detail
    src = _src()
    src.climate_sha = "changed"
    assert A.check_provenance(src).status == A.FAIL


def check_leakage_fails():
    src = _src()
    s = src.splits["training"].samples
    s.loc[0, "climate_key"] = "london"
    src.splits["training"].samples = s
    src.splits["training"].results = _results(s)
    b = A.build(src)
    assert _status(b, "L1") == A.FAIL
    src = _src()
    s = src.splits["holdout"].samples
    s.loc[0, "design_id"] = "A003"                    # a training design inside holdout_id
    assert A.check_split_isolation(src, A.build(src).data).status == A.FAIL


def check_input_bounds_fail():
    src = _src()
    src.splits["training"].samples.loc[2, "wwr_south"] = 0.95
    assert _status(A.build(src), "I1") == A.FAIL
    src = _src()
    src.splits["training"].samples["n_floors"] = src.splits["training"].samples["n_floors"].astype(float)
    src.splits["training"].samples.loc[0, "n_floors"] = 2.5
    assert _status(A.build(src), "I1") == A.FAIL


def check_geometry_fails():
    src = _src()
    src.splits["training"].results.loc[3, "cond_area_m2"] *= 1.01
    assert _status(A.build(src), "G1") == A.FAIL
    src = _src()
    src.splits["holdout"].results.loc[0, "n_floors"] += 1
    assert _status(A.build(src), "G1") == A.FAIL


def check_bad_targets_fail():
    src = _src()
    src.splits["training"].results.loc[0, "cool_sens_kwh_m2"] = -0.5
    assert _status(A.build(src), "T1") == A.FAIL
    src = _src()
    src.splits["training"].results.loc[0, "peak_cool_steady_w_m2"] = np.nan
    assert _status(A.build(src), "S1") == A.FAIL


def check_simulation_quality():
    src = _src()
    src.splits["training"].err_summary.loc[0, ["clean", "reported_severe"]] = [0, 1]
    assert A.check_simulation_quality(src).status == A.FAIL
    src.accepted = {src.splits["training"].err_summary.run_id[0]: "convergence"}
    c = A.check_simulation_quality(src)
    assert c.status == A.PASS and "exceptions kept" in c.detail, c.detail


def check_physics_warns():
    src = _src()
    tr = src.splits["training"]
    tr.results = _results(tr.samples, heat=np.where(tr.samples.climate_key == "colombo", 5.0, 40.0))
    b = A.build(src)
    assert _status(b, "T2") == A.WARN and b.gate == "REVIEW"


def check_outlier_flags():
    df = pd.DataFrame({"climate_key": ["x"] * 30, **{t: [1.0] * 30 for t in A.TARGET_NAMES}})
    df.loc[:28, "eui_kwh_m2"] = np.linspace(99, 101, 29)
    df.loc[29, "eui_kwh_m2"] = 1000.0
    f = A.outlier_flags(df)
    assert f[29] == "eui_kwh_m2" and (f[:29] == "").all(), f.tolist()


def check_gate_logic():
    C = A.Check
    assert A.gate([C("a", "", A.PASS, ""), C("b", "", A.INFO, "")]) == "GO"
    assert A.gate([C("a", "", A.PASS, ""), C("b", "", A.WARN, "")]) == "REVIEW"
    assert A.gate([C("a", "", A.WARN, ""), C("b", "", A.FAIL, "")]) == "NO-GO"


def check_write_and_loader_detects_tampering():
    src = _src()
    b = A.build(src)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        m = A.write_outputs(src, b, out)
        assert m["gate"] == "GO" and m["columns"]["group"] == "design_id"
        assert len(m["columns"]["inputs"]) == len(INPUTS) + 7
        df, cols = A.load_dataset("training", out)
        assert list(df.columns) == cols["all"] and df.design_id.iloc[0] == "A001"
        pd.testing.assert_frame_equal(df, b.data["training"], check_dtype=False)
        sha1 = m["files"]["training"]["sha256"]
        A.write_outputs(src, b, out)
        assert json.loads((out / A.MANIFEST_NAME).read_text())["files"]["training"]["sha256"] == sha1, \
            "CSV output must be byte-deterministic (DS6)"
        p = out / "dataset_training.csv"
        p.write_text(p.read_text().replace("A001", "A00X", 1))
        try:
            A.load_dataset("training", out)
            raise AssertionError("tampered file loaded")
        except A.AssemblyError:
            pass
        assert A.verify(out) == 1


def check_end_to_end_on_campaign():
    if not ((ROOT / "outputs/main/results.csv").exists() and (ROOT / "outputs/holdout/results.csv").exists()):
        print("        (skipped: campaign outputs not present)")
        return
    src = A.load_sources()
    b = A.build(src)
    bad = [f"{c.id} {c.detail}" for c in b.checks if c.status == A.FAIL]
    assert not bad, bad
    tr, ho = b.data["training"], b.data["holdout"]
    assert len(tr) == 3000 and len(ho) == 400
    assert tr.shape[1] == 5 + 1 + 21 + 7 + 10
    assert tr.groupby("climate_key").size().eq(300).all() and tr.climate_key.nunique() == 10
    assert (ho.batch == "holdout_ood").sum() == 100 and (ho.climate_key == "london").sum() == 100
    # values are the source values, untouched (DS6)
    r = src.splits["training"].results.set_index("run_id")
    s = src.splits["training"].samples.set_index("run_id")
    t = tr.set_index("run_id")
    for rid in ["main_A001_atlanta", tr.run_id.iloc[-1], "main_B0573_chicago"]:
        for k, v in A.TARGETS.items():
            assert t.loc[rid, k] == r.loc[rid, v], (rid, k)
        for n in src.inputs:
            assert t.loc[rid, n] == s.loc[rid, n], (rid, n)
    with tempfile.TemporaryDirectory() as tmp:
        A.write_outputs(src, b, Path(tmp))
        df, cols = A.load_dataset("holdout", Path(tmp))
        assert len(df) == 400 and len(cols["inputs"]) == 28 and len(cols["targets"]) == 10


ALL_CHECKS = [check_column_order, check_clean_build_is_go, check_join_values, check_missing_result_fails,
              check_extra_and_duplicate_results_fail, check_failed_run_fails, check_provenance_fails,
              check_leakage_fails, check_input_bounds_fail, check_geometry_fails, check_bad_targets_fail,
              check_simulation_quality, check_physics_warns, check_outlier_flags, check_gate_logic,
              check_write_and_loader_detects_tampering, check_end_to_end_on_campaign]


def main():
    total = failed = 0
    print("=" * 72)
    print("DATASET ASSEMBLY (2.7) - CHECKS")
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
