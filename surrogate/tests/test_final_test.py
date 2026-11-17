"""
test_final_test.py - checks for surrogate/final_test.py and surrogate/timing.py (Sub-task 3.7).

Key checks: config block appended after the frozen 3.6 file and read by the code; the write-once rules
(result file, lock file, dirty git tree, flags); the model-input path for any site equals the 3.6 path
and refuses unknown or duplicate sites; London's indices are refused without allow_holdout; both
variants predict exactly what the fitted objects give; holdout shape checks; intervals are aligned
and equal Conformal.interval; the S3 / S4 / S7 / S8 rules and the Phase 4 decision on hand-made cases.

Run:  ./venv/bin/python surrogate/tests/test_final_test.py   (or all suites: surrogate/run_tests.py)
Never reads the holdout. Uses the saved 3.6 models (outputs/models) for the prediction checks.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from surrogate import climate_indices as CI
from surrogate import explain as X
from surrogate import features as FE
from surrogate import final_test as FT
from surrogate import timing as T
from surrogate import uncertainty as U
from surrogate.config import ALL_TARGETS, PRIMARY_TARGETS

_CACHE: dict = {}


def _train():
    if "df" not in _CACHE:
        from surrogate import cv
        _CACHE["df"] = cv.load_training()[0]
    return _CACHE["df"]


def _fm():
    if "fm" not in _CACHE:
        _CACHE["fm"] = X.final_models(dev=False)
    return _CACHE["fm"]


def _summary(skill, r2):
    return pd.DataFrame({"target": ALL_TARGETS, "design_skill": skill, "r2": r2})


# ----------------------------------------------------------------------------

def check_config_block_appended_and_read():
    import hashlib
    from surrogate.config import CONFIG_PATH
    text = CONFIG_PATH.read_text(encoding="utf-8")
    marker = "\n# ── Added 17 Nov 2026 (Sub-task 3.7"
    assert text.count(marker) == 1
    assert hashlib.md5(text[: text.index(marker)].encode("utf-8")).hexdigest() == "ee39251eee227e5f84b3c21ea2a644e1"
    assert FT.LEVEL == 0.90 and FT.BAND == (0.85, 0.95) and FT.LONDON == "london"
    assert FT._S3["design_skill_min"] == 0.88 and FT._S3["pooled_r2_max_drop"] == 0.03
    assert FT._S4["mape_max_pct"] == {"cooling_demand_kwh_m2": 10, "eui_kwh_m2": 10, "heating_demand_kwh_m2": 25}
    assert FT._S4["spearman_min"] == 0.90
    assert FT.ALT_FORMS == {"heating_demand_kwh_m2": ("mlp_physics", "ratio"), "eui_kwh_m2": ("mlp_physics", "decomposed")}
    assert T.CFG["single_calls"] == 500 and T.CFG["batch_sizes"] == [10000, 100000] and T.CFG["batch_repeats"] == 5
    assert T.CFG["s8"]["batch_size_for_gate"] == 100000 and T.CFG["torch_threads"] == 1


def check_write_once_rules():
    clean = {"commit": "abc", "modified_tracked": []}
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "final_test"
        FT.check_write_once(d, clean)                                   # empty: allowed
        for bad in ({"commit": None, "modified_tracked": None, "error": "x"},
                    {"commit": "abc", "modified_tracked": ["surrogate/final_test.py"]}):
            try:
                FT.check_write_once(d, bad)
                raise AssertionError("dirty / unknown git state accepted")
            except FT.FinalTestRefused:
                pass
        FT.write_lock(d, {"x": 1})
        try:
            FT.write_lock(d, {"x": 2})                                  # O_EXCL: never overwritten
            raise AssertionError("second lock written")
        except FileExistsError:
            pass
        for name in (FT.LOCK, FT.RESULT):
            if not (d / name).exists():
                (d / name).write_text("{}")
            try:
                FT.check_write_once(d, clean)
                raise AssertionError(f"{name} present but run allowed")
            except FT.FinalTestRefused:
                pass
    for dev, confirm in ((False, False), (True, True)):
        try:
            FT.stage_score(dev=dev, confirm=confirm, log=lambda m: None)
            raise AssertionError("score ran without exactly one of --dev / --confirm")
        except FT.FinalTestRefused:
            pass


def check_model_inputs_any_site():
    df = _train().sample(300, random_state=1)
    raw = df[["climate_key", *FE.RAW_INPUTS]]
    a, b = X.matrices(raw), FT.matrices_with(raw, FT.index_table_with())
    for k in ("C-G", "C-G+I"):
        assert list(a[k].columns) == list(b[k].columns) and np.array_equal(a[k].to_numpy(), b[k].to_numpy())
    fake = {c: 1.0 for c in CI.INDEX_COLUMNS}
    tab = FT.index_table_with({"newsite": fake})
    assert tab.shape == (11, 7) and tab.loc["newsite"].tolist() == [1.0] * 7
    for bad in ({"denver": fake},):
        try:
            FT.index_table_with(bad)
            raise AssertionError("training climate accepted as an extra site")
        except ValueError:
            pass
    try:
        FT.matrices_with(raw.assign(climate_key="nowhere"), FT.index_table_with())
        raise AssertionError("unknown site accepted")
    except KeyError:
        pass
    try:
        CI.indices_for_key("london")
        raise AssertionError("London indices computed without allow_holdout")
    except CI.HoldoutClimateError:
        pass


def check_variants_predict_fitted_objects():
    fm = _fm()
    raw = _train().sample(200, random_state=2)[["climate_key", *FE.RAW_INPUTS]]
    tab = FT.index_table_with()
    v = FT.predict_variants(fm, raw, tab)
    assert np.array_equal(v["primary"].to_numpy(), fm.predict(raw).to_numpy())       # = 3.6 path, exactly
    forms = fm.objects["mlp_physics"].predict_forms(FT.matrices_with(raw, tab)["C-G+I"])
    assert np.array_equal(v["all_mlp"]["heating_demand_kwh_m2"].to_numpy(),
                          forms[("heating_demand_kwh_m2", "ratio")].to_numpy())
    assert np.array_equal(v["all_mlp"]["eui_kwh_m2"].to_numpy(), forms[("eui_kwh_m2", "decomposed")].to_numpy())
    same = [t for t in ALL_TARGETS if t not in FT.ALT_FORMS]
    assert np.array_equal(v["all_mlp"][same].to_numpy(), v["primary"][same].to_numpy())
    assert list(v["primary"].index) == list(raw.index)


def check_holdout_shape_rules():
    df = _train()
    keys = sorted(df["climate_key"].unique())
    ind = pd.concat([df[df["climate_key"] == k].head(30) for k in keys]).assign(design_set="holdout")
    ind["design_id"] = [f"C{i:03d}" for i in range(1, len(ind) + 1)]
    lon = df[df["climate_key"] == "denver"].head(100).assign(design_set="anchor", climate_key="london")
    lon["design_id"] = [f"A{i:03d}" for i in range(1, len(lon) + 1)]
    ind_ok, lon_ok = FT.check_holdout_shape(pd.concat([ind, lon]))
    assert len(ind_ok) == 300 and len(lon_ok) == 100
    assert FT.check_training_climate_inputs(ind_ok) == 0.0
    dup = ind.copy()
    dup.iloc[1, dup.columns.get_loc("design_id")] = dup["design_id"].iloc[0]
    for broken in (pd.concat([ind.iloc[1:], lon]), pd.concat([ind, lon.iloc[1:]]),
                   pd.concat([ind, lon.assign(climate_key="paris")]), pd.concat([dup, lon])):
        try:
            FT.check_holdout_shape(broken)
            raise AssertionError("wrong holdout shape accepted")
        except AssertionError as e:
            if "accepted" in str(e):
                raise
    try:
        FT.check_training_climate_inputs(ind_ok.assign(hdd18=ind_ok["hdd18"] + 1))
        raise AssertionError("climate inputs off the training values accepted")
    except AssertionError as e:
        if "accepted" in str(e):
            raise


def check_intervals_aligned_and_equal_conformal():
    df = _train().sample(400, random_state=3).reset_index(drop=True)
    Xm = CI.feature_matrix(df).reset_index(drop=True)
    rng = np.random.default_rng(0)
    iv = {}
    for t in ALL_TARGETS:
        yhat = df[t].to_numpy() + rng.normal(0, 0.5, len(df))
        conf = U.Conformal(scale="lgbm", params={**U.SCALE_PARAMS, "n_estimators": 20})
        iv[t] = {"standard": conf.fit(Xm, yhat, df[t].to_numpy(), groups=df["design_id"].to_numpy())}
    pred = df[ALL_TARGETS] + 0.1
    lf = FT.long_frame(df, pred)
    out = FT.add_intervals(lf, Xm, iv, "standard")
    for t in ALL_TARGETS:
        g = out[out["target"] == t]
        lo, hi = iv[t]["standard"].interval(Xm, pred[t].to_numpy(), level=0.9)
        assert np.array_equal(g["lo_90"].to_numpy(), lo) and np.array_equal(g["hi_90"].to_numpy(), hi)
    try:
        FT.add_intervals(lf, Xm.iloc[:-1], iv, "standard")
        raise AssertionError("misaligned rows accepted")
    except AssertionError as e:
        if "accepted" in str(e):
            raise
    cov = FT.coverage(out, heat=["denver"])
    nh = cov.set_index("target").loc["heating_demand_kwh_m2", "n"]
    assert nh == int((df["climate_key"] == "denver").sum())               # heating: heating climates only
    assert cov.set_index("target").loc["cooling_demand_kwh_m2", "n"] == len(df)


def check_s3_rules():
    cv = _summary(0.95, 0.99)
    assert FT.evaluate_s3(_summary(0.95, 0.99), cv)["status"] == "PASS"
    assert FT.evaluate_s3(_summary(0.88, 0.96), cv)["status"] == "PASS"            # both limits inclusive
    assert FT.evaluate_s3(_summary(0.8799, 0.99), cv)["status"] == "FAIL"
    assert FT.evaluate_s3(_summary(0.95, 0.9599), cv)["status"] == "FAIL"          # drop 0.0301 > 0.03
    h = _summary(0.95, 0.99)
    h.loc[h["target"] == "peak_cooling_w_m2", "design_skill"] = 0.5                # secondary: flag only
    r = FT.evaluate_s3(h, cv)
    assert r["status"] == "PASS" and r["targets"]["peak_cooling_w_m2"]["status"] == "flag"


def check_s4_s7_rules_and_decision():
    lon = pd.DataFrame({"target": ALL_TARGETS, "mape_pct": 5.0, "spearman": 0.95, "bias_pct": 1.0})
    assert FT.evaluate_s4(lon)["status"] == "PASS"
    for t, col, v in (("cooling_demand_kwh_m2", "mape_pct", 10.01), ("heating_demand_kwh_m2", "mape_pct", 25.01),
                      ("eui_kwh_m2", "spearman", 0.899)):
        x = lon.copy()
        x.loc[x["target"] == t, col] = v
        assert FT.evaluate_s4(x)["status"] == "WARN"
    x = lon.copy()
    x.loc[x["target"] == "heating_demand_kwh_m2", "mape_pct"] = 24.9                # heating limit is 25 %
    x.loc[x["target"] == "peak_heating_w_m2", "mape_pct"] = 99.0                    # secondary: not in S4
    assert FT.evaluate_s4(x)["status"] == "PASS"
    cov = pd.DataFrame({"target": ALL_TARGETS, "coverage_90": 0.90, "n": 300, "mean_width_90": 1.0})
    assert FT.evaluate_s7(cov)["status"] == "PASS"
    for v in (0.8499, 0.9501):
        c = cov.copy()
        c.loc[c["target"] == "zone_cooling_west_kwh_m2", "coverage_90"] = v
        assert FT.evaluate_s7(c)["status"] == "WARN"
    assert FT.phase4_decision({"S1": "PASS", "S6": "OK_after_adjudication"}) == "GO"
    assert FT.phase4_decision({"S1": "PASS", "S7": "WARN"}) == "REVIEW"
    assert FT.phase4_decision({"S1": "FAIL", "S7": "WARN"}) == "NO-GO"
    assert FT.phase4_decision({"S1": "PASS", "S8": "PENDING"}) == "PENDING"


def check_timing_rules_and_helpers():
    assert T.evaluate_s8(49.9, 10000.0)["status"] == "PASS"
    assert T.evaluate_s8(50.0, 20000.0)["status"] == "FAIL"
    assert T.evaluate_s8(10.0, 9999.0)["status"] == "FAIL"
    pool = _train()[["climate_key", *FE.RAW_INPUTS]].head(50)
    r = T.time_single(lambda row: len(row), pool, calls=5, warmup=2)
    assert r["calls"] == 5 and r["median_ms"] >= 0 and r["p95_ms"] >= r["median_ms"] - 1e-9
    b = T.time_batch(lambda batch: batch, pool, size=120, repeats=2)
    assert b["designs"] == 120 and b["designs_per_s"] > 0 and len(b["all_s"]) == 2


ALL_CHECKS = [check_config_block_appended_and_read, check_write_once_rules, check_model_inputs_any_site,
              check_variants_predict_fitted_objects, check_holdout_shape_rules,
              check_intervals_aligned_and_equal_conformal, check_s3_rules, check_s4_s7_rules_and_decision,
              check_timing_rules_and_helpers]


def main():
    total = failed = 0
    print("=" * 72)
    print("SURROGATE FINAL HOLDOUT TEST AND TIMING (3.7) - CHECKS")
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
