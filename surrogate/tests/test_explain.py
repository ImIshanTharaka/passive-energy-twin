"""
test_explain.py - checks for surrogate/explain.py (Sub-task 3.6 step 2).

Key checks: config block appended after the frozen 3.5 file and read by the code; one fitted object per
pick; the raw-input path (raw_frame + matrices) equals the 3.4 feature matrix and refuses London; final
objects use exactly the CV settings; fitted models predict through the raw path exactly as the objects
do, and save -> load is identical with a SHA-256 check; OAT runs read correctly with the duplicate
orientation pair detected; the S5 counting rules and the S6 reversal size on hand-made cases; sweep
grid and start designs; zone areas reproduce EnergyPlus; random designs inside the bounds; SHAP climate
block (every evaluated climate is a whole training climate), additivity, and group shares.

Run:  ./venv/bin/python surrogate/tests/test_explain.py   (or all suites: surrogate/run_tests.py)
Small models on the 25 % dev sample (seconds), never the official files.
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

_CACHE: dict = {}


def _data():
    if "df" not in _CACHE:
        _CACHE["df"], _CACHE["heat"] = X.load(dev=True)
    return _CACHE["df"], _CACHE["heat"]


def _fitted():
    if "fm" not in _CACHE:
        df, _ = _data()
        _CACHE["fm"], _CACHE["info"] = X.fit_final(df, dev=True, log=lambda m: None)
    return _CACHE["fm"]


# ----------------------------------------------------------------------------

def check_config_block_appended_and_read():
    import hashlib
    from surrogate.config import CONFIG_PATH
    text = CONFIG_PATH.read_text(encoding="utf-8")
    marker = "\n# ── Added 14 Nov 2026 (Sub-task 3.6"
    assert text.count(marker) == 1
    assert hashlib.md5(text[: text.index(marker)].encode("utf-8")).hexdigest() == "9b02ad7095d9350c48852179f4eb3aa5"
    assert X.OAT_REL == 0.02 and X.MIN_SIGN_SHARE == 0.90 and X.MAX_REVERSING == 0.01
    assert (X.SWEEP_N, X.SWEEP_POINTS, X.SWEEP_SEED) == (200, 21, 42)
    assert X.MAX_EVALS == 500 and X.SHAP_SEED == 42 and X.PERIM == FE.PERIM_DEPTH and X.ADD_TOL == 1e-4
    assert X.OAT_CLIMATES == ("colombo", "minneapolis") and len(X.EXPECTATIONS) == 7
    assert set(X.STRONG_SET) == {"glazing_shgc", "glazing_u", "wwr_north", "wwr_east", "wwr_south", "wwr_west"}
    assert X.GUARD["max_hours_per_object_view"] == 2.0 and X.GUARD["fallback"]["rows_per_climate"] == 50


def check_one_object_per_pick():
    from surrogate.config import ALL_TARGETS
    m = X.TARGET_OBJECT
    assert set(m) == set(ALL_TARGETS)
    assert m[X.HEAT] == "gpr_heating" and m[X.EUI] == "xgb_eui" and m[X.PEAK_HEAT] == "mlp_physics"
    assert all(m[t] == "mlp_cg" for t in ALL_TARGETS if t not in (X.HEAT, X.EUI, X.PEAK_HEAT))
    assert set(m.values()) == set(X.OBJECTS)


def check_raw_path_equals_feature_matrix_and_refuses_london():
    df, _ = _data()
    raw = X.raw_frame(df[FE.DESIGN_INPUTS], df["climate_key"])
    assert np.allclose(raw[FE.CLIMATE_INPUTS].to_numpy(), df[FE.CLIMATE_INPUTS].to_numpy(), rtol=0, atol=1e-12)
    M = X.matrices(raw)
    ref = CI.feature_matrix(df).reset_index(drop=True)
    assert list(M["C-G+I"].columns) == list(ref.columns) and M["C-G"].shape[1] == 63 and M["C-G+I"].shape[1] == 70
    assert np.allclose(M["C-G+I"].to_numpy(), ref.to_numpy(), rtol=0, atol=1e-12)
    try:
        X.raw_frame(df[FE.DESIGN_INPUTS].head(2), "london")
    except KeyError:
        pass
    else:
        raise AssertionError("London accepted outside final_test.py")
    assert "london" not in X.climate_table().index and len(X.climate_table()) == 10


def check_official_settings_equal_cv_settings():
    """The final objects use exactly the frozen 3.3 / 3.4 settings (official, not dev)."""
    from surrogate import benchmark as B
    from surrogate import climate_gen as CG
    from surrogate.config import ALL_TARGETS, PRIMARY_TARGETS
    from surrogate.tune import load_best
    gpr = X.object_params("gpr_heating", dev=False)
    assert gpr["top_k"] == 15 and gpr == CG.params_for("gpr", [X.HEAT], False, 15)[X.HEAT]
    xgb = X.object_params("xgb_eui", dev=False)
    best = load_best(False)
    for t in (X.HEAT, X.COOL, X.EUI):
        assert xgb[t] == best[f"xgb__{t}"]["params"], t
    mlp = X.object_params("mlp_cg", dev=False)
    assert mlp == best["mlp__all"]["params"] == B.tuned_params_for("mlp", ALL_TARGETS, False, feature_set="C-G")[None]
    assert X.object_params("mlp_physics", dev=False) == CG.params_for("mlp", PRIMARY_TARGETS, False, None)[None] == mlp


def check_fitted_models_raw_path_and_persistence():
    df, _ = _data()
    fm = _fitted()
    rows = df.sample(60, random_state=1)
    raw = rows[["climate_key", *FE.RAW_INPUTS]]
    pred = fm.predict(raw)
    Xm = X.matrices(raw)
    direct = {X.HEAT: fm.objects["gpr_heating"].predict(Xm["C-G+I"]),
              X.EUI: fm.objects["xgb_eui"].predict(Xm["C-G+I"]),
              X.PEAK_HEAT: fm.objects["mlp_physics"].predict_forms(Xm["C-G+I"])[(X.PEAK_HEAT, "direct")].to_numpy()}
    cg = np.asarray(fm.objects["mlp_cg"].predict(Xm["C-G"]))
    from surrogate.config import ALL_TARGETS
    for j, t in enumerate(ALL_TARGETS):
        exp = direct.get(t, cg[:, j])
        assert np.allclose(pred[t].to_numpy(), exp, rtol=0, atol=1e-10), t
    assert list(pred.index) == list(raw.index)
    old = X.CHUNK
    try:
        X.CHUNK = 7                                             # chunking changes nothing beyond float32 rounding
        assert np.allclose(fm.predict(raw).to_numpy(), pred.to_numpy(), rtol=1e-5, atol=1e-6)
    finally:
        X.CHUNK = old
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        fm.save(d, _CACHE["info"])
        back = X.FinalModels.load(d)
        assert np.array_equal(back.predict(raw).to_numpy(), pred.to_numpy())
        with open(d / "xgb_eui.pkl", "ab") as fh:
            fh.write(b"x")
        try:
            X.FinalModels.load(d)
        except RuntimeError:
            pass
        else:
            raise AssertionError("changed model file not detected")


def check_oat_runs_read_and_duplicate_detected():
    from runner.output_parser import TARGETS
    oat = X.load_oat()
    assert len(oat) == 86 and set(oat["climate_key"]) == {"colombo", "minneapolis"}
    assert (oat["design_id"] == "BASE").sum() == 2
    params = oat.loc[oat["design_id"] != "BASE", "parameter"].unique()
    assert sorted(params) == sorted(FE.DESIGN_INPUTS)
    res = pd.read_csv(X.OAT_DIR / "results.csv").set_index("run_id")
    for t, src in TARGETS.items():
        assert np.allclose(oat.set_index("run_id")[t], res.loc[oat["run_id"], src].to_numpy()), t
    dup = []
    for c in X.OAT_CLIMATES:
        g = oat[oat["climate_key"] == c].set_index("design_id")
        base = g.loc["BASE", FE.DESIGN_INPUTS].to_numpy(float)
        dup += [i for i in g.index if i != "BASE" and np.allclose(g.loc[i, FE.DESIGN_INPUTS].to_numpy(float), base)]
    assert sorted(set(dup)) == ["orientation__hi"]


def check_s5_counting_rules():
    rows = []
    def add(p, b, c, t, ep_b, d_ep, d_sur, dup=False):
        rows.append({"parameter": p, "bound": b, "climate_key": c, "target": t, "duplicate_of_base": dup,
                     "ep_base": ep_b, "ep_run": ep_b + d_ep, "d_ep": d_ep, "sur_base": ep_b,
                     "sur_run": ep_b + d_sur, "d_sur": d_sur})
    heat = ["minneapolis"]
    add("glazing_shgc", "hi", "colombo", X.COOL, 100, 30, 25)            # strong, counted, ok
    add("glazing_shgc", "hi", "colombo", X.HEAT, 0.4, 0.3, -0.1)         # heating in hot climate: not scored
    add("glazing_u", "hi", "minneapolis", X.HEAT, 10, 0.1, -0.2)         # 1 % <= 2 %: not counted
    add("wall_r_value", "lo", "minneapolis", X.HEAT, 10, 3, 2)           # other, ok
    add("orientation", "hi", "colombo", X.COOL, 100, 0, 0, dup=True)     # duplicate: dropped
    t = X.classify_oat(pd.DataFrame(rows), heat)
    assert list(t["scored"]) == [True, False, True, True, False]
    assert list(t["counted"]) == [True, False, False, True, False]
    assert list(t["strong"]) == [True, True, True, False, False]
    r = X.s5(t)
    assert r["status"] == "PASS" and r["n_counted"] == 2 and r["n_dropped_duplicate_pairs"] == 1
    add("wwr_south", "hi", "colombo", "zone_cooling_south_kwh_m2", 100, 50, -1)   # strong miss -> FAIL
    r = X.s5(X.classify_oat(pd.DataFrame(rows), heat))
    assert r["status"] == "FAIL" and not r["strong_all_sign_correct"] and len(r["strong_misses"]) == 1
    rows.pop()
    for k in range(9):                                                   # 2 ok + 9 wrong others -> share < 90 %
        add("floor_area", f"x{k}", "minneapolis", X.EUI, 100, 5, -5)
    r = X.s5(X.classify_oat(pd.DataFrame(rows), heat))
    assert r["strong_all_sign_correct"] and not r["share_ok"] and r["status"] == "FAIL"


def check_reversal_size():
    Y = np.array([[1, 2, 3, 4], [1, 3, 2, 4], [4, 3, 2, 1], [1, 5, 0, 6]], dtype=float)
    assert np.allclose(X.reversal_size(Y, "up"), [0, 1, 3, 5])
    assert np.allclose(X.reversal_size(Y, "down"), [3, 3, 0, 6])
    assert np.allclose(X.reversal_size(Y[:, ::-1], "down"), X.reversal_size(Y, "up"))


def check_sweep_grid_and_starts():
    from surrogate.uncertainty import design_bounds
    df, _ = _data()
    b = design_bounds()
    for inp in ("glazing_shgc", "glazing_u", "wwr_west"):
        g = X.sweep_grid(inp)
        assert len(g) == 21 and g[0] == b.loc[inp, "min"] and g[-1] == b.loc[inp, "max"] and (np.diff(g) > 0).all()
    s1, s2 = X.sweep_starts(df, 5), X.sweep_starts(df, 5)
    assert s1["run_id"].tolist() == s2["run_id"].tolist() and len(s1) == 5 * df["climate_key"].nunique()
    assert (s1.groupby("climate_key").size() == 5).all() and s1["run_id"].isin(df["run_id"]).all()
    fm = _fitted()
    P = X.sweep_predictions(fm, s1.head(3), "glazing_shgc", [X.COOL])
    assert P[X.COOL].shape == (3, 21)
    rep = s1.head(3).copy()
    rep["glazing_shgc"] = X.sweep_grid("glazing_shgc")[4]
    assert np.allclose(P[X.COOL][:, 4], fm.predict_designs(rep, rep["climate_key"], [X.COOL])[X.COOL].to_numpy())


def check_zone_areas_reproduce_energyplus():
    from surrogate import cv
    df, _ = cv.load_training()
    gap = X.zone_gap(df[X.COOL], df, df)
    assert np.abs(gap).max() < 1e-4, np.abs(gap).max()
    a = X.zone_areas(df)
    assert np.allclose(a.sum(axis=1), df["floor_area"], rtol=1e-12)


def check_lhs_designs_inside_bounds():
    from surrogate.uncertainty import DomainChecker
    new = X.lhs_designs(20, ["colombo", "denver"], 42)
    assert len(new) == 40 and set(new["n_floors"].unique()) <= {1, 2, 3, 4, 5}
    flags = DomainChecker.from_training().check(new[FE.DESIGN_INPUTS], X.climate_table().loc["colombo",
                                                                                                  FE.CLIMATE_INPUTS].to_dict())
    assert flags["design_in_domain"].all()


def check_shap_climate_block_additivity_and_shares():
    df, heat = _data()
    fm = _fitted()
    climates = sorted(df["climate_key"].unique())
    f = X.OverallModel(fm, "xgb_eui", climates)
    seen = []
    real_frame = f.frame

    def spy(Z):
        fr = real_frame(Z)
        tab = X.climate_table()
        assert np.allclose(fr[FE.CLIMATE_INPUTS].to_numpy(), tab.loc[fr["climate_key"], FE.CLIMATE_INPUTS].to_numpy())
        seen.extend(fr["climate_key"].unique())
        return fm.predict_object("xgb_eui", fr).to_numpy()
    rows = X.sample_rows(df, climates, 1, 42).head(3)
    bg = X.sample_background(df, climates, 12, 42)
    vals, base = X.permutation_shap(spy, f.encode(rows), f.encode(bg), 60)
    pred = f(f.encode(rows))
    assert vals.shape == (3, 22, 1) and X.check_additivity(vals, base, pred) < 1e-9
    assert set(seen) <= set(climates) and len(set(seen)) > 1
    for bad in (np.array([[0.5]]), np.array([[len(climates)]])):
        try:
            f.codes_to_keys(bad.ravel())
        except AssertionError:
            pass
        else:
            raise AssertionError("climate player accepted a non-climate code")
    fr, info = X.explain_object(fm, "gpr_heating", "per_climate", df, heat,
                                {"rows_per_climate": 1, "background_overall": 8, "background_per_climate": 6,
                                 "max_evals": 50})
    assert set(fr["climate_key"]) <= set(heat) and info["additivity_max_rel_gap"] < X.ADD_TOL
    sh = X.group_shares(fr, list(FE.DESIGN_INPUTS), ["target"])
    groups = [c for c in sh.columns if c != "target"]
    assert np.allclose(sh[groups].sum(axis=1), 1.0) and "climate" not in groups and len(groups) == 6


ALL_CHECKS = [check_config_block_appended_and_read, check_one_object_per_pick,
              check_raw_path_equals_feature_matrix_and_refuses_london, check_official_settings_equal_cv_settings,
              check_fitted_models_raw_path_and_persistence, check_oat_runs_read_and_duplicate_detected,
              check_s5_counting_rules, check_reversal_size, check_sweep_grid_and_starts,
              check_zone_areas_reproduce_energyplus, check_lhs_designs_inside_bounds,
              check_shap_climate_block_additivity_and_shares]


def main():
    total = failed = 0
    print("=" * 72)
    print("SURROGATE INTERPRETABILITY AND PHYSICS CONSISTENCY (3.6) - CHECKS")
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
