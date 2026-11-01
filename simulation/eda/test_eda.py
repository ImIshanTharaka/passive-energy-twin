"""
test_eda.py - Checks for eda/eda_lib.py (Sub-task 2.8).

Each calculation is checked on synthetic data with a known answer (facade azimuth,
design skill, fold isolation, physics sign detection, gate logic, holdout guard),
then the library is run end-to-end on the real dataset when outputs/dataset exists
(read-only; fast RF settings).

Run:  python eda/test_eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eda import eda_lib as L          # noqa: E402

RNG_SEED = 7
SMALL_RF = dict(n_estimators=30, min_samples_leaf=2, max_features=1 / 3, n_jobs=-1, random_state=0)


def _frame(n_designs=60, climates=("hot", "mild", "cold"), anchors=20, seed=RNG_SEED):
    """Synthetic training frame: `anchors` designs appear in every climate, the rest once."""
    rng = np.random.default_rng(seed)
    rows = []
    lat = {"hot": 10.0, "mild": 35.0, "cold": 45.0}
    clim_effect = {"hot": 100.0, "mild": 60.0, "cold": 40.0}
    specific = iter(range(anchors, n_designs * len(climates)))
    for c in climates:
        ids = [f"A{i:03d}" for i in range(anchors)] + [f"B{next(specific):03d}" for _ in range(n_designs - anchors)]
        for d in ids:
            rows.append({"run_id": f"main_{d}_{c}", "batch": "main", "design_id": d, "climate_key": c,
                         "latitude": lat.get(c, 30.0), "clim": clim_effect.get(c, 50.0)})
    df = pd.DataFrame(rows)
    design_vals = {}
    for d in df["design_id"].unique():
        design_vals[d] = rng.uniform(0, 1, 3)
    D = np.vstack([design_vals[d] for d in df["design_id"]])
    df["x1"], df["x2"], df["x3"] = D[:, 0], D[:, 1], D[:, 2]
    df["y"] = df["clim"] + 30 * df["x1"] - 10 * df["x2"] + rng.normal(0, 1, len(df))
    return df


# ── facades ──────────────────────────────────────────────────────────────────

def check_facade_azimuth_mapping():
    assert L.facade_azimuth("north", 0) == 0
    assert L.facade_azimuth("north", 90) == 90            # design_space.yaml: north faces east at 90
    assert L.facade_azimuth("east", 90) == 180
    assert L.facade_azimuth("south", 0) == 180
    assert L.facade_azimuth("west", 180) == 90
    assert L.facade_azimuth("west", 135) == 45
    np.testing.assert_allclose(L.facade_azimuth("south", [0, 90, 180]), [180, 270, 0])


def check_azimuth_bins():
    got = list(L.azimuth_bin([0, 10, 350, 22.4, 22.6, 90, 180, 225, 270, 315, 337.6]))
    assert got == ["N", "N", "N", "N", "NE", "E", "S", "SW", "W", "NW", "N"], got


def check_facade_check_detects_west_gain():
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame({"run_id": [f"r{i}" for i in range(n)], "climate_key": "mild", "latitude": 35.0,
                       "orientation": rng.uniform(0, 180, n), "glazing_shgc": rng.uniform(0.1, 0.8, n)})
    for name, f in L.FACADES.items():
        df[f["wwr"]] = rng.uniform(0.1, 0.8, n)
        df[f["overhang"]] = rng.uniform(0, 1.5, n)
        az = L.facade_azimuth(name, df["orientation"])
        sun = np.clip(np.cos(np.radians(az - 250)), 0, None)           # peaks facing ~WSW, 0 facing north
        df[f["zone"]] = 20 + 40 * df[f["wwr"]] * df["glazing_shgc"] * (0.3 + sun) + rng.normal(0, 0.5, n)
    res = L.facade_azimuth_check(L.facade_table(df))
    assert bool(res["pass"].iloc[0]), res
    # same data with the wrong facade->zone wiring must not pass
    bad = df.copy()
    bad["zone_cooling_west_kwh_m2"], bad["zone_cooling_east_kwh_m2"] = df["zone_cooling_east_kwh_m2"], df["zone_cooling_west_kwh_m2"]
    bad["zone_cooling_north_kwh_m2"], bad["zone_cooling_south_kwh_m2"] = df["zone_cooling_south_kwh_m2"], df["zone_cooling_north_kwh_m2"]
    assert not bool(L.facade_azimuth_check(L.facade_table(bad))["pass"].iloc[0])


# ── metrics ──────────────────────────────────────────────────────────────────

def check_design_skill_reference_points():
    df = _frame()
    base = L.climate_baseline(df["y"], df["climate_key"])
    assert abs(L.design_skill(df["y"], base, df["climate_key"])) < 1e-12        # climate mean -> 0
    assert L.design_skill(df["y"], df["y"], df["climate_key"]) == 1.0           # perfect -> 1
    worse = base + np.random.default_rng(0).normal(0, 50, len(df))
    assert L.design_skill(df["y"], worse, df["climate_key"]) < 0                # worse than climate means
    assert L.r2(df["y"], base) > 0.5                                            # climate alone explains a lot


def check_design_skill_matches_worked_example():
    # the 6-building example explained to the user (E3): R2 0.95/0.97/0.997, skill 0/0.44/0.94
    y = np.array([200, 220, 240, 60, 80, 100.0])
    c = np.array(["col"] * 3 + ["sf"] * 3)
    lazy, weak, good = [220, 220, 220, 80, 80, 80], [215, 220, 225, 75, 80, 85], [205, 220, 235, 65, 80, 95]
    assert round(L.r2(y, lazy), 3) == 0.948 and L.design_skill(y, lazy, c) == 0.0
    assert round(L.r2(y, weak), 3) == 0.971 and round(L.design_skill(y, weak, c), 4) == 0.4375
    assert round(L.r2(y, good), 3) == 0.997 and round(L.design_skill(y, good, c), 4) == 0.9375


def check_r2_matches_sklearn():
    from sklearn.metrics import r2_score
    rng = np.random.default_rng(3)
    y, yh = rng.normal(size=200), rng.normal(size=200)
    assert abs(L.r2(y, yh) - r2_score(y, yh)) < 1e-12
    m = L.regression_metrics([10, 20, 30], [12, 18, 30])
    assert abs(m["mae"] - 4 / 3) < 1e-12 and abs(m["cv_rmse"] - np.sqrt(8 / 3) / 20) < 1e-12


def check_heating_climate_selection():
    df = pd.DataFrame({"climate_key": ["a"] * 3 + ["b"] * 3 + ["c"] * 3,
                       "heating_demand_kwh_m2": [0.2, 0.5, 3.0, 1.0, 1.0, 1.0, 5, 6, 7]})
    assert L.heating_climates(df) == ["c"]          # b's median is exactly 1.0 -> not "> 1"


# ── CV ───────────────────────────────────────────────────────────────────────

def check_grouped_cv_isolates_designs():
    df = _frame()
    oof = L.cv_grouped(df, ["x1", "x2", "x3", "latitude"], ["y"], params=SMALL_RF)
    assert len(oof) == len(df) and oof["run_id"].is_unique                      # every row predicted once
    fold_of = oof.merge(df[["run_id", "design_id"]], on="run_id").groupby("design_id")["fold"].nunique()
    assert (fold_of == 1).all(), "an anchor design was split across folds"


def check_cv_recovers_design_signal():
    df = _frame(n_designs=120)
    # all 4 features per split here: with max_features=1/3 of 4 inputs each split sees one feature
    oof = L.cv_grouped(df, ["x1", "x2", "x3", "latitude"], ["y"], params=dict(SMALL_RF, max_features=1.0))
    summ, per = L.evaluate_predictions(oof, heat=[])
    s = summ.iloc[0]
    assert s["r2"] > 0.9 and s["design_skill"] > 0.6, s.to_dict()
    assert len(per) == 3 and s["climate_only_r2"] > 0.5


def check_random_split_runs_and_is_flagged():
    df = _frame()
    oof = L.cv_random(df, ["x1", "x2", "x3", "latitude"], ["y"], params=SMALL_RF)
    assert len(oof) == round(0.2 * len(df)) and (oof["scheme"] == "random").all()


def check_holdout_guard():
    df = _frame()
    for bad_batch in ("holdout",):
        h = df.copy()
        h.loc[h.index[:5], "batch"] = bad_batch
        for fn in (lambda d: L.cv_grouped(d, ["x1"], ["y"], params=SMALL_RF),
                   lambda d: L.cv_random(d, ["x1"], ["y"], params=SMALL_RF),
                   lambda d: L.fit_full(d, ["x1"], "y", params=SMALL_RF)):
            try:
                fn(h)
            except L.HoldoutLeakError:
                continue
            raise AssertionError("holdout rows reached a model function")
    try:
        L.require_training(df.drop(columns="batch"))
    except L.HoldoutLeakError:
        return
    raise AssertionError("frame without batch column accepted")


# ── physics / coverage ───────────────────────────────────────────────────────

def _physics_frame(flip=False, null=False):
    rng = np.random.default_rng(5)
    n = 300
    df = pd.DataFrame({"climate_key": np.repeat(["hot", "cold"], n), "glazing_shgc": rng.uniform(0.1, 0.8, 2 * n)})
    sign = -1 if flip else 1
    effect = 0 if null else 50
    df["cooling_demand_kwh_m2"] = 60 + sign * effect * df["glazing_shgc"] + rng.normal(0, 3, 2 * n)
    return df


def check_physics_detects_flipped_sign():
    rule = [L.Rule("P1", "glazing_shgc", "cooling_demand_kwh_m2", +1, "all", "FAIL")]
    ok = L.physics_rule_status(L.evaluate_physics(_physics_frame(), rule, heat=[]))
    assert ok["status"].iloc[0] == "PASS", ok
    bad = L.physics_rule_status(L.evaluate_physics(_physics_frame(flip=True), rule, heat=[]))
    assert bad["status"].iloc[0] == "FAIL" and bad["opposite"].iloc[0] == ["cold", "hot"], bad
    null = L.physics_rule_status(L.evaluate_physics(_physics_frame(null=True), rule, heat=[]))
    assert null["status"].iloc[0] == "WARN", null                     # strong rule with no effect is suspicious
    soft = [L.Rule("P6", "glazing_shgc", "cooling_demand_kwh_m2", +1, "all", "WARN")]
    assert L.physics_rule_status(L.evaluate_physics(_physics_frame(null=True), soft, heat=[]))["status"].iloc[0] == "INFO"
    assert L.physics_rule_status(L.evaluate_physics(_physics_frame(flip=True), soft, heat=[]))["status"].iloc[0] == "WARN"


def check_physics_rules_reference_real_columns():
    inputs = [c for cols in L.FEATURE_GROUPS.values() for c in cols]
    targets = L.PRIMARY_TARGETS + L.SECONDARY_TARGETS
    for r in L.PHYSICS_RULES:
        assert r.x in inputs and r.y in targets and r.severity in ("FAIL", "WARN"), r
    assert len({r.id for r in L.PHYSICS_RULES}) == len(L.PHYSICS_RULES)


def check_coverage_and_correlation_flags():
    rng = np.random.default_rng(2)
    n = 300
    bounds = {"a": {"min": 0.0, "max": 1.0, "integer": False}, "k": {"min": 1, "max": 5, "integer": True}}
    good = pd.DataFrame({"climate_key": "c", "a": (rng.permutation(n) + rng.uniform(0, 1, n)) / n,
                         "k": np.repeat([1, 2, 3, 4, 5], n // 5)})
    assert not L.coverage_uniformity(good, ["a", "k"], bounds)["flag"].any()
    skew = good.assign(a=good["a"] ** 3, k=np.where(np.arange(n) < 200, 1, good["k"]))
    assert L.coverage_uniformity(skew, ["a", "k"], bounds)["flag"].all()
    cor = good.assign(b=good["a"] + rng.normal(0, 0.05, n), c=rng.uniform(0, 1, n))
    res = L.input_correlation(cor, ["a", "b", "c"])
    assert res["flag"].iloc[0] and res["pair"].iloc[0] == "a ~ b"
    assert not L.input_correlation(cor, ["a", "c"])["flag"].iloc[0]


# ── SHAP / gate ──────────────────────────────────────────────────────────────

def check_shap_grouping_is_additive():
    df = _frame()
    feats = ["x1", "x2", "x3", "latitude"]
    groups = {"design": ["x1", "x2", "x3"], "climate": ["latitude"]}
    m = L.fit_full(df, feats, "y", params=SMALL_RF)
    sv = L.shap_values(m, df[feats].iloc[:40])
    gs = L.shap_grouped(sv, groups)
    np.testing.assert_allclose(gs.sum(axis=1), sv.sum(axis=1))
    pred = m.predict(df[feats].iloc[:40].to_numpy(float))
    base = pred - sv.sum(axis=1).to_numpy()
    assert np.ptp(base) < 1e-6                               # SHAP additivity: constant expected value
    assert abs(L.shap_share(gs).sum() - 1) < 1e-12
    assert L.shap_importance(sv).index[0] == "latitude"      # climate dominates the synthetic y
    try:
        L.shap_grouped(sv, {"design": ["x1", "x2"], "climate": ["latitude"]})
    except ValueError:
        return
    raise AssertionError("incomplete groups accepted")


def check_gate_logic():
    C = L.Check
    assert L.gate_status([C("a", "", "FAIL", "PASS"), C("b", "", "WARN", "INFO")]) == "GO"
    assert L.gate_status([C("a", "", "FAIL", "PASS"), C("b", "", "WARN", "WARN")]) == "REVIEW"
    assert L.gate_status([C("a", "", "FAIL", "FAIL"), C("b", "", "WARN", "WARN")]) == "NO-GO"
    summ = pd.DataFrame([{"target": t, "r2": 0.9, "design_skill": 0.8, "skill_climates": 10}
                         for t in L.PRIMARY_TARGETS + L.SECONDARY_TARGETS])
    assert L.gate_status(L.signal_checks(summ)) == "GO"
    summ.loc[summ["target"] == "peak_cooling_w_m2", "r2"] = 0.6
    assert L.gate_status(L.signal_checks(summ)) == "REVIEW"           # secondary -> warning only
    summ.loc[summ["target"] == "eui_kwh_m2", "design_skill"] = 0.69
    assert L.gate_status(L.signal_checks(summ)) == "NO-GO"            # primary skill below 0.70


# ── real dataset (read-only) ─────────────────────────────────────────────────

def check_end_to_end_on_dataset():
    if not (L.DATASET_DIR / "dataset_training.csv").exists():
        print("        (skipped: no outputs/dataset)")
        return
    df, cols, sha = L.load_split("training")
    hold, _, _ = L.load_split("holdout")
    assert L.check_rows(df, hold).status == "PASS" and len(sha) == 64
    L.check_groups_cover(cols["inputs"])
    assert len(cols["inputs"]) == 28 and set(L.PRIMARY_TARGETS + L.SECONDARY_TARGETS) == set(cols["targets"])
    heat = L.heating_climates(df)
    assert len(heat) == 7 and not {"chennai", "colombo", "miami"} & set(heat), heat
    bounds = L.design_bounds()
    assert set(cols["design_inputs"]) <= set(bounds)
    cov = L.coverage_uniformity(df, cols["design_inputs"], bounds)
    assert len(cov) == 21 * 10
    fac = L.facade_table(df)
    assert len(fac) == 4 * len(df) and fac["residual"].notna().all()
    try:
        L.require_training(hold)
    except L.HoldoutLeakError:
        return
    raise AssertionError("real holdout accepted by require_training")


def check_decision_rule():
    C = L.Check
    ok = [C("G0", "", "FAIL", "PASS"), C("P1", "", "FAIL", "PASS")]
    lc_up = pd.DataFrame({"fraction": [0.4, 0.7, 1.0] * 2, "target": ["eui_kwh_m2"] * 3 + ["x"] * 3,
                          "design_skill": [0.58, 0.64, 0.67, 0.5, 0.5, 0.5]})
    lc_flat = lc_up.assign(design_skill=[0.66, 0.68, 0.67, 0.5, 0.5, 0.5])
    assert L.decide(ok, lc_up)["decision"] == "GO"
    assert L.decide(ok + [C("P3-wall", "", "FAIL", "WARN")], lc_up)["decision"] == "REVIEW"
    model = ok + [C("R2:eui_kwh_m2", "", "FAIL", "FAIL")]
    d = L.decide(model, lc_up)
    assert d["decision"].startswith("PROCEED") and d["gate"] == "NO-GO", d     # gate itself unchanged
    assert L.decide(model, lc_flat)["decision"] == "NO-GO"                      # not rising -> investigate
    assert L.decide(model, None)["decision"] == "NO-GO"                         # no evidence -> no exception
    data = model + [C("P4-west", "", "FAIL", "FAIL")]
    assert "fix the data" in L.decide(data, lc_up)["reason"]                    # data failure always blocks


def check_learning_curve_runs():
    df = _frame(n_designs=60)
    lc = L.learning_curve_skill(df, ["x1", "x2", "x3", "latitude"], ["y"], heat=[],
                                params=dict(SMALL_RF, max_features=1.0), fractions=(0.5, 1.0))
    assert list(lc["fraction"]) == [0.5, 1.0] and lc["n_designs"].is_monotonic_increasing


ALL_CHECKS = [check_facade_azimuth_mapping, check_azimuth_bins, check_facade_check_detects_west_gain,
              check_design_skill_reference_points, check_design_skill_matches_worked_example,
              check_r2_matches_sklearn, check_heating_climate_selection, check_grouped_cv_isolates_designs,
              check_cv_recovers_design_signal, check_random_split_runs_and_is_flagged, check_holdout_guard,
              check_physics_detects_flipped_sign, check_physics_rules_reference_real_columns,
              check_coverage_and_correlation_flags, check_shap_grouping_is_additive, check_gate_logic,
              check_decision_rule, check_learning_curve_runs, check_end_to_end_on_dataset]


def main():
    total = failed = 0
    print("=" * 72)
    print("EDA LIBRARY (2.8) - CHECKS")
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
