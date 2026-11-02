"""
test_cv.py - checks for surrogate/cv.py, metrics.py and config.yaml (Sub-task 3.1 step 5).

Synthetic checks with known answers (fold isolation, LOCO isolation, holdout guard, every metric),
then checks on the real training split (read-only): config matches the 2.8 library and the dataset
manifest, folds are identical to 2.8, and the harness reproduces eda_lib.cv_grouped exactly on a
small forest. The holdout split is never read.

Run:  ./venv/bin/python surrogate/tests/test_cv.py   (or all suites: surrogate/run_tests.py)
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from eda import eda_lib as L
from surrogate import cv, metrics as M
from surrogate.config import CONFIG, ALL_TARGETS, PRIMARY_TARGETS, SECONDARY_TARGETS
from surrogate.paths import DATASET_DIR

SMALL_RF = dict(n_estimators=30, min_samples_leaf=2, max_features=1 / 3, n_jobs=-1, random_state=0)
_REAL: dict = {}


def _real():
    if not _REAL:
        _REAL["df"], _REAL["cols"] = cv.load_training()
    return _REAL["df"], _REAL["cols"]


def _frame(n_designs=40, climates=("hot", "mild", "cold", "polar"), anchors=10, seed=7):
    """Synthetic training frame: `anchors` designs in every climate, the rest climate-specific."""
    rng = np.random.default_rng(seed)
    effect = {"hot": 100.0, "mild": 60.0, "cold": 40.0, "polar": 20.0}
    rows, nxt = [], anchors
    for c in climates:
        ids = [f"A{i:03d}" for i in range(anchors)]
        for _ in range(n_designs - anchors):
            ids.append(f"B{nxt:03d}")
            nxt += 1
        rows += [{"run_id": f"main_{d}_{c}", "batch": "main", "design_id": d, "climate_key": c,
                  "clim": effect[c]} for d in ids]
    df = pd.DataFrame(rows)
    vals = {d: rng.uniform(0, 1, 2) for d in df["design_id"].unique()}
    D = np.vstack([vals[d] for d in df["design_id"]])
    df["x1"], df["x2"] = D[:, 0], D[:, 1]
    df["y"] = df["clim"] + 30 * df["x1"] - 10 * df["x2"] + rng.normal(0, 0.5, len(df))
    return df


class ClimateMean:
    """Predicts the training mean of each climate code (column 0): the 'lazy' baseline."""
    def fit(self, X, y):
        X = np.asarray(X, float)
        self.means = pd.Series(y).groupby(X[:, 0]).mean().to_dict()
        self.overall = float(np.mean(y))
        return self

    def predict(self, X):
        return np.array([self.means.get(v, self.overall) for v in np.asarray(X, float)[:, 0]])


# ── config ───────────────────────────────────────────────────────────────────

def check_config_matches_eda_lib():
    p = CONFIG["protocol"]
    assert p["seed"] == L.SEED and p["grouped_cv"]["n_folds"] == L.N_FOLDS
    assert PRIMARY_TARGETS == L.PRIMARY_TARGETS and SECONDARY_TARGETS == L.SECONDARY_TARGETS
    assert tuple(CONFIG["targets"]["heating"]) == L.HEATING_TARGETS
    assert CONFIG["targets"]["heating_climate_min_median"] == L.HEATING_CLIMATE_MIN_MEDIAN
    assert CONFIG["reference_2_8"]["rf_params"] == L.RF_PARAMS


def check_config_matches_dataset_and_2_8_results():
    manifest = json.loads((DATASET_DIR / "dataset_manifest.json").read_text(encoding="utf-8"))
    for split in ("training", "holdout"):
        assert manifest["files"][split]["sha256"] == CONFIG["dataset_sha256"][split], split
    eda = json.loads((L.ROOT / "outputs" / "eda" / "eda_metrics.json").read_text(encoding="utf-8"))
    ref = {r["target"]: r["design_skill"] for r in eda["signal_grouped"]}
    assert set(ref) == set(ALL_TARGETS)
    for t, v in CONFIG["reference_2_8"]["design_skill"].items():
        assert abs(ref[t] - v) < 5e-7, f"{t}: config {v} vs 2.8 {ref[t]}"


def check_gate_thresholds_frozen():
    g, s = CONFIG["gate"], CONFIG["selection"]
    assert g["S1"]["design_skill_min"] == 0.90 and g["S1"]["pooled_r2_min"] == 0.95
    assert g["S2"]["design_skill_min"] == 0.85 and g["S0"]["tolerance"] == 0.002
    assert s["primary_skill_min"] == 0.90 and s["secondary_skill_min"] == 0.85
    assert CONFIG["m7_trigger"]["loco_median_abs_bias_pct"] == 10.0
    assert CONFIG["signed_off"] == "2026-10-30"


# ── folds (synthetic) ────────────────────────────────────────────────────────

def check_grouped_folds_isolate_designs():
    df = _frame()
    folds = cv.grouped_folds(df)
    assert len(folds) == 5
    for f in folds:
        assert not set(df["design_id"].iloc[f.train]) & set(df["design_id"].iloc[f.test])
    tested = np.sort(np.concatenate([f.test for f in folds]))
    assert np.array_equal(tested, np.arange(len(df)))


def check_loco_never_trains_on_held_out_climate():
    df = _frame()
    folds = cv.loco_folds(df)
    assert [f.name for f in folds] == sorted(df["climate_key"].unique())
    for f in folds:
        assert set(df["climate_key"].iloc[f.test]) == {f.name}
        assert f.name not in set(df["climate_key"].iloc[f.train])
    oof = cv.run_loco(df, df[["x1", "x2"]], ["y"], lambda t: RandomForestRegressor(**SMALL_RF))
    assert len(oof) == len(df) and oof["run_id"].is_unique
    assert (oof["fold"] == oof["climate_key"]).all()


def check_holdout_rows_rejected_everywhere():
    df = _frame()
    bad = df.copy()
    bad.loc[0, "batch"] = "holdout_ood"
    calls = [lambda: cv.grouped_folds(bad), lambda: cv.loco_folds(bad),
             lambda: cv.cross_val_predict(bad, bad[["x1"]], ["y"], lambda t: ClimateMean(),
                                          cv.grouped_folds(df), "grouped")]
    for call in calls:
        try:
            call()
        except L.HoldoutLeakError:
            continue
        raise AssertionError("a holdout row was accepted")
    try:
        cv.grouped_folds(df.drop(columns="batch"))
    except L.HoldoutLeakError:
        pass
    else:
        raise AssertionError("frame without a batch column was accepted")


def check_x_must_align_with_df():
    df = _frame()
    try:
        cv.cross_val_predict(df, df[["x1"]].iloc[:-1], ["y"], lambda t: ClimateMean(), cv.grouped_folds(df), "g")
    except ValueError:
        return
    raise AssertionError("misaligned X accepted")


def check_harness_is_model_agnostic():
    """A non-sklearn object works: the climate-mean model has design skill about 0; a model that sees
    the (linear) design signal has skill close to 1."""
    df = _frame()
    df["code"] = df["climate_key"].map({"hot": 0, "mild": 1, "cold": 2, "polar": 3}).astype(float)
    lazy = cv.run_grouped(df, df[["code"]], ["y"], lambda t: ClimateMean())
    sk_lazy = M.design_skill(lazy["y"], lazy["yhat"], lazy["climate_key"])
    assert abs(sk_lazy) < 0.1, sk_lazy
    X = pd.get_dummies(df["climate_key"], dtype=float).join(df[["x1", "x2"]])
    lin = cv.run_grouped(df, X, ["y"], lambda t: LinearRegression())
    sk_lin = M.design_skill(lin["y"], lin["yhat"], lin["climate_key"])
    assert sk_lin > 0.95, sk_lin


# ── metrics (synthetic, known answers) ───────────────────────────────────────

def _loco_oof(shift_pct: float, reverse: bool = False):
    rows = []
    for c, level in [("a", 50.0), ("b", 100.0)]:
        y = level + np.arange(20, dtype=float)
        yhat = (y[::-1] if reverse else y) + level * shift_pct / 100      # shift by shift_pct % of the level
        rows.append(pd.DataFrame({"run_id": [f"{c}{i}" for i in range(20)], "design_id": [f"d{i}" for i in range(20)],
                                  "climate_key": c, "target": "cooling_demand_kwh_m2", "fold": c,
                                  "y": y, "yhat": yhat, "fit_s": 0.0, "scheme": "loco"}))
    return pd.concat(rows, ignore_index=True)


def check_loco_bias_and_bias_removed_r2():
    s, tab = M.loco_summary(_loco_oof(10.0), heat=[])
    # yhat = y + 10 % of each climate's base level -> bias = 10 % * level / mean(y)
    a = tab.set_index("climate_key").loc["a"]
    assert abs(a["bias_pct"] - 100 * 5.0 / 59.5) < 1e-9, a["bias_pct"]
    assert abs(a["r2_bias_removed"] - 1.0) < 1e-12 and abs(a["spearman"] - 1.0) < 1e-12 and a["top_hit"] == 1.0
    row = s.iloc[0]
    assert abs(row["bias_removed_r2"] - 1.0) < 1e-12 and row["climates"] == 2
    assert row["worst_climate"] == "b"


def check_reversed_ranking_detected():
    s, tab = M.loco_summary(_loco_oof(0.0, reverse=True), heat=[])
    assert (tab["spearman"] < -0.99).all() and (tab["top_hit"] == 0).all()
    assert abs(s.iloc[0]["median_abs_bias_pct"]) < 1e-9      # level right, ranking wrong


def check_top_hit_rate_known_case():
    y = np.arange(20.0)
    assert M.top_hit_rate(y, y) == 1.0                       # k = 2: designs 0 and 1
    assert M.top_hit_rate(y, np.r_[1.0, 50.0, y[2:]]) == 0.5  # design 1 pushed out, design 2 enters
    assert M.top_hit_rate(y, -y) == 0.0


def check_mape_skips_near_zero_truth():
    y = np.array([0.0, 0.2, 10.0, 20.0])
    yhat = np.array([5.0, 5.0, 11.0, 18.0])
    assert abs(M.mape_pct(y, yhat) - 100 * np.mean([0.1, 0.1])) < 1e-12


def check_heating_targets_scored_on_heating_climates():
    oof = _loco_oof(5.0)
    oof["target"] = "heating_demand_kwh_m2"
    s, tab = M.loco_summary(oof, heat=["a"])
    assert list(tab["climate_key"]) == ["a"] and s.iloc[0]["climates"] == 1


def check_m7_trigger_logic():
    s = pd.DataFrame({"target": ["heating_demand_kwh_m2", "eui_kwh_m2", "cooling_demand_kwh_m2"],
                      "median_abs_bias_pct": [7.6, 3.9, 50.0]})
    fired, vals = M.m7_triggered(s)
    assert not fired and vals == {"heating_demand_kwh_m2": 7.6, "eui_kwh_m2": 3.9}   # cooling not in the rule
    s.loc[1, "median_abs_bias_pct"] = 10.5
    assert M.m7_triggered(s)[0]


def check_summarise_adds_ranking_columns():
    df = _frame()
    oof = cv.run_grouped(df, df[["clim", "x1", "x2"]], ["y"], lambda t: RandomForestRegressor(**SMALL_RF))
    summary, per = M.summarise(oof, heat=[])
    assert {"design_skill", "r2", "spearman_median", "top_hit_mean"} <= set(summary.columns)
    assert {"spearman", "top_hit"} <= set(per.columns) and len(per) == 4


# ── real dataset (read-only, training split only) ────────────────────────────

def check_real_folds_identical_to_2_8():
    df, cols = _real()
    groups = df["design_id"].astype(str).to_numpy()
    ref = list(GroupKFold(n_splits=L.N_FOLDS, shuffle=True, random_state=L.SEED)
               .split(df[cols["inputs"]].to_numpy(float), groups=groups))
    ours = cv.grouped_folds(df)
    assert len(ours) == len(ref) == 5
    for f, (tr, te) in zip(ours, ref):
        assert np.array_equal(f.train, tr) and np.array_equal(f.test, te)
    assert len(cv.loco_folds(df)) == 10 and df["climate_key"].nunique() == 10


def check_real_harness_reproduces_eda_lib_exactly():
    """Same folds + same RF -> identical out-of-fold predictions to eda_lib.cv_grouped (small forest)."""
    df, cols = _real()
    t = ["eui_kwh_m2"]
    ref = L.cv_grouped(df, cols["inputs"], t, params=SMALL_RF)
    ours = cv.run_grouped(df, df[cols["inputs"]], t, lambda _: RandomForestRegressor(**SMALL_RF))
    m = ref.merge(ours, on=["run_id", "target"], suffixes=("_ref", "_new"))
    assert len(m) == len(df)
    assert (m["fold_ref"].astype(str) == m["fold_new"]).all()
    np.testing.assert_allclose(m["yhat_new"], m["yhat_ref"], rtol=0, atol=1e-9)
    heat = M.heating_climates(df)
    a, _ = L.evaluate_predictions(ref, heat)
    b, _ = M.summarise(ours, heat)
    assert abs(a["design_skill"].iloc[0] - b["design_skill"].iloc[0]) < 1e-12


def check_real_heating_climates_match_2_8():
    df, _ = _real()
    eda = json.loads((L.ROOT / "outputs" / "eda" / "eda_metrics.json").read_text(encoding="utf-8"))
    assert M.heating_climates(df) == eda["heating_climates"]


ALL_CHECKS = [check_config_matches_eda_lib, check_config_matches_dataset_and_2_8_results, check_gate_thresholds_frozen,
              check_grouped_folds_isolate_designs, check_loco_never_trains_on_held_out_climate,
              check_holdout_rows_rejected_everywhere, check_x_must_align_with_df, check_harness_is_model_agnostic,
              check_loco_bias_and_bias_removed_r2, check_reversed_ranking_detected, check_top_hit_rate_known_case,
              check_mape_skips_near_zero_truth, check_heating_targets_scored_on_heating_climates,
              check_m7_trigger_logic, check_summarise_adds_ranking_columns,
              check_real_folds_identical_to_2_8, check_real_harness_reproduces_eda_lib_exactly,
              check_real_heating_climates_match_2_8]


def main():
    total = failed = 0
    print("=" * 72)
    print("SURROGATE CV HARNESS (3.1) - CHECKS")
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
