"""
test_benchmark.py - checks for surrogate/benchmark.py (Sub-task 3.3 step 3).

The pre-registered M10 selection rule and the D5 transfer rule on hand-made tables with known
answers; CV-stage gate checks; refusal to score without (or with mismatched) tuned settings; and a
small real out-of-fold run for output shape. Training split only.

Run:  ./venv/bin/python surrogate/tests/test_benchmark.py   (or all suites: surrogate/run_tests.py)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from surrogate import benchmark as B, cv, features as FE, models, tune
from surrogate.config import CONFIG, PRIMARY_TARGETS, SECONDARY_TARGETS

H, EUI, PK = "heating_demand_kwh_m2", "eui_kwh_m2", "peak_cooling_w_m2"


def _cands(rows):
    return pd.DataFrame([{"model": m, "feature_set": "C-G", "grouped_design_skill": s, "loco_mape_pct": mape}
                         for m, s, mape in rows])


# ── M10 ─────────────────────────────────────────────────────────────────────────────────────

def check_primary_lowest_loco_mape_wins():
    r = B.select_target(_cands([("rf", 0.95, 20.0), ("xgb", 0.96, 12.0), ("lgbm", 0.97, 15.0), ("mlp", 0.93, 30.0)]), H)
    assert r["model"] == "xgb" and r["status"] == "provisional_until_3_4" and not r["gate_fail"], r


def check_primary_tie_goes_to_lightgbm():
    r = B.select_target(_cands([("xgb", 0.96, 12.0), ("lgbm", 0.95, 12.5)]), EUI)     # exactly 0.5 points
    assert r["model"] == "lgbm" and "LightGBM within the tie margin" in r["rule"], r
    r = B.select_target(_cands([("xgb", 0.96, 12.0), ("lgbm", 0.95, 12.51)]), EUI)    # just outside
    assert r["model"] == "xgb", r


def check_primary_skill_bar_filters_first():
    # the lowest MAPE has skill 0.89 < 0.90 -> excluded
    r = B.select_target(_cands([("mlp", 0.89, 5.0), ("rf", 0.92, 20.0), ("lgbm", 0.95, 18.0)]), H)
    assert r["model"] == "lgbm" and r["n_passing"] == 2, r
    # nobody passes -> best grouped skill, gate FAIL recorded
    r = B.select_target(_cands([("mlp", 0.85, 5.0), ("rf", 0.88, 20.0)]), H)
    assert r["model"] == "rf" and r["gate_fail"], r


def check_secondary_highest_skill_and_tie():
    c = pd.DataFrame([{"model": "gpr", "feature_set": "C-G", "grouped_design_skill": 0.950},
                      {"model": "lgbm", "feature_set": "C-G", "grouped_design_skill": 0.946}])
    r = B.select_target(c, PK)
    assert r["model"] == "lgbm" and r["status"] == "final" and r["skill_bar"] == 0.85, r
    c.loc[1, "grouped_design_skill"] = 0.9449
    assert B.select_target(c, PK)["model"] == "gpr"


def check_gate_cv_stage():
    table = pd.DataFrame([{"model": "lgbm", "feature_set": "C-G", "target": H, "grouped_design_skill": 0.95, "grouped_r2": 0.94},
                          {"model": "lgbm", "feature_set": "C-G", "target": PK, "grouped_design_skill": 0.84, "grouped_r2": 0.95}])
    sel = {H: {"model": "lgbm", "feature_set": "C-G"}, PK: {"model": "lgbm", "feature_set": "C-G"}}
    g = B.gate_cv_stage(sel, table)
    assert g[H]["check"] == "S1" and g[H]["result"] == "FAIL"           # pooled R2 0.94 < 0.95
    assert g[PK]["check"] == "S2" and g[PK]["result"] == "WARN"


# ── D5 and helpers ──────────────────────────────────────────────────────────────────────────

def check_d5_rule():
    rows = []
    for m, cg, r in (("xgb", 0.95, 0.85), ("gpr", 0.90, 0.90), ("mlp", 0.80, 0.85)):
        for t in PRIMARY_TARGETS:
            rows += [{"model": m, "feature_set": "C-G", "target": t, "grouped_design_skill": cg},
                     {"model": m, "feature_set": "R", "target": t, "grouped_design_skill": r}]
    d5 = B.d5_check(pd.DataFrame(rows))
    assert set(d5) == {"xgb", "gpr", "mlp"}
    assert not d5["xgb"]["triggered"] and d5["gpr"]["triggered"] and d5["mlp"]["triggered"]   # equal = not higher


def check_best_other_excludes_lightgbm():
    t = pd.DataFrame([{"model": m, "grouped_design_skill": s} for m, s in
                      (("lgbm", 0.99), ("xgb", 0.95), ("gpr", 0.96), ("rf", 0.93), ("mlp", 0.94))])
    assert B.best_other(t) == "gpr"


def check_refuses_missing_or_mismatched_tuned_settings():
    with tempfile.TemporaryDirectory() as tmp:
        old = tune.OUT
        tune.OUT = Path(tmp)
        try:
            try:
                B.tuned_params_for("lgbm", PRIMARY_TARGETS, dev=False)
            except SystemExit:
                pass
            else:
                raise AssertionError("scored without tuned settings")
            d = Path(tmp) / "tuning"
            d.mkdir()
            rec = {"feature_set": "C-G", "params": {"n_estimators": 10}}
            (d / "best_params.json").write_text(json.dumps({"studies": {f"lgbm__{t}": rec for t in PRIMARY_TARGETS}}))
            got = B.tuned_params_for("lgbm", PRIMARY_TARGETS, dev=False, feature_set="C-G")
            assert got[H] == {"n_estimators": 10}
            try:
                B.tuned_params_for("lgbm", PRIMARY_TARGETS, dev=False, feature_set="C-S")
            except SystemExit:
                pass
            else:
                raise AssertionError("mismatched feature set accepted")
            g = B.tuned_params_for("gpr", PRIMARY_TARGETS, dev=False, gpr_top_k=15)
            assert g[H]["top_k"] == 15 and g[H]["nu"] == 2.5
        finally:
            tune.OUT = old


def check_gpr_fallback_read_from_probe():
    with tempfile.TemporaryDirectory() as tmp:
        old = B.OUT
        B.OUT = Path(tmp)
        try:
            try:
                B.resolve_gpr_top_k(dev=False)
            except SystemExit:
                pass
            else:
                raise AssertionError("official stage ran without the timing probe")
            probe = B.out_dir(False) / "timing_probe.json"
            probe.write_text(json.dumps({"gpr_fallback_needed": True}))
            assert B.resolve_gpr_top_k(False) == 15 and B.resolve_gpr_top_k(False, 15) == 15
            assert B.default_for("gpr", False, 15)["top_k"] == 15 and "top_k" not in B.default_for("lgbm", False, 15)
            probe.write_text(json.dumps({"gpr_fallback_needed": False}))
            assert B.resolve_gpr_top_k(False) is None
            try:
                B.resolve_gpr_top_k(False, 15)
            except SystemExit:
                pass
            else:
                raise AssertionError("flag disagreeing with the probe accepted")
        finally:
            B.OUT = old


def check_factorial_table_both_orders():
    """2 x 2 cells -> absolute and relative effects in both orders; GPR has no tuned cells."""
    def tab(rows, fs_col=True):
        out = []
        for m, fs, t, sk in rows:
            r = {"model": m, "target": t, "grouped_design_skill": sk}
            if fs_col:
                r["feature_set"] = fs
            out.append(r)
        return pd.DataFrame(out)
    untuned = tab([(m, fs, t, sk) for t in PRIMARY_TARGETS for m, fs, sk in
                   (("xgb", "R", 0.80), ("xgb", "C-G", 0.95), ("gpr", "R", 0.90), ("gpr", "C-G", 0.95))])
    tuned_cg = tab([("xgb", "C-G", t, 0.98) for t in PRIMARY_TARGETS] + [("gpr", "C-G", t, 0.95) for t in PRIMARY_TARGETS])
    tuned_r = tab([("xgb", "R", t, 0.90) for t in PRIMARY_TARGETS])
    f = B.factorial_table(untuned, tuned_cg, tuned_r)
    x = f[(f.model == "xgb") & (f.target == H)].iloc[0]
    assert np.isclose(x.features_gain_untuned, 0.15) and np.isclose(x.features_gain_tuned, 0.08)
    assert np.isclose(x.tuning_gain_R, 0.10) and np.isclose(x.tuning_gain_CG, 0.03)
    assert np.isclose(x.features_rel_untuned, 0.75) and np.isclose(x.features_rel_tuned, 0.8)    # 0.08 / 0.10
    assert np.isclose(x.tuning_rel_R, 0.5) and np.isclose(x.tuning_rel_CG, 0.6)                  # 0.03 / 0.05
    assert np.isclose(x.both_rel, 0.9)
    g = f[(f.model == "gpr") & (f.target == H)].iloc[0]
    assert not g.tuned_by_optuna and np.isnan(g.R_tuned) and np.isnan(g.CG_tuned) and np.isclose(g.features_gain_untuned, 0.05)
    assert np.isclose(B.rel_gain(0.704, 0.942), 0.238 / 0.296)


def check_small_real_oof_shapes():
    df, _ = cv.load_training()
    keep = set(df["design_id"].astype(str).unique()[:60])
    df = df[df["design_id"].astype(str).isin(keep)].reset_index(drop=True)
    X = FE.engineer(df)[FE.feature_set("C-G")]
    folds = cv.grouped_folds(df)
    p = {**models.default_params("lgbm"), "n_estimators": 20}
    oof = B.oof_for("lgbm", {t: p for t in PRIMARY_TARGETS}, df, X, PRIMARY_TARGETS, folds, "grouped")
    assert len(oof) == 3 * len(df) and list(oof["target"].unique()) == PRIMARY_TARGETS
    pm = {**models.default_params("mlp"), "max_epochs": 5}
    oof = B.oof_for("mlp", {None: pm}, df, X, SECONDARY_TARGETS, folds, "grouped")
    assert len(oof) == 7 * len(df) and set(oof["target"]) == set(SECONDARY_TARGETS)
    fit = B.fit_seconds_per_fold(oof)
    assert (fit > 0).all() and np.isclose(fit.nunique(), 1)          # one net -> same time for every target


ALL_CHECKS = [check_primary_lowest_loco_mape_wins, check_primary_tie_goes_to_lightgbm,
              check_primary_skill_bar_filters_first, check_secondary_highest_skill_and_tie, check_gate_cv_stage,
              check_d5_rule, check_best_other_excludes_lightgbm, check_refuses_missing_or_mismatched_tuned_settings,
              check_gpr_fallback_read_from_probe, check_factorial_table_both_orders, check_small_real_oof_shapes]


def main():
    total = failed = 0
    print("=" * 72)
    print("SURROGATE BENCHMARK (3.3) - CHECKS")
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
