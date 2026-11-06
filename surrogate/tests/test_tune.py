"""
test_tune.py - checks for surrogate/tune.py (Sub-task 3.3 step 3).

The objective equals an independently computed grouped-CV RMSE (heating climates for heating
targets); the MLP objective is the mean RMSE/SD over the primary targets; budgets come from config;
a resumed study does not repeat trials; outputs are written; range-edge flags are correct.
Runs in a temporary output folder. Training split only.

Run:  ./venv/bin/python surrogate/tests/test_tune.py   (or all suites: surrogate/run_tests.py)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from surrogate import cv, features as FE, metrics as M, models, tune
from surrogate.config import CONFIG, PRIMARY_TARGETS

optuna.logging.set_verbosity(optuna.logging.WARNING)
CFG = CONFIG["benchmark_3_3"]
_REAL: dict = {}
LGBM_TRIAL = {"n_estimators": 100, "learning_rate": 0.1, "num_leaves": 15, "min_child_samples": 20, "subsample": 0.9,
              "colsample_bytree": 0.8, "reg_alpha": 1e-3, "reg_lambda": 1e-3}


def _real():
    if not _REAL:
        df, _ = cv.load_training()
        _REAL.update(df=df, X=FE.engineer(df)[FE.feature_set("C-G")], heat=M.heating_climates(df))
    return _REAL["df"], _REAL["X"], _REAL["heat"]


class _TmpOut:
    """Point tune.OUT at a temporary folder for the duration of a check."""
    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = tune.OUT
        tune.OUT = Path(self.tmp.name)
        return tune.OUT

    def __exit__(self, *exc):
        tune.OUT = self.old
        self.tmp.cleanup()


def check_budgets_from_config():
    for m in ("rf", "xgb", "lgbm"):
        assert tune.budget(m, False) == 60
    assert tune.budget("mlp", False) == 30
    assert tune.budget("lgbm", True) == tune.DEV_TRIALS
    try:
        tune.budget("gpr", False)
    except ValueError:
        return
    raise AssertionError("GPR accepted for Optuna tuning (E5)")


def check_scored_rmse_uses_heating_climates_for_heating_targets():
    oof = pd.DataFrame({"target": ["heating_demand_kwh_m2"] * 4 + ["cooling_demand_kwh_m2"] * 2,
                        "climate_key": ["cold", "cold", "hot", "hot", "hot", "cold"],
                        "y": [10.0, 20.0, 0.0, 0.0, 50.0, 30.0], "yhat": [12.0, 18.0, 9.0, 9.0, 53.0, 26.0]})
    assert abs(tune.scored_rmse(oof, "heating_demand_kwh_m2", ["cold"]) - 2.0) < 1e-12
    assert abs(tune.scored_rmse(oof, "cooling_demand_kwh_m2", ["cold"]) - np.sqrt((9 + 16) / 2)) < 1e-12


def check_objective_equals_independent_cv_rmse():
    df, X, heat = _real()
    t = "heating_demand_kwh_m2"
    folds = cv.grouped_folds(df)
    obj = tune.make_objective("lgbm", t, df, X, heat, folds, dev=False, sds={})
    value = obj(optuna.trial.FixedTrial(LGBM_TRIAL))
    params = models.tuned_params("lgbm", LGBM_TRIAL)
    oof = cv.run_grouped(df, X, [t], models.make("lgbm", params))
    g = oof[oof["climate_key"].isin(heat)]
    ref = float(np.sqrt(np.mean((g["yhat"] - g["y"]) ** 2)))
    assert abs(value - ref) < 1e-9, (value, ref)


def check_mlp_objective_is_mean_normalised_primary_rmse():
    df, X, heat = _real()
    sds = {t: tune.scored_sd(df, t, heat) for t in PRIMARY_TARGETS}
    obj = tune.make_objective("mlp", tune.MULTI_STUDY, df, X, heat, cv.grouped_folds(df), dev=True, sds=sds)
    trial = optuna.trial.FixedTrial({"n_layers": 2, "width": 64, "dropout": 0.0, "lr": 1e-3, "weight_decay": 1e-4,
                                     "batch_size": 256})
    value = obj(trial)
    ref = np.mean([trial.user_attrs[f"rmse__{t}"] / sds[t] for t in PRIMARY_TARGETS])
    assert abs(value - ref) < 1e-12
    assert len([k for k in trial.user_attrs if k.startswith("skill__")]) == 10      # all targets scored


def check_study_resumes_without_repeating_and_saves_outputs():
    df, X, heat = _real()
    with _TmpOut() as out:
        st = tune.run_study("lgbm", "eui_kwh_m2", df, X, heat, dev=True, log=lambda s: None)
        tune.save_study(st, dev=True)
        n1 = len(st.trials)
        st2 = tune.run_study("lgbm", "eui_kwh_m2", df, X, heat, dev=True, log=lambda s: None)
        assert len(st2.trials) == n1 == tune.DEV_TRIALS, (n1, len(st2.trials))
        d = out / "tuning_dev"
        assert (d / "optuna.db").exists() and (d / "trials" / "lgbm__eui_kwh_m2.csv").exists()
        best = json.loads((d / "best_params.json").read_text())["studies"]["lgbm__eui_kwh_m2"]
        for k, v in CFG["fixed"]["lgbm"].items():
            assert best["params"][k] == v, k
        assert best["feature_set"] == "C-G" and best["n_trials"] == tune.DEV_TRIALS
        assert tune.load_best(dev=True)["lgbm__eui_kwh_m2"]["best_value"] == best["best_value"]
        assert tune.feature_set_for("lgbm") == "C-G"                    # no D5 file -> C-G


def check_feature_set_follows_d5_file():
    with _TmpOut() as out:
        (out / "benchmark").mkdir(parents=True)
        (out / "benchmark" / "feature_sets.json").write_text(json.dumps({"feature_sets": {"mlp": "C-S"}}))
        assert tune.feature_set_for("mlp") == "C-S" and tune.feature_set_for("lgbm") == "C-G"


def check_edge_flags():
    f = tune.edge_flags("lgbm", {"n_estimators": 2000, "learning_rate": 0.05, "num_leaves": 8, "subsample": 0.75})
    assert any("n_estimators at high end" in x for x in f) and any("num_leaves at low end" in x for x in f)
    assert not any("learning_rate" in x or "subsample" in x for x in f)


ALL_CHECKS = [check_budgets_from_config, check_scored_rmse_uses_heating_climates_for_heating_targets,
              check_objective_equals_independent_cv_rmse, check_mlp_objective_is_mean_normalised_primary_rmse,
              check_study_resumes_without_repeating_and_saves_outputs, check_feature_set_follows_d5_file,
              check_edge_flags]


def main():
    total = failed = 0
    print("=" * 72)
    print("SURROGATE TUNING (3.3) - CHECKS")
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
