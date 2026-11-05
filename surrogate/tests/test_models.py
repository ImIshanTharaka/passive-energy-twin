"""
test_models.py - checks for surrogate/models.py (Sub-task 3.3 step 2).

Settings match config.yaml and the library defaults; Optuna suggestions stay inside the frozen
ranges; every factory fits and predicts real C-G rows; fixed seeds give identical predictions;
the GPR learns its scaler and feature selection from training rows only. Training split only.

Run:  ./venv/bin/python surrogate/tests/test_models.py   (or all suites: surrogate/run_tests.py)
"""

from __future__ import annotations

import json
import sys

import numpy as np
import optuna

from surrogate import ablation as A
from surrogate import cv, features as FE, models
from surrogate.config import CONFIG, ALL_TARGETS

optuna.logging.set_verbosity(optuna.logging.WARNING)
CFG = CONFIG["benchmark_3_3"]
_REAL: dict = {}


def _real(n=300):
    """First n training rows (C-G features) and their EUI - small but real."""
    if not _REAL:
        df, _ = cv.load_training()
        X = FE.engineer(df)[FE.feature_set(CFG["feature_set"])]
        _REAL.update(df=df, X=X)
    df, X = _REAL["df"], _REAL["X"]
    return df.iloc[:n], X.iloc[:n], df["eui_kwh_m2"].to_numpy()[:n]


# ── settings ────────────────────────────────────────────────────────────────────────────────

def check_model_list_and_budgets():
    assert models.MODELS == ("rf", "xgb", "lgbm", "gpr", "mlp")
    assert CFG["tuning"]["trials"] == {"rf": 60, "xgb": 60, "lgbm": 60, "mlp": 30, "gpr": 0}
    assert models.TUNABLE == ("rf", "xgb", "lgbm", "mlp")
    assert CFG["feature_set"] == "C-G" and len(FE.feature_set("C-G")) == 63
    assert CFG["fresh_fold_seed"] != CONFIG["protocol"]["seed"]


def check_defaults_match_earlier_steps_and_libraries():
    assert models.default_params("rf") == CONFIG["reference_2_8"]["rf_params"] == A.model_params("rf")
    assert models.default_params("lgbm") == A.model_params("lgbm")
    # XGBoost: written-out values = what the library actually uses (booster config after a fit)
    import xgboost
    m = xgboost.XGBRegressor().fit(np.random.default_rng(0).random((40, 3)), np.arange(40.0))
    c = json.loads(m.get_booster().save_config())["learner"]["gradient_booster"]
    tp = c["tree_train_param"]
    d = models.default_params("xgb")
    assert int(c["gbtree_model_param"]["num_trees"]) == d["n_estimators"]
    for ours, lib in (("learning_rate", "eta"), ("max_depth", "max_depth"), ("min_child_weight", "min_child_weight"),
                      ("subsample", "subsample"), ("colsample_bytree", "colsample_bytree"),
                      ("reg_lambda", "lambda"), ("reg_alpha", "alpha"), ("max_bin", "max_bin")):
        assert abs(float(tp[lib]) - float(d[ours])) < 1e-6, f"xgb {ours}: config {d[ours]} vs library {tp[lib]}"
    assert models.default_params("gpr")["nu"] == 2.5 and models.default_params("gpr")["top_k"] is None


def check_suggestions_stay_inside_frozen_ranges():
    for model in models.TUNABLE:
        space = models.search_space(model)
        seen = []
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
        study.optimize(lambda tr: seen.append(models.suggest(tr, model)) or 0.0, n_trials=40)
        for p in seen:
            for name, s in space.items():
                v = p[name]
                if s["type"] == "fixed":
                    assert v == s["value"]
                elif s["type"] == "categorical":
                    assert v in s["choices"], (model, name, v)
                else:
                    assert s["low"] <= v <= s["high"], (model, name, v)
                    assert isinstance(v, int) == (s["type"] == "int"), (model, name, type(v))
            for k, v in CFG["fixed"].get(model, {}).items():
                assert p[k] == v, (model, k)
    try:
        models.search_space("gpr")
    except KeyError:
        pass
    else:
        raise AssertionError("gpr must have no Optuna search space (E5)")


def check_fixed_values_cannot_be_tuned():
    try:
        models.tuned_params("lgbm", {"random_state": 1})
    except ValueError:
        return
    raise AssertionError("a fixed setting was overridden")


def check_unknown_model_rejected():
    for bad in ("svm", "LGBM"):
        try:
            models.default_params(bad)
        except KeyError:
            continue
        raise AssertionError(f"{bad} accepted")


# ── factories on real rows ──────────────────────────────────────────────────────────────────

def _small(model):
    p = models.default_params(model)
    if model == "rf":
        p["n_estimators"] = 30
    if model == "mlp":
        p.update(max_epochs=30)
    return p


def check_every_factory_fits_and_predicts():
    df, X, y = _real(300)
    tr, te = np.arange(240), np.arange(240, 300)
    for model in models.MODELS:
        est = models.make(model, _small(model))("eui_kwh_m2")
        kw = {"groups": df["design_id"].to_numpy()[tr]} if model == "mlp" else {}
        est.fit(X.iloc[tr], y[tr], **kw)
        yhat = np.asarray(est.predict(X.iloc[te]))
        assert yhat.shape == (60,) and np.isfinite(yhat).all(), model
        assert np.corrcoef(yhat, y[te])[0, 1] > 0.5, f"{model} predictions unrelated to the truth"


def check_multi_output_factory():
    df, X, _ = _real(300)
    Y = df[ALL_TARGETS].to_numpy()
    est = models.make_multi("mlp", _small("mlp"))()
    est.fit(X, Y, groups=df["design_id"].to_numpy())
    assert est.predict(X.iloc[:7]).shape == (7, 10)
    try:
        models.make_multi("lgbm", {})
    except ValueError:
        return
    raise AssertionError("single-output model accepted by make_multi")


def check_fixed_seeds_reproduce():
    df, X, y = _real(300)
    for model in ("rf", "xgb", "lgbm", "mlp"):
        p = _small(model)
        if model in models.TUNABLE:                       # tuned path adds the fixed seeds
            p = {**p, **CFG["fixed"].get(model, {})}
        a = models.make(model, p)("y").fit(X, y).predict(X.iloc[:50])
        b = models.make(model, p)("y").fit(X, y).predict(X.iloc[:50])
        # RF: parallel trees are summed in thread order -> differences ~1e-14 (same tolerance as test_cv)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9 if model == "rf" else 0, err_msg=f"{model} not reproducible")


def check_gpr_learns_only_from_training_rows():
    _, X, y = _real(200)
    tr = np.arange(150)
    g = models.GPR(**{**models.default_params("gpr"), "top_k": 5}).fit(X.iloc[tr], y[tr])
    np.testing.assert_allclose(g.scaler_.mean_, X.iloc[tr].to_numpy().mean(0))
    assert len(g.selected_) == 5 and len(g.selected_features_) == 5
    assert set(g.selected_features_) <= set(X.columns)
    # changing the non-training rows changes nothing in the fitted model
    X2 = X.copy()
    X2.iloc[150:] = X2.iloc[150:] * 3.0
    g2 = models.GPR(**{**models.default_params("gpr"), "top_k": 5}).fit(X2.iloc[tr], y[tr])
    assert list(g2.selected_) == list(g.selected_)
    np.testing.assert_allclose(g2.predict(X.iloc[150:]), g.predict(X.iloc[150:]), rtol=1e-10)
    mu, sd = g.predict(X.iloc[150:], return_std=True)
    assert mu.shape == sd.shape == (50,) and (sd > 0).all()
    assert g.length_scales_.shape == (5,) and np.isfinite(g.noise_level_)


ALL_CHECKS = [check_model_list_and_budgets, check_defaults_match_earlier_steps_and_libraries,
              check_suggestions_stay_inside_frozen_ranges, check_fixed_values_cannot_be_tuned,
              check_unknown_model_rejected, check_every_factory_fits_and_predicts, check_multi_output_factory,
              check_fixed_seeds_reproduce, check_gpr_learns_only_from_training_rows]


def main():
    total = failed = 0
    print("=" * 72)
    print("SURROGATE MODELS (3.3) - CHECKS")
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
