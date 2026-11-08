"""
test_target_forms.py - checks for the 3.4 target forms (surrogate/models.py: TargetForm,
DecomposedEUI, OALevel, MLPPhysics), the fit context in surrogate/cv.py, and the C-G+I feature
matrix (Sub-task 3.4 step 2; claude/Phase3.4_Execution_Plan.md F2-F4).

The key checks: every form converts back exactly; the EUI decomposition adds up exactly to EUI;
the OA line is fitted on the training climates of the fold only (and equals an independent least-
squares fit); changing the held-out climate's targets cannot change its LOCO predictions (end-to-
end leakage test through the CV harness); the CV harness passes training rows only as context;
the physics MLP builds its 4 extra outputs correctly and its decomposed EUI is the sum of its parts.
Small, fast models are used: these checks test wiring, not accuracy. Training split only.

Run:  ./venv/bin/python surrogate/tests/test_target_forms.py   (or all suites: surrogate/run_tests.py)
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from surrogate import climate_indices as CI
from surrogate import cv, models
from surrogate.config import ALL_TARGETS, CONFIG

_CACHE: dict = {}
HEAT, COOL, EUI, PEAK = models.HEAT, models.COOL, models.EUI, models.PEAK_HEAT


def _data():
    if not _CACHE:
        df, _ = cv.load_training()
        X = CI.feature_matrix(df)
        _CACHE.update(df=df, X=X)
    return _CACHE["df"], _CACHE["X"]


def _lgbm_small():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=40, num_leaves=15, random_state=42, n_jobs=1, deterministic=True,
                         force_col_wise=True, verbose=-1)


class _Recorder(BaseEstimator, RegressorMixin):
    """Records what fit() receives; predicts the training mean."""
    seen: list = []

    def fit(self, X, y, context=None):
        _Recorder.seen.append({"n": len(X), "context": None if context is None else context.copy()})
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class _RecorderMulti(_Recorder):
    def fit(self, X, Y, groups=None, context=None):
        _Recorder.seen.append({"n": len(X), "groups": groups, "context": None if context is None else context.copy()})
        self.mean_ = np.asarray(Y, dtype=float).mean(axis=0)
        return self

    def predict(self, X):
        return np.tile(self.mean_, (len(X), 1))


# ----------------------------------------------------------------------------

def check_constants_match_config_and_indices():
    tf = CONFIG["target_forms"]
    assert models.ELEC == tf["eui_constant_kwh_m2"] == 48.87
    assert models.HDD_OFFSET == tf["hdd_offset"] == 50
    assert CI.HDD_BASES[0] == tf["hdd_base_c"] == 15 and models.HDD_COL == "hdd15" in CI.INDEX_COLUMNS
    sp = tf["index_setpoints"]
    assert (CI.T_HEAT, CI.T_COOL, CI.RH_ROOM) == (sp["heating_c"], sp["cooling_c"], sp["room_rh"])
    df, _ = _data()
    elec = df[EUI] - df[HEAT] - df[COOL]           # = 48.87 + OA term; OA can be negative
    assert np.isfinite(elec).all()


def check_feature_matrix_is_cg_plus_indices():
    from surrogate import features as FE
    df, X = _data()
    cg = FE.feature_set("C-G")
    assert len(cg) == 63 and list(X.columns) == list(cg) + CI.INDEX_COLUMNS and X.shape == (len(df), 70)
    assert X.index.equals(df.index)
    t = CI.training_indices()
    assert np.array_equal(X["hdd15"].to_numpy(), t.loc[df["climate_key"], "hdd15"].to_numpy())


def check_forms_convert_back_exactly():
    df, X = _data()
    for t in (HEAT, PEAK, COOL):
        y = df[t].to_numpy(dtype=float)
        for form in models.FORMS:
            back = models.from_form(models.to_form(y, form, X), form, X)
            assert np.allclose(back, y, rtol=1e-12, atol=1e-12), (t, form)
    y = df[HEAT].to_numpy(dtype=float)
    ratio = models.to_form(y, "ratio", X)
    assert np.allclose(ratio, y / (X["hdd15"].to_numpy() + 50), rtol=0, atol=0)
    for bad in (X.to_numpy(), X.drop(columns="hdd15")):
        try:
            models.to_form(y, "ratio", bad)
        except ValueError:
            pass
        else:
            raise AssertionError("ratio form accepted X without an hdd15 column")
    try:
        models.to_form(np.array([-1.5]), "log1p")
    except ValueError:
        pass
    else:
        raise AssertionError("log1p accepted y <= -1")


def check_target_form_equals_manual_fit():
    df, X = _data()
    tr, te = np.arange(0, 2000), np.arange(2000, 2300)
    y = df[HEAT].to_numpy(dtype=float)
    for form in ("log1p", "ratio"):
        w = models.TargetForm(_lgbm_small(), form).fit(X.iloc[tr], y[tr])
        manual = _lgbm_small().fit(X.iloc[tr], models.to_form(y[tr], form, X.iloc[tr]))
        want = models.from_form(manual.predict(X.iloc[te]), form, X.iloc[te])
        assert np.array_equal(w.predict(X.iloc[te]), want), form
    # make_form factory builds the same thing from the 3.3 settings
    f = models.make_form("lgbm", models.default_params("lgbm"), "ratio")(HEAT)
    assert isinstance(f, models.TargetForm) and f.form == "ratio"
    for bad in (("mlp", "ratio"), ("lgbm", "cube")):
        try:
            models.make_form(bad[0], models.default_params(bad[0]), bad[1])
        except ValueError:
            pass
        else:
            raise AssertionError(f"make_form accepted {bad}")


def check_decomposition_identity_exact():
    df, _ = _data()
    oa = models.oa_term(df[EUI], df[HEAT], df[COOL])
    mean = pd.Series(oa).groupby(df["climate_key"].to_numpy()).transform("mean").to_numpy()
    dev = oa - mean
    rebuilt = models.ELEC + df[HEAT].to_numpy() + df[COOL].to_numpy() + mean + dev
    assert np.allclose(rebuilt, df[EUI].to_numpy(), rtol=0, atol=1e-9)
    # within-climate deviations average to zero
    assert np.abs(pd.Series(dev).groupby(df["climate_key"].to_numpy()).mean()).max() < 1e-9


def check_decomposed_eui_wiring_with_exact_submodels():
    """1-nearest-neighbour sub-models reproduce their training targets exactly, so on the training
    rows the only difference from EUI is the OA line's error at each climate mean."""
    df, X = _data()
    rows = np.arange(0, 3000, 3)
    ctx = df.iloc[rows].reset_index(drop=True)
    knn = KNeighborsRegressor(n_neighbors=1)
    m = models.DecomposedEUI(heat=knn, cool=knn, dev=knn).fit(X.iloc[rows], ctx[EUI].to_numpy(), context=ctx)
    parts = m.predict_parts(X.iloc[rows])
    s = parts[["elec", "heating", "cooling", "oa_level", "oa_dev"]].sum(axis=1)
    assert np.allclose(parts["eui"], s, rtol=0, atol=1e-12) and np.array_equal(m.predict(X.iloc[rows]), parts["eui"].to_numpy())
    assert np.allclose(parts["heating"], ctx[HEAT], atol=1e-9) and np.allclose(parts["cooling"], ctx[COOL], atol=1e-9)
    clim = ctx["climate_key"].to_numpy()
    line_err = (m.level_.fitted_ - m.oa_climate_mean_).loc[clim].to_numpy()
    assert np.allclose(parts["eui"].to_numpy() - ctx[EUI].to_numpy(), line_err, atol=1e-9)
    # needs context, and index values constant per climate
    try:
        models.DecomposedEUI(heat=knn, cool=knn, dev=knn).fit(X.iloc[rows], ctx[EUI].to_numpy())
    except ValueError:
        pass
    else:
        raise AssertionError("fit without context accepted")
    Xbad = X.iloc[rows].copy()
    Xbad.iloc[0, Xbad.columns.get_loc("occ_enth_excess")] += 1.0
    try:
        models.DecomposedEUI(heat=knn, cool=knn, dev=knn).fit(Xbad, ctx[EUI].to_numpy(), context=ctx)
    except ValueError:
        pass
    else:
        raise AssertionError("index varying within a climate accepted")


def check_oa_line_uses_training_climates_only_and_equals_lstsq():
    df, X = _data()
    for fold in cv.loco_folds(df)[:3] + cv.grouped_folds(df)[:1]:
        ctx = df.iloc[fold.train].reset_index(drop=True)
        m = models.DecomposedEUI(heat=LinearRegression(), cool=LinearRegression(), dev=LinearRegression())
        m.fit(X.iloc[fold.train], ctx[EUI].to_numpy(), context=ctx)
        train_climates = sorted(set(ctx["climate_key"]))
        assert sorted(m.level_.climates_) == train_climates
        if fold.name in set(df["climate_key"]):
            assert fold.name not in m.level_.climates_
        # independent fit: per-climate means of the OA term, 3 indices + intercept, normal equations
        oa = (ctx[EUI] - 48.87 - ctx[HEAT] - ctx[COOL]).groupby(ctx["climate_key"]).mean()
        idx = CI.training_indices().loc[oa.index, CI.OA_LEVEL_INDICES].to_numpy()
        A = np.column_stack([np.ones(len(idx)), idx])
        coef = np.linalg.solve(A.T @ A, A.T @ oa.to_numpy())
        assert np.allclose(m.level_.coef_, coef, rtol=1e-6, atol=1e-9), (fold.name, m.level_.coef_, coef)


def check_cv_passes_training_rows_only_as_context():
    df, X = _data()
    for folds in (cv.loco_folds(df), cv.grouped_folds(df)):
        _Recorder.seen = []
        cv.cross_val_predict(df, X, [EUI], lambda t: _Recorder(), folds, "x")
        assert len(_Recorder.seen) == len(folds)
        for rec, f in zip(_Recorder.seen, folds):
            assert list(rec["context"]["run_id"]) == list(df["run_id"].iloc[f.train])
            assert not set(rec["context"]["run_id"]) & set(df["run_id"].iloc[f.test])
        _Recorder.seen = []
        cv.cross_val_predict_multi(df, X, ALL_TARGETS, lambda: _RecorderMulti(), folds, "x")
        for rec, f in zip(_Recorder.seen, folds):
            assert list(rec["context"]["run_id"]) == list(df["run_id"].iloc[f.train])
            assert list(rec["groups"]) == list(df["design_id"].astype(str).iloc[f.train])
    # estimators without a context argument are called exactly as before
    from sklearn.dummy import DummyRegressor
    out = cv.cross_val_predict(df, X, [COOL], lambda t: DummyRegressor(), cv.loco_folds(df), "x")
    assert len(out) == len(df)


def check_loco_predictions_ignore_held_out_targets():
    """End-to-end leakage test: scrambling the held-out climate's targets changes nothing."""
    df, X = _data()
    fac = lambda t: models.DecomposedEUI(heat=_lgbm_small(), cool=_lgbm_small(), dev=_lgbm_small())  # noqa: E731
    rng = np.random.default_rng(0)
    for f in [f for f in cv.loco_folds(df) if f.name in ("denver", "miami")]:   # one fold at a time:
        a = cv.cross_val_predict(df, X, [EUI], fac, [f], "loco")            # another fold's test rows
        df2 = df.copy()                                                     # are this fold's training rows
        for t in (EUI, HEAT, COOL):
            v = df2[t].to_numpy(copy=True)
            v[f.test] = rng.permutation(v[f.test]) * 3 + 50
            df2[t] = v
        b = cv.cross_val_predict(df2, X, [EUI], fac, [f], "loco")
        assert np.array_equal(a["yhat"].to_numpy(), b["yhat"].to_numpy()), f.name
        assert not np.array_equal(a["y"].to_numpy(), b["y"].to_numpy())


def check_make_decomposed_uses_one_family_and_eui_only():
    p = {t: models.default_params("lgbm") for t in (HEAT, COOL, EUI)}
    m = models.make_decomposed("lgbm", p)(EUI)
    assert all(type(x).__name__ == "LGBMRegressor" for x in (m.heat, m.cool, m.dev))
    for bad in (lambda: models.make_decomposed("lgbm", p)(HEAT), lambda: models.make_decomposed("mlp", p)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("make_decomposed accepted a wrong use")


def check_mlp_physics_outputs_and_forms():
    df, X = _data()
    rows = np.arange(0, 3000, 5)
    sub, Xs = df.iloc[rows].reset_index(drop=True), X.iloc[rows].reset_index(drop=True)
    tr, te = np.arange(0, 480), np.arange(480, 600)
    Y = sub[ALL_TARGETS].to_numpy()
    params = {**models.default_params("mlp"), "max_epochs": 5, "patience": 2, "width": 16, "n_layers": 1}
    m = models.MLPPhysics(params).fit(Xs.iloc[tr], Y[tr], groups=sub["design_id"].iloc[tr].to_numpy(),
                                      context=sub.iloc[tr])
    assert m.output_names_ == list(ALL_TARGETS) + [n for n, _, _ in models.MLP_EXTRA] and len(m.output_names_) == 14
    # extra training columns equal hand-computed values
    ctx = sub.iloc[tr]
    extra = m.extra_outputs(Xs.iloc[tr], pd.DataFrame(Y[tr], columns=ALL_TARGETS), ctx["climate_key"])
    scale = Xs["hdd15"].iloc[tr].to_numpy() + 50
    oa = ctx[EUI] - 48.87 - ctx[HEAT] - ctx[COOL]
    dev = (oa - oa.groupby(ctx["climate_key"]).transform("mean")).to_numpy()
    want = np.column_stack([ctx[HEAT] / scale, np.log1p(ctx[HEAT]), ctx[PEAK] / scale, dev])
    assert np.allclose(extra, want, rtol=1e-12, atol=1e-12)
    # predictions: direct block = predict(); forms rebuilt from the raw outputs
    raw, forms = m.predict_raw(Xs.iloc[te]), m.predict_forms(Xs.iloc[te])
    assert np.array_equal(m.predict(Xs.iloc[te]), raw[list(ALL_TARGETS)].to_numpy())
    assert forms.shape == (len(te), 14) and forms.columns.names == ["target", "form"]
    s = Xs["hdd15"].iloc[te].to_numpy() + 50
    assert np.allclose(forms[(HEAT, "ratio")], raw["heating_ratio"] * s)
    assert np.allclose(forms[(HEAT, "log1p")], np.expm1(raw["heating_log1p"]))
    assert np.allclose(forms[(PEAK, "ratio")], raw["peak_heating_ratio"] * s)
    eui = 48.87 + forms[(HEAT, "ratio")] + raw[COOL] + m.level_.predict(Xs.iloc[te]) + raw["oa_dev"]
    assert np.allclose(forms[(EUI, "decomposed")], eui)
    # through the CV harness (context + groups passed automatically)
    out = cv.cross_val_predict_multi(sub, Xs, ALL_TARGETS, lambda: models.MLPPhysics(params),
                                     cv.grouped_folds(sub, n_folds=2), "x")
    assert len(out) == len(sub) * len(ALL_TARGETS) and np.isfinite(out["yhat"]).all()


ALL_CHECKS = [check_constants_match_config_and_indices, check_feature_matrix_is_cg_plus_indices,
              check_forms_convert_back_exactly, check_target_form_equals_manual_fit,
              check_decomposition_identity_exact, check_decomposed_eui_wiring_with_exact_submodels,
              check_oa_line_uses_training_climates_only_and_equals_lstsq, check_cv_passes_training_rows_only_as_context,
              check_loco_predictions_ignore_held_out_targets, check_make_decomposed_uses_one_family_and_eui_only,
              check_mlp_physics_outputs_and_forms]


def main():
    total = failed = 0
    print("=" * 72)
    print("SURROGATE TARGET FORMS (3.4) - CHECKS")
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
