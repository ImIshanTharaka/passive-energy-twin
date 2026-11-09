"""
test_climate_gen.py - checks for surrogate/climate_gen.py (Sub-task 3.4 step 4).

The key checks: the decomposed EUI assembled from out-of-fold parts equals models.DecomposedEUI
refitted per fold; the selection rule (M10 with the 3.4 tie-break) on hand-made candidate tables;
the fix-4 trigger (F5) and the M7 trigger end to end through `stage_select`; climate distances;
the reduced input set of fix 4; the MLP out-of-fold frame. Small models, training split only.

Run:  ./venv/bin/python surrogate/tests/test_climate_gen.py   (or all suites: surrogate/run_tests.py)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from surrogate import climate_gen as CG
from surrogate import climate_indices as CI
from surrogate import cv, models
from surrogate.config import ALL_TARGETS, CONFIG

HEAT, COOL, EUI, PEAK = models.HEAT, models.COOL, models.EUI, models.PEAK_HEAT
_CACHE: dict = {}
SMALL = {"n_estimators": 40, "num_leaves": 15, "learning_rate": 0.1, "random_state": 42, "n_jobs": 1,
         "deterministic": True, "force_col_wise": True, "verbose": -1}


def _data():
    if not _CACHE:
        df, _ = cv.load_training()
        keep = set(sorted(df["design_id"].astype(str).unique())[::4])
        df = df[df["design_id"].astype(str).isin(keep)].reset_index(drop=True)
        _CACHE.update(df=df, X=CI.feature_matrix(df))
    return _CACHE["df"], _CACHE["X"]


def _cand(model, form, target, skill, mape, bias=1.0, fs="C-G+I"):
    return {"model": model, "feature_set": fs, "form": form, "target": target, "grouped_design_skill": skill,
            "grouped_r2": 0.99, "loco_mape_pct": mape, "loco_median_abs_bias_pct": bias,
            "loco_max_abs_bias_pct": bias * 2, "loco_worst_climate": "denver", "loco_bias_removed_r2": 0.9,
            "loco_spearman_median": 0.95}


# ----------------------------------------------------------------------------

def check_composed_decomposed_equals_refitted_model():
    df, X = _data()
    pbt = {t: dict(SMALL) for t in (HEAT, COOL, EUI)}
    folds = [f for f in cv.loco_folds(df) if f.name in ("chicago", "san_francisco")] + cv.grouped_folds(df)[:2]
    for scheme, fs in (("loco", folds[:2]), ("grouped", folds[2:])):
        oof, feats = CG.single_family_oof("lgbm", pbt, df, X, fs, scheme, log=lambda s: None)
        assert feats == {}
        got = oof[(oof["target"] == EUI) & (oof["form"] == "decomposed")].sort_values(["fold", "run_id"])
        ref = cv.cross_val_predict(df, X, [EUI], models.make_decomposed("lgbm", pbt), fs, scheme)
        ref = ref.sort_values(["fold", "run_id"])
        assert np.array_equal(got["run_id"].to_numpy(), ref["run_id"].to_numpy())
        assert np.allclose(got["yhat"].to_numpy(), ref["yhat"].to_numpy(), rtol=0, atol=1e-9), scheme
        forms = set(map(tuple, oof[["target", "form"]].drop_duplicates().to_numpy()))
        assert forms == {(HEAT, "direct"), (HEAT, "log1p"), (HEAT, "ratio"), (COOL, "direct"),
                         (EUI, "direct"), (EUI, "decomposed")}
        n_test = sum(len(f.test) for f in fs)
        assert (oof.groupby(["target", "form"]).size() == n_test).all()


def check_fix4_runs_only_needed_parts():
    df, X = _data()
    pbt = {t: dict(SMALL) for t in (HEAT, COOL, EUI)}
    fs = cv.grouped_folds(df)[:2]
    full, _ = CG.single_family_oof("lgbm", pbt, df, X, fs, "grouped", log=lambda s: None)
    for cand in ((HEAT, "log1p"), (EUI, "decomposed"), (COOL, "direct")):
        part, _ = CG.single_family_oof("lgbm", pbt, df, X, fs, "grouped", log=lambda s: None, only=[cand])
        got = set(map(tuple, part[["target", "form"]].drop_duplicates().to_numpy()))
        want = {cand} | ({(HEAT, "ratio"), (COOL, "direct")} if cand == (EUI, "decomposed") else set())
        assert got == want, (cand, got)
        a = part[(part["target"] == cand[0]) & (part["form"] == cand[1])].sort_values(["fold", "run_id"])
        b = full[(full["target"] == cand[0]) & (full["form"] == cand[1])].sort_values(["fold", "run_id"])
        assert np.array_equal(a["yhat"].to_numpy(), b["yhat"].to_numpy()), cand


def check_selection_rule():
    t = models.HEAT
    # lowest MAPE among skill >= 0.90; a better MAPE below the bar is ignored
    c = pd.DataFrame([_cand("mlp", "ratio", t, 0.99, 10.0), _cand("gpr", "direct", t, 0.89, 5.0),
                      _cand("xgb", "ratio", t, 0.95, 12.0)])
    r = CG.select_target(c, t)
    assert (r["model"], r["form"]) == ("mlp", "ratio") and not r["gate_fail"]
    # tie within 0.5 points -> the LightGBM candidate with the BEST value (not the first one listed)
    c = pd.DataFrame([_cand("mlp", "ratio", t, 0.99, 10.0), _cand("lgbm", "direct", t, 0.95, 10.45),
                      _cand("lgbm", "ratio", t, 0.96, 10.2), _cand("lgbm", "log1p", t, 0.96, 10.6)])
    r = CG.select_target(c, t)
    assert (r["model"], r["form"]) == ("lgbm", "ratio") and "LightGBM" in r["rule"], r
    assert r["best_by_value"]["model"] == "mlp"
    # nobody passes -> best skill + gate fail
    c = pd.DataFrame([_cand("mlp", "ratio", t, 0.85, 10.0), _cand("rf", "direct", t, 0.88, 30.0)])
    r = CG.select_target(c, t)
    assert r["model"] == "rf" and r["gate_fail"]
    # secondary: highest grouped skill, tie 0.005 -> LightGBM
    c = pd.DataFrame([_cand("mlp", "direct", PEAK, 0.982, 9.0), _cand("lgbm", "ratio", PEAK, 0.979, 8.0),
                      _cand("lgbm", "direct", PEAK, 0.973, 8.0)])
    r = CG.select_target(c, PEAK)
    assert (r["model"], r["form"]) == ("lgbm", "ratio"), r


def _run_select(grid_rows, fix4_rows=None):
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        pd.DataFrame(grid_rows).to_csv(d / "grid_table.csv", index=False)
        if fix4_rows is not None:
            pd.DataFrame(fix4_rows).to_csv(d / "fix4_table.csv", index=False)
        old_out, old_base, old_rec = CG.out_dir, CG.baseline_pool, CG.record_stage
        CG.out_dir = lambda dev: d
        CG.baseline_pool = lambda dev, log=print: pd.DataFrame()
        CG.record_stage = lambda *a, **k: None
        try:
            res = CG.stage_select(True, log=lambda s: None)
            saved = json.loads((d / "selection.json").read_text())
            report = (d / "climate_gen_report.md").read_text()
        finally:
            CG.out_dir, CG.baseline_pool, CG.record_stage = old_out, old_base, old_rec
    assert saved["status"] == res["status"] and "Selection" in report
    return res


def check_fix4_and_m7_triggers_end_to_end():
    base = [_cand("mlp", "ratio", HEAT, 0.99, 14.0, bias=4.0), _cand("mlp", "direct", COOL, 0.99, 3.0, bias=1.0),
            _cand("mlp", "decomposed", EUI, 0.98, 5.0, bias=3.0), _cand("mlp", "direct", PEAK, 0.98, 9.0)]
    r = _run_select(base)
    assert r["status"] == "final" and r["fix4"]["needed_for"] == [] and r["m7"]["triggered"] is False
    assert r["families"] == ["mlp"] and r["one_family"]
    # heating bias 7 % > 5 % -> fix 4 needed, M7 not concluded until it has run
    rows = [dict(x) for x in base]
    rows[0]["loco_median_abs_bias_pct"] = 7.0
    r = _run_select(rows)
    assert r["status"] == "awaiting_fix4" and r["m7"] is None
    assert [(n["target"], n["model"], n["form"]) for n in r["fix4"]["pending"]] == [(HEAT, "mlp", "ratio")]
    # after fix 4: worse fix-4 candidate -> original stays selected, M7 evaluated (7 % < 10 %)
    fix = [_cand("mlp", "ratio", HEAT, 0.99, 15.0, bias=6.0, fs=CG.FS_FIX4)]
    r = _run_select(rows, fix)
    assert r["status"] == "final" and r["selection"][HEAT]["feature_set"] == "C-G+I"
    assert r["m7"]["triggered"] is False and abs(r["m7"]["values"][HEAT] - 7.0) < 1e-12
    # EUI bias 12 % (fix 4 run, still 11 %) -> M7 triggered
    rows[2]["loco_median_abs_bias_pct"] = 12.0
    fix = fix + [_cand("mlp", "decomposed", EUI, 0.98, 6.0, bias=11.0, fs=CG.FS_FIX4)]
    r = _run_select(rows, fix)
    assert r["status"] == "final" and r["m7"]["triggered"] is True
    assert CONFIG["m7_trigger"]["loco_median_abs_bias_pct"] == r["m7"]["limit"] == 10.0
    # mixed families recorded
    rows[1]["model"] = "lgbm"
    r = _run_select(rows, fix)
    assert r["families"] == ["lgbm", "mlp"] and not r["one_family"]


def check_climate_distances_and_fix4_inputs():
    d = CG.climate_distances()
    assert len(d) == 10 and (d["distance_to_nearest"] > 0).all()
    from scipy.spatial.distance import cdist
    from surrogate import features as FE
    full, _ = cv.load_training()
    c = full.groupby("climate_key")[FE.CLIMATE_INPUTS].first()
    z = (c - c.mean()) / c.std(ddof=0)
    dist = cdist(z.to_numpy(), z.to_numpy())
    np.fill_diagonal(dist, np.inf)
    assert np.allclose(d.loc[c.index, "distance_to_nearest"].to_numpy(), dist.min(axis=1))
    assert list(d.loc[c.index, "nearest_climate"]) == list(c.index[dist.argmin(axis=1)])
    df, X = _data()
    cols = CG.reduced_columns(X)
    assert len(cols) == X.shape[1] - 3 and not {"mean_dbt", "latitude", "mean_wind"} & set(cols)
    assert all(c in cols for c in CI.INDEX_COLUMNS + ["hdd18", "cdd10", "annual_ghi", "mean_rh"])


def check_mlp_oof_frame():
    df, X = _data()
    sub = df.iloc[:300].reset_index(drop=True)
    Xs = X.iloc[:300].reset_index(drop=True)
    params = {**models.default_params("mlp"), "max_epochs": 3, "patience": 2, "width": 16, "n_layers": 1}
    folds = cv.grouped_folds(sub, n_folds=2)
    oof = CG.mlp_oof(params, sub, Xs, folds, "grouped")
    got = set(map(tuple, oof[["target", "form"]].drop_duplicates().to_numpy()))
    assert got == {(HEAT, "direct"), (HEAT, "ratio"), (HEAT, "log1p"), (COOL, "direct"), (EUI, "direct"),
                   (EUI, "decomposed"), (PEAK, "direct"), (PEAK, "ratio")}, got
    assert (oof.groupby(["target", "form"]).size() == len(sub)).all() and np.isfinite(oof["yhat"]).all()
    y = oof[(oof["target"] == EUI)].groupby("run_id")["y"].nunique()
    assert (y == 1).all()


def check_scale_match_and_sensitivity_config():
    import hashlib
    from surrogate.config import CONFIG_PATH
    df, X = _data()
    y = df[HEAT].to_numpy(dtype=float)
    tr, te = np.arange(0, 500), np.arange(500, 700)
    w = models.TargetForm(models.make("lgbm", dict(SMALL))(HEAT), "ratio", scale_match=True).fit(X.iloc[tr], y[tr])
    z = models.to_form(y[tr], "ratio", X.iloc[tr])
    s = np.std(y[tr]) / np.std(z)
    assert abs(w.scale_ - s) < 1e-12 and 100 < s < 1e5           # ratio is ~1/1000 of heating
    manual = models.make("lgbm", dict(SMALL))(HEAT).fit(X.iloc[tr], z * s)
    want = models.from_form(manual.predict(X.iloc[te]) / s, "ratio", X.iloc[te])
    assert np.array_equal(w.predict(X.iloc[te]), want)
    # default (no scale match) unchanged
    assert models.TargetForm(models.make("lgbm", dict(SMALL))(HEAT), "ratio").fit(X.iloc[tr], y[tr]).scale_ == 1.0
    # config: the 3.4 block above the sensitivity block is byte-identical to the frozen 8 Nov file
    text = CONFIG_PATH.read_text(encoding="utf-8")
    marker = "\n# ── Added 9 Nov 2026, AFTER the final 3.4 selection"
    assert text.count(marker) == 1
    assert hashlib.md5(text[: text.index(marker)].encode("utf-8")).hexdigest() == "f62456b360ea36c649efd33cd068b59a"
    sc = CONFIG["sensitivity_3_4"]
    assert sc["model"] == "lgbm" and sc["enters_selection"] is False
    assert [tuple(c) for c in sc["candidates"]] == [(HEAT, "ratio"), (PEAK, "ratio"), (EUI, "decomposed")]


def check_cli_guards():
    try:
        CG.main(["--stage", "grid", "--models", "lgbm"])
    except SystemExit as e:
        assert e.code == 2
    else:
        raise AssertionError("--models accepted without --dev")
    assert CG.FS == "C-G+I" and CG.FS_FIX4 == "C-G+I-r"
    assert CG.CFG["models"] == list(models.MODELS)


ALL_CHECKS = [check_composed_decomposed_equals_refitted_model, check_fix4_runs_only_needed_parts, check_selection_rule,
              check_fix4_and_m7_triggers_end_to_end, check_climate_distances_and_fix4_inputs,
              check_mlp_oof_frame, check_scale_match_and_sensitivity_config, check_cli_guards]


def main():
    total = failed = 0
    print("=" * 72)
    print("SURROGATE CLIMATE GENERALISATION (3.4) - CHECKS")
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
