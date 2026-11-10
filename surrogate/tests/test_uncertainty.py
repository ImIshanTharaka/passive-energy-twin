"""
test_uncertainty.py - checks for surrogate/uncertainty.py (Sub-task 3.5 step 1).

Key checks: the finite-sample conformal quantile; ~90 % coverage on synthetic data for plain and
normalised conformal, and that normalised intervals adapt to difficulty where plain ones do not;
cross-fitting and the nested evaluation never score a row with a calibration that saw it; lower-bound
clipping and the GPR mappings; the G2 / G5 selection rules; the design-space check against
design_space.yaml and the training data; climate distances identical to climate_gen; D_max; levels;
the London guard; persistence round trip.

Run:  ./venv/bin/python surrogate/tests/test_uncertainty.py   (or all suites: surrogate/run_tests.py)
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from surrogate import climate_indices as CI
from surrogate import features as FE
from surrogate import uncertainty as U

_CACHE: dict = {}


def _checker() -> U.DomainChecker:
    if "dom" not in _CACHE:
        _CACHE["dom"] = U.DomainChecker.from_training()
    return _CACHE["dom"]


def _synthetic(n, rng, n_groups=200):
    """y = f(x) + noise with SD 0.05 + x1 (heteroscedastic); yhat = f(x) (an unbiased model)."""
    X = pd.DataFrame({"x1": rng.uniform(0, 1, n), "x2": rng.uniform(0, 1, n)})
    f = 3 * X["x2"].to_numpy() + 1
    y = f + rng.normal(0, 0.05 + X["x1"].to_numpy(), n)
    return X, f, y, rng.integers(0, n_groups, n).astype(str)


# ----------------------------------------------------------------------------

def check_quantile_index():
    s = np.arange(1, 20, dtype=float)                    # n = 19 -> k = ceil(20 * 0.9) = 18
    assert U.conformal_quantile(s, 0.9) == 18.0
    assert U.conformal_quantile(np.arange(1, 10.0), 0.9) == 9.0      # n = 9 -> k = 9 = n
    assert U.conformal_quantile(np.arange(1, 9.0), 0.9) == float("inf")   # n = 8 -> k = 9 > n
    assert U.conformal_quantile(s[::-1], 0.5) == 10.0    # order of input irrelevant
    for bad in ([], [1.0, np.nan]):
        try:
            U.conformal_quantile(bad, 0.9)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad}")


def check_plain_coverage_exchangeable():
    """Split conformal: mean coverage >= 0.90 and close to it (n = 99 -> exact 0.90)."""
    rng = np.random.default_rng(0)
    covs = []
    for _ in range(400):
        r_cal = rng.standard_t(4, 99)
        r_new = rng.standard_t(4, 200)
        conf = U.Conformal(scale="none", lower=None).fit(None, np.zeros(99), r_cal)
        lo, hi = conf.interval(None, np.zeros(200))
        covs.append(np.mean((r_new >= lo) & (r_new <= hi)))
    m = float(np.mean(covs))
    assert 0.89 <= m <= 0.915, m


def check_normalised_coverage_adapts():
    rng = np.random.default_rng(1)
    Xc, fc, yc, gc = _synthetic(3000, rng)
    Xt, ft, yt, _ = _synthetic(6000, rng)
    norm = U.Conformal(scale="lgbm", lower=None).fit(Xc, fc, yc, groups=gc)
    plain = U.Conformal(scale="none", lower=None).fit(Xc, fc, yc)
    hard = Xt["x1"].to_numpy() > 0.5
    res = {}
    for name, c in (("norm", norm), ("plain", plain)):
        lo, hi = c.interval(Xt, ft)
        inside = (yt >= lo) & (yt <= hi)
        res[name] = (inside.mean(), inside[~hard].mean(), inside[hard].mean(), float(np.mean(hi - lo)))
    n_all, n_easy, n_hard, n_w = res["norm"]
    p_all, p_easy, p_hard, p_w = res["plain"]
    assert 0.88 <= n_all <= 0.92 and 0.88 <= p_all <= 0.92, (res)
    assert 0.85 <= n_easy <= 0.95 and 0.85 <= n_hard <= 0.95, res          # adapts
    assert p_easy > 0.95 and p_hard < 0.85, res                             # plain does not
    assert n_w < p_w, res                                                    # and is narrower


def check_inner_splits_partition():
    many = np.repeat([f"d{i}" for i in range(50)], 4)
    few = np.repeat([f"c{i}" for i in range(9)], 30)
    for g, n_splits in ((many, U.INNER_SPLITS), (few, 9)):
        sp = U.inner_splits(g)
        assert len(sp) == n_splits
        scored = np.concatenate([te for _, te in sp])
        assert len(scored) == len(g) and len(np.unique(scored)) == len(g)   # each row scored once
        for tr, te in sp:
            assert not set(g[tr]) & set(g[te])
    try:
        U.inner_splits(np.array(["a"] * 5))
    except ValueError:
        pass
    else:
        raise AssertionError("one group accepted")


def _oof_frames(rng):
    """Small OOF frames shaped like the saved files: 60 designs x 4 climates; grouped folds by design."""
    designs = [f"A{i:03d}" for i in range(60)]
    climates = ["atlanta", "chicago", "denver", "miami"]
    fold_of = {d: str(i % 5) for i, d in enumerate(designs)}
    rows = [{"run_id": f"main_{d}_{c}", "design_id": d, "climate_key": c} for d in designs for c in climates]
    base = pd.DataFrame(rows)
    X = pd.DataFrame({"x1": rng.uniform(0, 1, len(base)), "x2": rng.uniform(0, 1, len(base))},
                     index=base["run_id"])
    y = 2 + X["x2"].to_numpy() + rng.normal(0, 0.1 + X["x1"].to_numpy())
    g = base.assign(fold=base["design_id"].map(fold_of), y=y, yhat=2 + X["x2"].to_numpy())
    loco = base.assign(fold=base["climate_key"], y=y,
                       yhat=2 + X["x2"].to_numpy() + base["climate_key"].map(
                           {"atlanta": 0.3, "chicago": -0.2, "denver": 0.1, "miami": -0.4}).to_numpy())
    return g, loco, X


def check_nested_never_calibrates_on_scored_rows():
    rng = np.random.default_rng(2)
    g, loco, X = _oof_frames(rng)
    small = dict(U.SCALE_PARAMS, n_estimators=20, min_child_samples=5)
    old = U.SCALE_PARAMS.copy()
    U.SCALE_PARAMS.update(small)
    try:
        cases = (("standard grouped", g, g, "fold", "design_id"),
                 ("new-site LOCO", loco, loco, "climate_key", "climate_key"),
                 ("standard in LOCO", g, loco, "climate_key", "design_id"))
        for name, cal, test, outer, inner in cases:
            for scale in ("lgbm", "none"):
                iv = U.nested_intervals(cal, test, X, outer=outer, inner_group=inner, scale=scale)
                assert len(iv) == len(test) and set(iv["run_id"]) == set(test["run_id"]), name
                for v, part in iv.groupby(outer if outer in iv else "fold"):
                    expected = int((cal[outer].astype(str) != str(v)).sum())
                    assert (part["n_cal"] == expected).all(), (name, v)
                    if outer == "fold":   # grouped: calibration designs disjoint from the scored designs
                        cal_designs = set(cal.loc[cal["fold"] != v, "design_id"])
                        assert not cal_designs & set(part["design_id"]), name
                assert (iv["lo_90"] <= iv["hi_90"]).all() and (iv["lo_90"] >= 0).all()
                assert (iv["hi_95"] - iv["lo_95"] >= iv["hi_90"] - iv["lo_90"] - 1e-12).all()
        # a deliberate leak (a scored row also in the calibration under another outer value) is refused
        leak = pd.concat([g, g.iloc[[0]].assign(fold="9")], ignore_index=True)
        try:
            U.nested_intervals(leak.drop_duplicates("run_id", keep="last"), g, X, outer="fold",
                               inner_group="design_id", scale="none")
        except AssertionError:
            pass
        else:
            raise AssertionError("leaking calibration row accepted")
    finally:
        U.SCALE_PARAMS.clear()
        U.SCALE_PARAMS.update(old)


def check_clipping_and_gpr_mappings():
    conf = U.Conformal(scale="none").fit(None, np.zeros(19), np.arange(1, 20.0))
    lo, hi = conf.interval(None, np.array([0.5, 30.0]))
    assert lo[0] == 0.0 and lo[1] == 30.0 - 18.0 and hi[0] == 18.5          # clipped at 0
    mu, sd = np.array([np.log1p(10.0), np.log1p(0.2)]), np.array([0.1, 0.5])
    lo, hi = U.gpr_native_interval(mu, sd, 0.90)
    z = 1.6448536269514722
    assert np.allclose(hi, np.expm1(mu + z * sd)) and np.allclose(lo, np.maximum(np.expm1(mu - z * sd), 0))
    h = 1e-6                                                                # delta method = derivative
    num = (np.expm1(mu + h) - np.expm1(mu - h)) / (2 * h) * sd
    assert np.allclose(U.gpr_sigma_original(mu, sd), num, rtol=1e-6)
    given = U.Conformal(scale="given").fit(None, np.zeros(19), np.arange(1, 20.0), sigma=np.full(19, 2.0))
    assert given.quantile() == 9.0
    lo, hi = given.interval(None, np.array([20.0]), sigma=np.array([1.0]))
    assert lo[0] == 11.0 and hi[0] == 29.0
    for bad in (np.zeros(19), np.full(18, 1.0)):
        try:
            U.Conformal(scale="given").fit(None, np.zeros(19), np.ones(19), sigma=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("bad sigma accepted")


def _pc(covs):
    return pd.DataFrame({"climate_key": [f"c{i}" for i in range(len(covs))], "coverage_90": covs})


def check_selection_rules():
    assert U.choose_scale(_pc([0.9, 0.9, 0.8]), _pc([0.9, 0.8, 0.8])) == "lgbm"     # 1 vs 2 outside
    assert U.choose_scale(_pc([0.9, 0.9, 0.9]), _pc([0.9, 0.9, 0.9])) == "lgbm"     # tie -> normalised
    assert U.choose_scale(_pc([0.8, 0.97, 0.9]), _pc([0.9, 0.9, 0.84])) == "none"   # 2 vs 1
    conf = pd.DataFrame({"mean_width_90": [10.0]})
    assert U.choose_heating(conf, pd.DataFrame({"mean_width_90": [8.9]}), _pc([0.9] * 7)) == "given"
    assert U.choose_heating(conf, pd.DataFrame({"mean_width_90": [9.1]}), _pc([0.9] * 7)) == "conformal"
    assert U.choose_heating(conf, pd.DataFrame({"mean_width_90": [5.0]}), _pc([0.9] * 6 + [0.84])) == "conformal"


def check_design_bounds_and_training_rows():
    b = U.design_bounds()
    assert list(b.index) == FE.DESIGN_INPUTS and len(b) == 21
    assert b.loc["floor_area", "min"] == 1500 and b.loc["floor_area", "max"] == 10000
    assert b.loc["n_floors", "integer"] and b.loc["wwr_north", "tol"] == 0.00005
    dom = _checker()
    lo = pd.DataFrame([b["min"].to_dict()])
    hi = pd.DataFrame([b["max"].to_dict()])
    assert (dom.design_violations(pd.concat([lo, hi], ignore_index=True)) == "").all()   # corners pass
    for c in ("floor_area", "glazing_shgc", "orientation"):
        step = 2 * b.loc[c, "tol"]
        for D in (lo.assign(**{c: b.loc[c, "min"] - step}), hi.assign(**{c: b.loc[c, "max"] + step})):
            assert dom.design_violations(D).iloc[0] == c, c
    assert dom.design_violations(lo.assign(n_floors=2.5)).iloc[0] == "n_floors"
    two = dom.design_violations(lo.assign(wwr_east=0.9, sidefin_depth=-0.1)).iloc[0]
    assert two == "wwr_east,sidefin_depth", two
    try:
        dom.require_in_domain(lo.assign(wwr_east=0.9))
    except U.DomainError:
        pass
    else:
        raise AssertionError("out-of-domain design accepted by strict mode")
    from surrogate import cv
    df, _ = cv.load_training()
    assert (dom.design_violations(df) == "").all()        # every training design is inside
    dom.require_in_domain(df)


def check_climate_distances_levels_and_guard():
    from surrogate import climate_gen as CG
    dom = _checker()
    mine, ref = dom.training_distances(), CG.climate_distances()
    assert list(mine.index) == list(ref.index)
    assert np.allclose(mine["distance_to_nearest"], ref["distance_to_nearest"], rtol=0, atol=1e-12)
    assert (mine["nearest_climate"] == ref["nearest_climate"]).all()
    assert abs(dom.d_max - 3.56) < 0.005 and mine["distance_to_nearest"].idxmax() == "phoenix", dom.d_max
    from surrogate import cv
    df, _ = cv.load_training()
    flags = dom.check(df.iloc[::50], climate=CI.add_indices(df.iloc[::50])[U.CLIMATE_COLUMNS])
    assert (flags["climate_level"] == "training").all() and flags["design_in_domain"].all()
    assert (flags["nearest_climate"] == df.iloc[::50]["climate_key"].to_numpy()).all()
    mid = (dom.climates.loc["chicago"] + dom.climates.loc["minneapolis"]) / 2   # between two neighbours
    design = df.iloc[[0]][FE.DESIGN_INPUTS]
    f = dom.check(design, climate=mid).iloc[0]
    assert f["climate_level"] == "new" and 0 < f["climate_distance"] <= dom.d_max, f
    far = mid.copy()
    far["latitude"] = 80.0                                                        # beyond the training range
    f = dom.check(design, climate=far).iloc[0]
    assert f["climate_level"] == "extrapolation" and f["climate_out_of_range"] == "latitude", f
    hot = mid.copy()
    hot["occ_cool_dh"] = dom.climates["occ_cool_dh"].max() * 1.5                  # index out of range only
    f = dom.check(design, climate=hot).iloc[0]
    assert f["climate_level"] == "extrapolation" and f["climate_out_of_range"] == "occ_cool_dh", f
    london = dom.holdout_keys[0]
    try:
        dom.check(design, climate=mid, climate_key=london)
    except CI.HoldoutClimateError:
        pass
    else:
        raise AssertionError("holdout climate accepted")
    dom.check(design, climate=mid, climate_key=london, allow_holdout=True)
    back = U.DomainChecker.from_dict(dom.to_dict())
    assert back.bounds.equals(dom.bounds) and np.allclose(back.climates, dom.climates)
    assert abs(back.d_max - dom.d_max) < 1e-12 and back.holdout_keys == dom.holdout_keys


def check_config_block_equals_code_and_selection():
    """uncertainty_3_5 appended after the frozen 3.4 file (byte-identical prefix); code reads it; the
    picks equal the official 3.4 selection (primary + peak heating) and 3.3 selection (other secondary)."""
    import hashlib
    import json
    from surrogate.config import ALL_TARGETS, CONFIG, CONFIG_PATH
    from surrogate.paths import OUT
    text = CONFIG_PATH.read_text(encoding="utf-8")
    marker = "\n# ── Added 10 Nov 2026 (Sub-task 3.5"
    assert text.count(marker) == 1
    assert hashlib.md5(text[: text.index(marker)].encode("utf-8")).hexdigest() == "006848e0fc19c23d0281b904db8fabe5"
    c = CONFIG["uncertainty_3_5"]
    assert U.ALPHA == 0.10 and U.LEVELS == (0.5, 0.8, 0.9, 0.95) and U.LOWER_BOUND == 0.0
    assert U.SCALE_PARAMS == {"n_estimators": 300, "num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 40,
                              "random_state": 42, "n_jobs": 1, "deterministic": True, "force_col_wise": True,
                              "verbose": -1}
    assert U.EPS_FRACTION == 0.01 and U.INNER_SPLITS == 5 and U.LOGO_MAX_GROUPS == 12 and U.INNER_SEED == 42
    assert U.COVERAGE_BAND == (0.85, 0.95) and U.HYBRID_MIN_WIDTH_GAIN == 0.10 and U.TRAINING_DISTANCE_TOL == 1e-6
    assert c["method"]["primary"] == "normalised" and c["domain"]["design"]["api_default"] == "strict"
    assert list(U.PICKS) == ALL_TARGETS
    s33 = json.loads((OUT / "benchmark" / "selection.json").read_text())["selection"]
    s34 = json.loads((OUT / "climate_gen" / "selection.json").read_text())
    assert s34["status"] == "final"
    for t in ALL_TARGETS:
        p = U.PICKS[t]
        ref = s34["selection"].get(t) or {**s33[t], "form": "direct"}
        assert (p["model"], p["feature_set"], p["form"]) == (ref["model"], ref["feature_set"], ref["form"]), t
        assert p["source"] == ("climate_gen_grid" if t in s34["selection"] and p["feature_set"] == "C-G+I"
                               else "benchmark_tuned" if t in s34["selection"] else "benchmark_secondary"), t


ALL_CHECKS = [check_config_block_equals_code_and_selection, check_quantile_index, check_plain_coverage_exchangeable, check_normalised_coverage_adapts,
              check_inner_splits_partition, check_nested_never_calibrates_on_scored_rows,
              check_clipping_and_gpr_mappings, check_selection_rules, check_design_bounds_and_training_rows,
              check_climate_distances_levels_and_guard]


def main():
    total = failed = 0
    print("=" * 72)
    print("SURROGATE UNCERTAINTY AND APPLICABILITY DOMAIN (3.5) - CHECKS")
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
