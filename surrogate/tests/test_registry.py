"""
test_registry.py - checks for surrogate/registry.py (Sub-task 3.8, plan claude/Phase3.8_Execution_Plan.md §3).

Key checks: config block appended after the frozen 3.7 file; a registry built into a scratch folder
loads with every SHA-256 verified and refuses a changed file; the slim heating GPR predicts exactly
as the full 3.6 file and refuses its own standard deviation; the public API gives the same values
and ranges as the tested 3.7 path (final_test.predict_variant + Conformal.interval) in all 10 training
climates and at a new site; a site built from a training EPW equals the site by key; climate levels,
readable warnings and the agreed messages; strict design refusal; input forms; chunking; optimiser
bounds; registry.json contents; no final_test / explain import and no holdout file in registry.py.

Run:  ./venv/bin/python surrogate/tests/test_registry.py   (or all suites: surrogate/run_tests.py)
Never reads the holdout. Uses the saved 3.6 / 3.7 model files (outputs/models).
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from surrogate import climate_indices as CI
from surrogate import features as FE
from surrogate import registry as R
from surrogate.config import ALL_TARGETS, CONFIG, CONFIG_PATH

_CACHE: dict = {}
MLP_TARGETS = [t for t in ALL_TARGETS if t not in ("heating_demand_kwh_m2", "eui_kwh_m2")]
EXACT_TARGETS = ["heating_demand_kwh_m2", "eui_kwh_m2"]


def _scratch() -> Path:
    if "dir" not in _CACHE:
        R.DEV_DIR.mkdir(parents=True, exist_ok=True)
        _CACHE["dir"] = Path(tempfile.mkdtemp(prefix="test_", dir=R.DEV_DIR))
    return _CACHE["dir"]


def _built() -> tuple[dict, R.Surrogate]:
    """A registry built into a scratch folder under outputs/registry_dev (git-ignored)."""
    if "sur" not in _CACHE:
        d = _scratch()
        info = R.build(dev=True, log=lambda *_: None, target_dir=d)
        _CACHE["info"], _CACHE["sur"] = info, R.Surrogate.load(d)
    return _CACHE["info"], _CACHE["sur"]


def _train():
    if "df" not in _CACHE:
        from surrogate import cv
        _CACHE["df"] = cv.load_training()[0]
    return _CACHE["df"]


def _ref():
    """The tested 3.7 path: fitted objects + interval models (imported here, never in registry.py)."""
    if "fm" not in _CACHE:
        from surrogate import explain as X
        from surrogate import final_test as FT
        _CACHE["X"], _CACHE["FT"] = X, FT
        _CACHE["fm"] = X.final_models(dev=False)
        _CACHE["iv"] = FT.load_intervals(dev=False)[0]
    return _CACHE["X"], _CACHE["FT"], _CACHE["fm"], _CACHE["iv"]


def _close(a, b, rel) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.abs(a - b) / np.maximum(np.abs(b), 1e-9)
    return float(d.max()) if d.size else 0.0


def _sample(n_per_climate: int, seed: int = 3) -> pd.DataFrame:
    df = _train()
    return df.groupby("climate_key").sample(n_per_climate, random_state=seed)[["climate_key", *FE.DESIGN_INPUTS]]


def _raw(D: pd.DataFrame, key: str, climate: dict) -> pd.DataFrame:
    raw = pd.DataFrame({"climate_key": key}, index=range(len(D)))
    for c in FE.DESIGN_INPUTS:
        raw[c] = D[c].to_numpy(float)
    for c in FE.CLIMATE_INPUTS:
        raw[c] = climate[c]
    return raw


def _assert_equal_values(got: pd.DataFrame, ref: pd.DataFrame, label: str, exact_tol: float = R.EXACT_TOL):
    for t in EXACT_TARGETS:
        e = _close(got[t], ref[t], exact_tol)
        assert e <= exact_tol, f"{label} {t}: rel. diff {e:.2e} > {exact_tol}"
    for t in MLP_TARGETS:
        e = _close(got[t], ref[t], R.MLP_TOL)
        assert e <= R.MLP_TOL, f"{label} {t}: rel. diff {e:.2e} > {R.MLP_TOL}"


# ----------------------------------------------------------------------------

def check_config_block_appended_and_read():
    text = CONFIG_PATH.read_text(encoding="utf-8")
    marker = "\n# ── Added 18 Nov 2026 (Sub-task 3.8"
    assert text.count(marker) == 1, "registry_3_8 marker missing or repeated"
    assert hashlib.md5(text[: text.index(marker)].encode("utf-8")).hexdigest() == \
        "e5487b737b2dbd60bf714325de43ac1f", "config.yaml before the 3.8 block changed"
    assert "registry_3_8" in CONFIG and text.index("\nregistry_3_8:") > text.index(marker)
    assert R.LEVELS == (0.5, 0.8, 0.9, 0.95) and R.DEFAULT_LEVEL == 0.9
    assert R.TYPE_BY_LEVEL == {"training": "standard", "new": "new_site", "extrapolation": "new_site"}
    assert R.SLIM_TOL == 1e-12 and R.EXACT_TOL == 1e-12 and R.MLP_TOL == 1e-5
    assert set(R.READABLE) == set(R.CLIMATE_COLUMNS) and len(R.READABLE) == 14
    assert CONFIG["registry_3_8"]["deployed"]["variant"] == "primary"


def check_build_load_and_hash_guard():
    info, sur = _built()
    assert info["dev"] and not info["official"]
    assert set(sur.objects) == {"gpr_heating", "xgb_eui", "mlp_cg", "mlp_physics"}
    assert set(sur.intervals) == set(ALL_TARGETS)
    # a changed file is refused
    d = _scratch()
    bad = Path(tempfile.mkdtemp(prefix="tamper_", dir=R.DEV_DIR))
    reg = json.loads((d / R.REGISTRY_FILE).read_text(encoding="utf-8"))
    reg["files"]["xgb_eui"]["sha256"] = "0" * 64
    (bad / R.REGISTRY_FILE).write_text(json.dumps(reg), encoding="utf-8")
    try:
        R.Surrogate.load(bad)
        raise AssertionError("a changed SHA-256 was not refused")
    except R.RegistryError as e:
        assert "xgb_eui" in str(e)
    finally:
        shutil.rmtree(bad, ignore_errors=True)
    try:
        R.Surrogate.load(bad)
        raise AssertionError("a missing registry.json was not refused")
    except R.RegistryError:
        pass
    # the official registry (after `--stage build` on the Mac) also loads with all hashes verified
    if (R.MODELS_DIR / R.REGISTRY_FILE).exists():
        off = R.Surrogate.load()
        assert off.registry["official"] and off.registry["files"]["gpr_heating"]["path"] == \
            f"models/{R.SLIM_FILE}"


def check_slim_gpr_identical_and_refuses_std():
    info, sur = _built()
    eq = info["slim_gpr_check"]
    assert eq["pass"] and eq["identical"] and eq["rows"] == 4000, eq
    slim = sur.objects["gpr_heating"]
    assert not hasattr(slim.base_.gp_, "L_") and type(slim.base_.gp_).__module__ == "surrogate.registry"
    assert info["files"]["gpr_heating"]["bytes"] < 1_000_000
    X, FT, fm, _ = _ref()
    raw = X.raw_frame(_sample(20)[FE.DESIGN_INPUTS], _sample(20)["climate_key"])
    M = X.matrices(raw)["C-G+I"]
    assert np.array_equal(slim.predict(M), fm.objects["gpr_heating"].predict(M))
    try:
        slim.base_.gp_.predict(M.iloc[:2, slim.base_.selected_].to_numpy(), return_std=True)
        raise AssertionError("the slim GPR returned a standard deviation")
    except RuntimeError as e:
        assert "slim" in str(e)


def check_values_equal_3_7_path_all_training_climates():
    _, sur = _built()
    X, FT, fm, _ = _ref()
    S = _sample(40)
    tab = FT.index_table_with()
    for key, g in S.groupby("climate_key"):
        D = g[FE.DESIGN_INPUTS]
        got = pd.DataFrame(sur.predict_values(D, sur.site(key)), columns=R.TARGETS)
        ref = FT.predict_variant(fm, X.raw_frame(D, key).reset_index(drop=True), tab, "primary")
        _assert_equal_values(got, ref.reset_index(drop=True), key)
        res = sur.predict(D, key)
        assert np.array_equal(res.values.to_numpy(), got.set_axis(D.index).to_numpy())


def check_intervals_equal_conformal_both_types():
    _, sur = _built()
    X, FT, fm, iv = _ref()
    S = _sample(15)
    # standard (training climates)
    for key, g in S.groupby("climate_key"):
        D = g[FE.DESIGN_INPUTS]
        Xm = FT.matrices_with(X.raw_frame(D, key).reset_index(drop=True), FT.index_table_with())["C-G+I"]
        for lev in R.LEVELS:
            res = sur.predict(D, key, level=lev)
            assert res.interval_type == "standard"
            for t in ALL_TARGETS:
                lo, hi = iv[t]["standard"].interval(Xm, res.values[t].to_numpy(float), level=lev)
                assert np.allclose(res.lower[t].to_numpy(), lo, rtol=1e-12, atol=0) and \
                    np.allclose(res.upper[t].to_numpy(), hi, rtol=1e-12, atol=0), f"{key} {t} {lev}"
    # new site: the centre of the 10 climates (none of them)
    centre = sur.climates.mean().to_dict()
    site = R.Site.from_values("centre", centre)
    D = S[FE.DESIGN_INPUTS].iloc[:80]
    res = sur.predict(D, site, level=0.9)
    assert res.climate["level"] in ("new", "extrapolation") and res.interval_type == "new_site"
    tab = FT.index_table_with({"centre": {c: centre[c] for c in CI.INDEX_COLUMNS}})
    raw = _raw(D, "centre", centre)
    ref = FT.predict_variant(fm, raw, tab, "primary")
    _assert_equal_values(res.values.reset_index(drop=True), ref, "centre")
    Xm = FT.matrices_with(raw, tab)["C-G+I"]
    for t in ALL_TARGETS:
        lo, hi = iv[t]["new_site"].interval(Xm, res.values[t].to_numpy(float), level=0.9)
        assert np.allclose(res.lower[t].to_numpy(), lo, rtol=1e-12, atol=0), t
        assert np.allclose(res.upper[t].to_numpy(), hi, rtol=1e-12, atol=0), t


def check_epw_site_equals_training_key():
    from climate.climate_features import load_climate_registry
    from surrogate.paths import EPW_DIR
    _, sur = _built()
    reg = load_climate_registry(check_files=False)
    D = _sample(3)[FE.DESIGN_INPUTS].iloc[:10]
    for key in sur.training_climates:
        s_epw = R.Site.from_epw(EPW_DIR / reg[key].epw_filename, name=key)
        s_key = sur.site(key)
        for c in R.CLIMATE_COLUMNS:
            assert abs(s_epw.climate[c] - s_key.climate[c]) <= 1e-9 * max(1.0, abs(s_key.climate[c])), \
                f"{key} {c}: EPW {s_epw.climate[c]} vs training {s_key.climate[c]}"
        chk = sur.climate_check(s_epw)
        assert chk["level"] == "training" and chk["nearest_climate"] == key, chk
        assert np.allclose(sur.predict_values(D, s_epw), sur.predict_values(D, s_key), rtol=1e-12, atol=0)


def check_climate_levels_and_readable_warnings():
    _, sur = _built()
    for key in sur.training_climates:
        assert sur.predict(_sample(1)[FE.DESIGN_INPUTS].iloc[:2], key).climate["level"] == "training"
    # out of range in latitude only -> extrapolation, readable name in the message
    base = sur.climates.loc["chicago"].to_dict()
    far = dict(base, latitude=float(sur.climates["latitude"].max()) + 5.0)
    res = sur.predict(_sample(1)[FE.DESIGN_INPUTS].iloc[:3], R.Site.from_values("far north", far))
    c = res.climate
    assert c["level"] == "extrapolation" and c["out_of_range"] == ["latitude"], c
    assert c["out_of_range_readable"] == ["latitude"] and res.interval_type == "new_site"
    assert "latitude" in res.message and sur.names[c["nearest_climate"]] in res.message
    # same levels as the 3.5 evidence: each training climate judged by a checker that has not seen it
    ev = pd.read_csv(R.OUT / "uncertainty" / "distance_table.csv").set_index("climate_key")
    for k in sur.training_climates:
        lev, _, nearest, oor = R.climate_level(sur.checker.without(k), sur.climates.loc[k].to_dict())
        assert lev == ev.at[k, "level_if_unseen"], f"{k}: {lev} vs 3.5 {ev.at[k, 'level_if_unseen']}"
    assert (ev["level_if_unseen"] == "extrapolation").sum() == 6      # 3.5: 6 of 10 (J4 evidence)
    # readable names for every out-of-range column
    assert R.readable("annual_ghi,latitude") == ["annual solar radiation", "latitude"]


def check_strict_refusal_and_flags():
    _, sur = _built()
    D = _sample(1)[FE.DESIGN_INPUTS].iloc[:4].astype(float).reset_index(drop=True)
    bad = D.copy()
    bad.loc[1, "floor_area"] = 99999.0
    try:
        sur.predict(bad, "colombo")
        raise AssertionError("an out-of-range design was not refused")
    except R.DesignError as e:
        assert str(e) == "floor_area = 99999 is outside the allowed range 1500-10000.", str(e)
    flo = D.copy()
    flo.loc[0, "n_floors"] = 2.5
    try:
        sur.predict_values(flo, "colombo")
        raise AssertionError("a non-integer n_floors was not refused")
    except R.DesignError as e:
        assert "n_floors = 2.5" in str(e) and "whole numbers only" in str(e)
    res = sur.predict(bad, "colombo", strict=False)
    assert res.flags["design_in_domain"].tolist() == [True, False, True, True]
    assert res.flags.at[1, "design_violations"] == "floor_area"
    assert np.isfinite(res.values.to_numpy()).all()
    v = sur.design_violations(bad)
    assert (v == sur.checker.design_violations(bad)).all()       # vectorised = 3.5 rule


def check_input_forms_identical():
    _, sur = _built()
    D = _sample(1)[FE.DESIGN_INPUTS].iloc[:5].reset_index(drop=True)
    ref = sur.predict_values(D, "miami")
    shuffled = D[list(reversed(FE.DESIGN_INPUTS))].assign(extra_column="ignored")
    forms = [shuffled, D.to_dict(orient="records"), [dict(r) for _, r in D.iterrows()]]
    for f in forms:
        assert np.array_equal(sur.predict_values(f, "miami"), ref)
    one = pd.DataFrame(sur.predict_values(D.iloc[0].to_dict(), "miami"), columns=R.TARGETS)
    _assert_equal_values(one, pd.DataFrame(ref[:1], columns=R.TARGETS), "one dict", exact_tol=1e-10)
    try:
        sur.predict_values(D.drop(columns=["glazing_u"]), "miami")
        raise AssertionError("a missing input was not refused")
    except KeyError as e:
        assert "glazing_u" in str(e)
    try:
        sur.site("london")
        raise AssertionError("london accepted as a training key")
    except KeyError:
        pass


def check_chunking_single_equals_batch():
    _, sur = _built()
    df = _train()
    D = df[df["climate_key"] == "denver"][FE.DESIGN_INPUTS]
    big = pd.concat([D] * 84, ignore_index=True).iloc[:25_000]          # spans 3 chunks of 10,000
    Yb = pd.DataFrame(sur.predict_values(big, "denver"), columns=R.TARGETS)
    for i in (0, 9_999, 10_000, 19_999, 24_999):
        y1 = pd.DataFrame(sur.predict_values(big.iloc[[i]], "denver"), columns=R.TARGETS)
        # one row vs a 10,000-row chunk: the GPR's matrix product runs in another BLAS block order, so
        # heating differs in the last digits (~1e-12); same-chunk comparisons above are bit-exact
        _assert_equal_values(y1, Yb.iloc[[i]].reset_index(drop=True), f"row {i}", exact_tol=1e-10)


def check_bounds_equal_design_space():
    import yaml
    from surrogate.paths import DESIGN_SPACE
    _, sur = _built()
    spec = yaml.safe_load(DESIGN_SPACE.read_text(encoding="utf-8"))["continuous_parameters"]
    b = sur.bounds()
    assert list(b.index) == FE.DESIGN_INPUTS and len(b) == 21
    for c in FE.DESIGN_INPUTS:
        assert b.at[c, "min"] == float(spec[c]["min"]) and b.at[c, "max"] == float(spec[c]["max"]), c
        assert bool(b.at[c, "integer"]) == (spec[c].get("type") == "integer"), c


def check_registry_json_complete():
    info, sur = _built()
    man = json.loads((R.MODELS_DIR / "manifest.json").read_text(encoding="utf-8"))
    need = {"created_utc", "official", "config_md5", "dataset_sha256_training", "deployed_variant", "objects",
            "target_object", "picks", "units", "files", "archive", "slim_gpr_check", "feature_sets", "intervals",
            "climates", "domain", "readable_names", "messages", "cv_reference", "runtime_modules", "environment",
            "git"}
    assert need <= set(info), sorted(need - set(info))
    assert info["dataset_sha256_training"] == CONFIG["dataset_sha256"]["training"]
    assert info["target_object"] == man["target_object"] and info["picks"] == man["picks"]
    for name in ("xgb_eui", "mlp_cg", "mlp_physics"):
        assert info["files"][name]["sha256"] == man["objects"][name]["sha256"]
    assert info["archive"]["gpr_heating_full"]["sha256"] == man["objects"]["gpr_heating"]["sha256"]
    assert len(info["feature_sets"]["C-G"]) == 63 and len(info["feature_sets"]["C-G+I"]) == 70
    assert len(info["climates"]["table"]) == 10 and "london" not in sur.climates.index
    assert info["intervals"]["max_abs_diff_vs_selection_json"] <= 1e-9
    for k in ("training", "new", "extrapolation", "heating_note", "design_refused"):
        assert info["messages"][k] == CONFIG["registry_3_8"]["messages"][k]
    assert set(info["cv_reference"]) in (set(), set(ALL_TARGETS))
    json.dumps(sur.info())                                              # JSON-ready


def check_no_final_test_import_and_no_holdout():
    src = Path(R.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add(node.module or "")
            mods |= {f"{node.module}.{a.name}" for a in node.names}
    banned = [m for m in mods if m.endswith(("final_test", "explain", "timing")) or "eda_lib" in m]
    assert not banned, f"registry.py imports {banned}"
    assert not re.search(r"holdout\.csv|dataset_holdout|load_split|load_holdout", src)
    code = ("import sys; from surrogate import registry; "
            "print(any(m in sys.modules for m in ('surrogate.final_test','surrogate.explain','surrogate.timing')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.stdout.strip() == "False", out.stdout + out.stderr


def check_domain_d_max_kept():
    from surrogate import uncertainty as U
    info, sur = _built()
    assert abs(sur.checker.d_max - 3.559) <= float(CONFIG["registry_3_8"]["domain"]["d_max_tol"])
    assert sur.checker.d_max == info["domain"]["d_max"] == U.DomainChecker.from_training().d_max
    nine = sur.checker.without("phoenix")
    assert nine.d_max == sur.checker.d_max                              # LOCO-style checker keeps D_max


def check_messages_render_without_internal_codes():
    _, sur = _built()
    pat = re.compile(r"\b(S\d|J\d|G\d|H\d|I\d|M\d{1,2}|C-G\+?I?|LOCO|3\.\d|2\.\d)\b")
    D = _sample(1)[FE.DESIGN_INPUTS].iloc[:1]
    sites = [sur.site("colombo"), R.Site.from_values("centre", sur.climates.mean().to_dict()),
             R.Site.from_values("far", dict(sur.climates.loc["chicago"].to_dict(), latitude=80.0))]
    seen = set()
    for s in sites:
        res = sur.predict(D, s)
        seen.add(res.climate["level"])
        for txt in (res.message, res.heating_note):
            assert "{" not in txt and "}" not in txt, txt
            assert not pat.search(txt), f"internal code in message: {pat.search(txt).group(0)}"
    assert {"training", "extrapolation"} <= seen


def check_prediction_object():
    _, sur = _built()
    D = _sample(1)[FE.DESIGN_INPUTS].iloc[:6]
    res = sur.predict(D, "atlanta", level=0.8)
    f = res.to_frame()
    assert list(f.index) == list(D.index) and "heating_demand_kwh_m2_lo_80" in f.columns
    assert ((res.lower <= res.values + 1e-12) & (res.values <= res.upper + 1e-12)).all().all()
    assert (res.lower >= 0).all().all()
    w80 = (res.upper - res.lower).to_numpy()
    r95 = sur.predict(D, "atlanta", level=0.95)
    assert (r95.upper - r95.lower).to_numpy().sum() > w80.sum()
    json.dumps(res.to_dict())
    try:
        sur.predict(D, "atlanta", level=0.85)
        raise AssertionError("an unsupported level was accepted")
    except ValueError:
        pass
    e = sur.predict(D.iloc[:0], "atlanta")
    assert len(e.values) == 0 and sur.predict_values(D.iloc[:0], "atlanta").shape == (0, 10)


ALL_CHECKS = [check_config_block_appended_and_read, check_build_load_and_hash_guard,
              check_slim_gpr_identical_and_refuses_std, check_values_equal_3_7_path_all_training_climates,
              check_intervals_equal_conformal_both_types, check_epw_site_equals_training_key,
              check_climate_levels_and_readable_warnings, check_strict_refusal_and_flags,
              check_input_forms_identical, check_chunking_single_equals_batch, check_bounds_equal_design_space,
              check_registry_json_complete, check_no_final_test_import_and_no_holdout, check_domain_d_max_kept,
              check_messages_render_without_internal_codes, check_prediction_object]


def main():
    total = failed = 0
    print("=" * 72)
    print("SURROGATE REGISTRY AND API (3.8) - CHECKS")
    print("=" * 72)
    try:
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
    finally:
        if "dir" in _CACHE:
            shutil.rmtree(_CACHE["dir"], ignore_errors=True)
    print("=" * 72)
    print(f"RESULTS: {total - failed}/{total} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
