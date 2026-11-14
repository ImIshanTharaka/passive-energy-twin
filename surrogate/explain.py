"""
explain.py - Sub-task 3.6: final models, physics consistency and SHAP explanations.

Plan: claude/Phase3.6_Execution_Plan.md (decisions H1-H8, agreed 14 Nov 2026). Every setting is read
from config.yaml block `explain_3_6` (appended 14 Nov 2026, before any 3.6 run).

In plain words
    1. fit      The final model of each target (3.3 / 3.4 selection, `uncertainty_3_5.picks`) is
                trained ONCE on all 3,000 training rows and saved with SHA-256 hashes (H1). 3.7 tests
                these files; 3.8 packages them. Four fitted objects serve the 10 targets:
                    gpr_heating   heating             GPR / C-G+I / log1p (top-15 features)
                    xgb_eui       EUI                 XGBoost / C-G+I / decomposed
                    mlp_cg        cooling, peak cooling, 5 zones   one MLP / C-G / direct
                    mlp_physics   peak heating        14-output MLP / C-G+I / direct
    2. oat      One-at-a-time test (gate S5, H5): the 86 EnergyPlus runs that move one parameter to
                its minimum or maximum (Colombo, Minneapolis). Surrogate change vs EnergyPlus change
                from the base design: same sign?
    3. sweeps   Monotonic sweeps (gate S6, H6): slide SHGC, glazing U or one facade's WWR over 21
                values on 200 training designs per climate; any move against physics larger than
                the model's CV RMSE counts as a reversal.
    4. zones    Zone-sum consistency (H8, report): area-weighted zone cooling vs building cooling.
    5. shap     Permutation SHAP in the 21 design inputs + the climate as ONE block (H2, H3), in
                original units; per-climate view; comparison with the 2.8 RF SHAP.

    ./venv/bin/python surrogate/explain.py --stage fit [--dev]
    ./venv/bin/python surrogate/explain.py --stage oat | sweeps | zones [--dev]
    ./venv/bin/python surrogate/explain.py --stage shap --probe [--dev]   # timing guard first (H2)
    ./venv/bin/python surrogate/explain.py --stage shap | figures | report [--dev]

Guards: only training rows are fitted or explained (eda_lib.HoldoutLeakError); only the 10 training
climates can be predicted here (London's indices are not in the table: KeyError); dev runs write to
outputs/explain_dev/ and outputs/models_dev/ and are never results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from surrogate import climate_indices as CI
from surrogate import features as FE
from surrogate import models
from surrogate.config import ALL_TARGETS, CONFIG, PRIMARY_TARGETS
from surrogate.paths import OAT_DIR, OUT, SIM_ROOT

# ── Settings: config.yaml block explain_3_6 (frozen 14 Nov 2026, before any 3.6 run) ────────────

CFG = CONFIG["explain_3_6"]
PICKS: dict = CONFIG["uncertainty_3_5"]["picks"]          # H1: same final model per target
HEAT, COOL, EUI = models.HEAT, models.COOL, models.EUI
PEAK_HEAT, PEAK_COOL = "peak_heating_w_m2", "peak_cooling_w_m2"
HEATING_TARGETS = tuple(CONFIG["targets"]["heating"])     # scored on heating climates only (handoff §4)
ZONES = ("core", "south", "east", "north", "west")
ZONE_T = {z: f"zone_cooling_{z}_kwh_m2" for z in ZONES}
CHUNK = int(CFG["final_models"]["predict_chunk_rows"])
OBJECTS: dict = CFG["final_models"]["objects"]

_O = CFG["oat"]
OAT_CLIMATES = tuple(_O["climates"])
OAT_REL = float(_O["counted_if_abs_delta_ep_gt_share_of_base"])     # 2 % (S5 as frozen)
STRONG_SET: dict = {k: list(v) for k, v in _O["strong_set"].items()}
MIN_SIGN_SHARE = float(_O["min_share_sign_correct"])                 # 0.90
ZERO_REL = float(_O["zero_response_rel"])

_S = CFG["sweeps"]
SWEEP_N, SWEEP_SEED, SWEEP_POINTS = int(_S["designs_per_climate"]), int(_S["seed"]), int(_S["n_points"])
EXPECTATIONS = [tuple(e) for e in _S["expectations"]]                # (input, target, up|down, all|heating)
MAX_REVERSING = float(_S["max_share_reversing"])                     # 1 %

_Z = CFG["zone_sum"]
PERIM = float(_Z["perimeter_depth_m"])
MIN_COOL = float(_Z["min_cooling_kwh_m2"])

_H = CFG["shap"]
MAX_EVALS, SHAP_SEED = int(_H["max_evals"]), int(_H["seed"])
ADD_TOL = float(_H["additivity_rel_tol"])
GUARD = _H["runtime_guard"]

DEV_SHAP = {"rows_per_climate": 2, "background_overall": 8, "background_per_climate": 6, "max_evals": 60}
DEV_SWEEP_N = 10
CLIMATE_PLAYER = "climate"
DESIGN_GROUPS = None      # filled lazily from eda_lib.FEATURE_GROUPS (needs simulation on the path)


def object_for(target: str) -> str:
    """Name of the fitted object that predicts `target` (from the picks)."""
    p = PICKS[target]
    if p["model"] == "gpr":
        return "gpr_heating"
    if p["model"] == "xgb" and p["form"] == "decomposed":
        return "xgb_eui"
    if p["model"] == "mlp" and p["feature_set"] == "C-G":
        return "mlp_cg"
    if p["model"] == "mlp" and p["feature_set"] == "C-G+I":
        return "mlp_physics"
    raise ValueError(f"no fitted object for pick {p} of {target}")


TARGET_OBJECT = {t: object_for(t) for t in ALL_TARGETS}


def object_targets(name: str) -> list[str]:
    return [t for t in ALL_TARGETS if TARGET_OBJECT[t] == name]


# ── paths, bookkeeping ──────────────────────────────────────────────────────────────────────────

def out_dir(dev: bool) -> Path:
    d = OUT / ("explain_dev" if dev else "explain")
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir(dev: bool) -> Path:
    d = OUT / ("models_dev" if dev else "models")
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def record_stage(d: Path, stage: str, info: dict, dev: bool):
    from surrogate import benchmark as B
    path = d / "explain_result.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"stages": {}}
    data["stages"][stage] = {"official": not dev, "dev": dev, "created_utc": datetime.now(timezone.utc).isoformat(),
                             "dataset_sha256_training": CONFIG["dataset_sha256"]["training"],
                             "environment": B.environment(), **info}
    path.write_text(json.dumps(data, indent=2, default=float), encoding="utf-8")


def read_result(dev: bool) -> dict:
    path = out_dir(dev) / "explain_result.json"
    return json.loads(path.read_text(encoding="utf-8"))["stages"] if path.exists() else {}


# ── data ────────────────────────────────────────────────────────────────────────────────────────

def load(dev: bool):
    """(df, heat): training rows (dev: benchmark.load's fixed 25 % design sample) and heating climates."""
    from surrogate import benchmark as B
    df, _, heat = B.load(dev)
    return df, list(heat)


_TABLES: dict = {}


def climate_table() -> pd.DataFrame:
    """The 10 training climates: 7 Phase 2 climate inputs (from the training rows) + 7 EPW indices."""
    if "climate" not in _TABLES:
        from surrogate import cv
        df, _ = cv.load_training()
        g = df.groupby("climate_key")[FE.CLIMATE_INPUTS]
        if (g.nunique() > 1).any().any():
            raise AssertionError("climate inputs vary within a climate")
        _TABLES["climate"] = g.first().join(index_table())
    return _TABLES["climate"]


def index_table() -> pd.DataFrame:
    if "index" not in _TABLES:
        _TABLES["index"] = CI.training_indices()
    return _TABLES["index"]


def raw_frame(designs: pd.DataFrame, climate_keys) -> pd.DataFrame:
    """21 design inputs + a training climate per row -> climate_key + the 28 raw inputs.

    Refuses a climate that is not one of the 10 training climates (London: 3.7 only)."""
    tab = climate_table()
    keys = np.asarray(climate_keys).astype(str)
    if keys.shape == ():
        keys = np.full(len(designs), str(keys))
    bad = sorted(set(keys) - set(tab.index))
    if bad:
        raise KeyError(f"not a training climate: {bad} (London is scored only in final_test.py, 3.7)")
    out = pd.DataFrame({"climate_key": keys}, index=designs.index)
    for c in FE.DESIGN_INPUTS:
        out[c] = designs[c].to_numpy(dtype=float)
    vals = tab.loc[keys, FE.CLIMATE_INPUTS].to_numpy(dtype=float)
    for j, c in enumerate(FE.CLIMATE_INPUTS):
        out[c] = vals[:, j]
    return out


def matrices(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Model inputs: {"C-G": 63 columns, "C-G+I": 70 columns}, same row order, reset index."""
    raw = raw.reset_index(drop=True)
    F = FE.engineer(raw)
    cg = F[FE.feature_set("C-G")]
    idx = index_table().loc[raw["climate_key"].astype(str).to_numpy(), CI.INDEX_COLUMNS].reset_index(drop=True)
    return {"C-G": cg, "C-G+I": pd.concat([cg, idx], axis=1)}


# ── 1. final models (H1) ────────────────────────────────────────────────────────────────────────

def object_params(name: str, dev: bool) -> dict:
    """Frozen settings of one object, exactly as in the CV runs (no re-tuning)."""
    from surrogate import benchmark as B
    from surrogate import climate_gen as CG
    spec = OBJECTS[name]
    if name == "gpr_heating":
        k = CG.resolve_gpr_top_k(dev)
        if not dev and k != int(spec["top_k"]):
            raise SystemExit(f"GPR top_k {k} differs from explain_3_6 ({spec['top_k']})")
        return CG.params_for("gpr", [HEAT], dev, k)[HEAT]
    if name == "xgb_eui":
        return CG.params_for("xgb", [HEAT, COOL, EUI], dev, None)
    if name == "mlp_cg":
        return B.tuned_params_for("mlp", ALL_TARGETS, dev, allow_missing=dev, feature_set=None if dev else "C-G")[None]
    if name == "mlp_physics":
        p = CG.params_for("mlp", PRIMARY_TARGETS, dev, None)[None]
        return {**p, "max_epochs": 15, "patience": 5} if dev else p
    raise KeyError(name)


def fit_object(name: str, df: pd.DataFrame, X: dict[str, pd.DataFrame], dev: bool):
    """Fit one object on the given training rows (same calls as the 3.3 / 3.4 CV code)."""
    from eda import eda_lib as L
    L.require_training(df)
    p = object_params(name, dev)
    ctx = df.reset_index(drop=True)
    groups = df["design_id"].astype(str).to_numpy()
    Y = df[ALL_TARGETS].to_numpy(dtype=float, copy=True)
    if name == "gpr_heating":
        est = models.make_form("gpr", p, "log1p")(HEAT)
        return est.fit(X["C-G+I"], df[HEAT].to_numpy(dtype=float, copy=True)), p
    if name == "xgb_eui":
        est = models.make_decomposed("xgb", p)(EUI)
        return est.fit(X["C-G+I"], df[EUI].to_numpy(dtype=float, copy=True), context=ctx), p
    if name == "mlp_cg":
        est = models.make_multi("mlp", p)()
        return est.fit(X["C-G"], Y, groups=groups), p
    if name == "mlp_physics":
        est = models.MLPPhysics(p)
        return est.fit(X["C-G+I"], Y, groups=groups, context=ctx), p
    raise KeyError(name)


class FinalModels:
    """The four fitted objects; predict() maps raw inputs (+ training climate) to all 10 targets."""

    def __init__(self, objects: dict, manifest: dict | None = None):
        missing = sorted(set(TARGET_OBJECT.values()) - set(objects))
        if missing:
            raise ValueError(f"missing fitted objects {missing}")
        self.objects = objects
        self.manifest = manifest or {}

    # prediction -------------------------------------------------------------------------------
    def _predict_chunk(self, name: str, X: dict[str, pd.DataFrame]) -> pd.DataFrame:
        est = self.objects[name]
        if name == "gpr_heating":
            return pd.DataFrame({HEAT: est.predict(X["C-G+I"])})
        if name == "xgb_eui":
            return pd.DataFrame({EUI: est.predict(X["C-G+I"])})
        if name == "mlp_cg":
            out = pd.DataFrame(np.asarray(est.predict(X["C-G"]), dtype=float).reshape(len(X["C-G"]), -1),
                               columns=ALL_TARGETS)
            return out[object_targets(name)]
        if name == "mlp_physics":
            forms = est.predict_forms(X["C-G+I"])
            return pd.DataFrame({t: forms[(t, "direct")].to_numpy() for t in object_targets(name)})
        raise KeyError(name)

    def predict_object(self, name: str, raw: pd.DataFrame) -> pd.DataFrame:
        """Targets of one object for raw rows (climate_key + 28 inputs), original units, chunked."""
        parts = []
        for i in range(0, len(raw), CHUNK):
            parts.append(self._predict_chunk(name, matrices(raw.iloc[i:i + CHUNK])))
        out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=object_targets(name))
        out.index = raw.index
        return out

    def predict(self, raw: pd.DataFrame, targets: list[str] | None = None) -> pd.DataFrame:
        targets = list(targets or ALL_TARGETS)
        names = list(dict.fromkeys(TARGET_OBJECT[t] for t in targets))
        parts = [self.predict_object(n, raw) for n in names]
        return pd.concat(parts, axis=1)[targets]

    def predict_designs(self, designs: pd.DataFrame, climate_keys, targets=None) -> pd.DataFrame:
        return self.predict(raw_frame(designs, climate_keys), targets)

    # persistence ------------------------------------------------------------------------------
    def save(self, d: Path, info: dict) -> dict:
        files = {}
        for name, est in self.objects.items():
            path = d / f"{name}.pkl"
            with open(path, "wb") as fh:
                pickle.dump(est, fh, protocol=pickle.HIGHEST_PROTOCOL)
            files[name] = {"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size,
                           "targets": object_targets(name), **info.get(name, {})}
        from surrogate import benchmark as B
        manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                    "dataset_sha256_training": CONFIG["dataset_sha256"]["training"],
                    "config_block": "explain_3_6.final_models", "target_object": TARGET_OBJECT,
                    "picks": PICKS, "environment": B.environment(), "objects": files}
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
        self.manifest = manifest
        return manifest

    @classmethod
    def load(cls, d: Path) -> "FinalModels":
        mpath = d / "manifest.json"
        if not mpath.exists():
            raise SystemExit(f"no fitted models in {d}: run `explain.py --stage fit` first")
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        if manifest["dataset_sha256_training"] != CONFIG["dataset_sha256"]["training"]:
            raise RuntimeError("saved models were fitted on another dataset")
        objects = {}
        for name, rec in manifest["objects"].items():
            path = d / rec["file"]
            if sha256(path) != rec["sha256"]:
                raise RuntimeError(f"{path.name}: SHA-256 differs from manifest.json (file changed?)")
            if name.startswith("mlp"):
                from surrogate.mlp import _torch
                _torch()                                     # safe OpenMP import order before unpickling
            with open(path, "rb") as fh:
                objects[name] = pickle.load(fh)
        return cls(objects, manifest)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def object_info(name: str, est, params: dict, fit_s: float) -> dict:
    spec = OBJECTS[name]
    info = {"model": spec["model"], "feature_set": spec["feature_set"], "form": spec["form"],
            "params": params, "fit_s": round(fit_s, 2)}
    if name == "gpr_heating":
        base = est.base_
        info.update(gpr_selected_features=base.selected_features_,
                    gpr_length_scales=[float(v) for v in base.length_scales_],
                    gpr_noise_level=float(base.noise_level_), gpr_bound_warnings=base.bound_warnings_)
    if name.startswith("mlp"):
        net = est.net_ if name == "mlp_physics" else est
        info.update(best_epoch=int(net.best_epoch_), n_epochs=int(net.n_epochs_))
    if name == "xgb_eui":
        info.update(oa_level_coef=[float(v) for v in est.level_.coef_],
                    oa_climate_mean={k: float(v) for k, v in est.oa_climate_mean_.items()})
    return info


def fit_final(df: pd.DataFrame, dev: bool, log=print) -> tuple[FinalModels, dict]:
    """H1: fit every object once on the given training rows. Returns (models, per-object info)."""
    from eda import eda_lib as L
    L.require_training(df)
    X = matrices(df[["climate_key", *FE.RAW_INPUTS]])
    ref = CI.feature_matrix(df).reset_index(drop=True)
    if not np.allclose(X["C-G+I"].to_numpy(), ref[X["C-G+I"].columns].to_numpy(), rtol=0, atol=1e-12):
        raise AssertionError("explain.matrices differs from climate_indices.feature_matrix")
    objects, info = {}, {}
    for name in OBJECTS:
        t0 = time.perf_counter()
        est, p = fit_object(name, df, X, dev)
        objects[name] = est
        info[name] = object_info(name, est, p, time.perf_counter() - t0)
        log(f"  fitted {name:12s} ({', '.join(object_targets(name))}) in {info[name]['fit_s']:.1f} s")
    return FinalModels(objects), info


def stage_fit(dev: bool, log=print):
    df, heat = load(dev)
    log(f"fit: {len(df)} training rows, {df['climate_key'].nunique()} climates")
    fm, info = fit_final(df, dev, log)
    d = models_dir(dev)
    manifest = fm.save(d, info)
    back = FinalModels.load(d)                             # round trip: identical predictions
    sample = df.sample(min(200, len(df)), random_state=0)
    raw = sample[["climate_key", *FE.RAW_INPUTS]]
    a, b = fm.predict(raw), back.predict(raw)
    if not np.array_equal(a.to_numpy(), b.to_numpy()):
        raise AssertionError("saved -> loaded models predict differently")
    insample = fm.predict(df[["climate_key", *FE.RAW_INPUTS]])
    fit_summary = {}
    for t in ALL_TARGETS:
        rows = df["climate_key"].isin(heat).to_numpy() if t in HEATING_TARGETS else np.ones(len(df), bool)
        e = insample[t].to_numpy()[rows] - df[t].to_numpy()[rows]
        fit_summary[t] = {"object": TARGET_OBJECT[t], "in_sample_rmse": float(np.sqrt(np.mean(e ** 2))),
                          "in_sample_mae": float(np.mean(np.abs(e)))}
    record_stage(out_dir(dev), "fit", {"models_dir": str(d.relative_to(OUT.parent)), "n_rows": len(df),
                                       "objects": {k: {kk: v[kk] for kk in ("file", "sha256", "fit_s")}
                                                   for k, v in manifest["objects"].items()},
                                       "in_sample": fit_summary,
                                       "gpr_selected_features": info["gpr_heating"]["gpr_selected_features"]}, dev)
    log(f"  saved to {d} (manifest.json; reload predicts identically)")
    log("  GPR heating top features: " + ", ".join(info["gpr_heating"]["gpr_selected_features"] or []))
    return fm


def final_models(dev: bool) -> FinalModels:
    return FinalModels.load(models_dir(dev))


# ── 2. OAT test (H5, gate S5) ───────────────────────────────────────────────────────────────────

def load_oat() -> pd.DataFrame:
    """86 OAT runs: run_id, design_id, climate_key, parameter, bound, 21 design inputs, 10 targets (EnergyPlus)."""
    from runner.output_parser import TARGETS as SOURCES
    samples = pd.read_csv(SIM_ROOT / _O["samples"].split("simulation/", 1)[1])
    results = pd.read_csv(SIM_ROOT / _O["results"].split("simulation/", 1)[1])
    r = results[["run_id", *SOURCES.values()]].rename(columns={v: k for k, v in SOURCES.items()})
    oat = samples.merge(r, on="run_id", how="inner", validate="one_to_one")
    if len(oat) != len(samples) or oat[ALL_TARGETS].isna().any().any():
        raise AssertionError(f"OAT: {len(oat)} of {len(samples)} runs have results")
    if set(oat["climate_key"]) != set(OAT_CLIMATES):
        raise AssertionError(f"OAT climates {sorted(set(oat['climate_key']))} != {OAT_CLIMATES}")
    split = oat["design_id"].astype(str).str.split("__", n=1, expand=True)
    oat["parameter"] = split[0]
    oat["bound"] = split[1] if split.shape[1] > 1 else None
    oat.loc[oat["design_id"] == "BASE", ["parameter", "bound"]] = ["BASE", "base"]
    return oat


def oat_table(fm: FinalModels, oat: pd.DataFrame, heat: list[str]) -> pd.DataFrame:
    """One row per (parameter, bound, climate, target): EnergyPlus and surrogate change from the base."""
    pred = fm.predict(raw_frame(oat, oat["climate_key"]))
    rows = []
    for c in OAT_CLIMATES:
        g = oat[oat["climate_key"] == c]
        base = g[g["design_id"] == "BASE"]
        if len(base) != 1:
            raise AssertionError(f"OAT {c}: {len(base)} base runs")
        b = base.index[0]
        base_x = oat.loc[b, FE.DESIGN_INPUTS].to_numpy(dtype=float)
        for i in g.index[g["design_id"] != "BASE"]:
            same = bool(np.allclose(oat.loc[i, FE.DESIGN_INPUTS].to_numpy(dtype=float), base_x, rtol=0, atol=1e-9))
            for t in ALL_TARGETS:
                ep_b, ep_r = float(oat.loc[b, t]), float(oat.loc[i, t])
                s_b, s_r = float(pred.loc[b, t]), float(pred.loc[i, t])
                rows.append({"parameter": oat.loc[i, "parameter"], "bound": oat.loc[i, "bound"], "climate_key": c,
                             "target": t, "duplicate_of_base": same, "ep_base": ep_b, "ep_run": ep_r,
                             "d_ep": ep_r - ep_b, "sur_base": s_b, "sur_run": s_r, "d_sur": s_r - s_b})
    tab = pd.DataFrame(rows)
    return classify_oat(tab, heat)


def classify_oat(tab: pd.DataFrame, heat: list[str]) -> pd.DataFrame:
    """H5 counting rules (pure function of the deltas; tested on synthetic tables)."""
    tab = tab.copy()
    tab["informative"] = ~tab["duplicate_of_base"].astype(bool)
    is_heat_t = tab["target"].isin(HEATING_TARGETS)
    tab["scored"] = tab["informative"] & (~is_heat_t | tab["climate_key"].isin(heat))
    tab["rel_ep"] = tab["d_ep"] / tab["ep_base"].abs().where(tab["ep_base"].abs() > 0)
    tab["counted"] = tab["scored"] & (tab["d_ep"].abs() > OAT_REL * tab["ep_base"].abs())
    tab["sign_ok"] = np.sign(tab["d_sur"]) == np.sign(tab["d_ep"])
    tab["strong"] = [t in STRONG_SET.get(p, []) for p, t in zip(tab["parameter"], tab["target"])]
    tab["zero_response"] = tab["d_sur"].abs() <= ZERO_REL * tab["sur_base"].abs().clip(lower=1e-12)
    tab["abs_err"] = (tab["d_sur"] - tab["d_ep"]).abs()
    tab["ratio"] = tab["d_sur"] / tab["d_ep"].where(tab["d_ep"] != 0)
    return tab


def s5(tab: pd.DataFrame) -> dict:
    """Gate S5: (a) every counted strong delta sign-correct; (b) >= 90 % of all counted deltas."""
    c = tab[tab["counted"]]
    strong = c[c["strong"]]
    share = float(c["sign_ok"].mean()) if len(c) else float("nan")
    ok_a = bool(strong["sign_ok"].all()) and len(strong) > 0
    ok_b = bool(share >= MIN_SIGN_SHARE)
    ss_res = float(((c["d_sur"] - c["d_ep"]) ** 2).sum())
    ss_tot = float(((c["d_ep"] - c["d_ep"].mean()) ** 2).sum())
    cols = ["parameter", "bound", "climate_key", "target", "d_ep", "d_sur", "rel_ep"]
    return {"status": "PASS" if ok_a and ok_b else "FAIL", "strong_all_sign_correct": ok_a, "share_ok": ok_b,
            "n_pairs_informative": int(tab.loc[tab["informative"], ["parameter", "bound"]].drop_duplicates().shape[0]),
            "n_dropped_duplicate_pairs": int(tab.loc[~tab["informative"], ["parameter", "bound"]]
                                             .drop_duplicates().shape[0]),
            "n_deltas_scored": int(tab["scored"].sum()), "n_counted": int(len(c)),
            "n_counted_strong": int(len(strong)), "n_strong_sign_correct": int(strong["sign_ok"].sum()),
            "share_sign_correct": share, "threshold_share": MIN_SIGN_SHARE,
            "r2_delta": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "median_abs_err_rel_base": float((c["abs_err"] / c["ep_base"].abs()).median()) if len(c) else float("nan"),
            "median_ratio": float(c["ratio"].median()) if len(c) else float("nan"),
            "strong_misses": strong.loc[~strong["sign_ok"], cols].to_dict("records"),
            "other_misses": c.loc[~c["sign_ok"] & ~c["strong"], cols].to_dict("records"),
            "zero_responses": tab.loc[tab["scored"] & tab["zero_response"] & (tab["d_ep"] != 0),
                                      ["parameter", "bound", "climate_key", "target", "d_ep"]].to_dict("records"),
            "by_target": c.groupby("target")["sign_ok"].agg(["sum", "count"]).astype(int)
                          .rename(columns={"sum": "correct", "count": "counted"}).to_dict("index")}


def stage_oat(dev: bool, log=print):
    fm = final_models(dev)
    _, heat = load(dev)
    oat = load_oat()
    tab = oat_table(fm, oat, heat)
    d = out_dir(dev)
    tab.to_csv(d / "oat_table.csv", index=False)
    res = s5(tab)
    (d / "oat_summary.json").write_text(json.dumps(res, indent=2, default=_json_default), encoding="utf-8")
    record_stage(d, "oat", {"S5": res["status"], "share_sign_correct": res["share_sign_correct"],
                            "n_counted": res["n_counted"], "n_counted_strong": res["n_counted_strong"],
                            "n_strong_misses": len(res["strong_misses"]), "r2_delta": res["r2_delta"]}, dev)
    log(f"oat: {res['n_pairs_informative']} informative pairs ({res['n_dropped_duplicate_pairs']} duplicate of base "
        f"dropped); {res['n_counted']} counted deltas, {res['share_sign_correct']:.1%} sign-correct; strong set "
        f"{res['n_strong_sign_correct']}/{res['n_counted_strong']}; R2(delta) {res['r2_delta']:.3f} -> S5 {res['status']}")
    return res


# ── 3. monotonic sweeps (H6, gate S6) ───────────────────────────────────────────────────────────

def sweep_starts(df: pd.DataFrame, n_per_climate: int, seed: int = SWEEP_SEED) -> pd.DataFrame:
    """n random training designs per climate (one RNG, climates in sorted order)."""
    rng = np.random.default_rng(seed)
    parts = []
    for c in sorted(df["climate_key"].astype(str).unique()):
        g = df[df["climate_key"].astype(str) == c]
        idx = rng.choice(len(g), size=min(n_per_climate, len(g)), replace=False)
        parts.append(g.iloc[np.sort(idx)])
    return pd.concat(parts).reset_index(drop=True)


def reversal_size(Y: np.ndarray, direction: str) -> np.ndarray:
    """Largest move against the expected direction along each row of Y (n_designs, n_points).

    up:   max over i < j of (y_i - y_j)  - the deepest dip below an earlier value (0 if monotone)
    down: max over i < j of (y_j - y_i)."""
    Y = np.asarray(Y, dtype=float)
    if direction == "down":
        Y = -Y
    elif direction != "up":
        raise ValueError(direction)
    return np.max(np.maximum.accumulate(Y, axis=1) - Y, axis=1)


def pick_rmse(heat: list[str]) -> dict[str, float]:
    """Pooled grouped-CV RMSE of each pick (saved official OOF, scored rows, original units)."""
    from surrogate import uncertainty as U
    out = {}
    for t in ALL_TARGETS:
        f = U.pick_oof(t, "grouped", dev=False)
        if t in HEATING_TARGETS:
            f = f[f["climate_key"].isin(heat)]
        out[t] = float(np.sqrt(np.mean((f["y"] - f["yhat"]) ** 2)))
    return out


def sweep_grid(inp: str) -> np.ndarray:
    from surrogate.uncertainty import design_bounds
    b = design_bounds().loc[inp]
    return np.linspace(float(b["min"]), float(b["max"]), SWEEP_POINTS)


def sweep_predictions(fm: FinalModels, starts: pd.DataFrame, inp: str, targets: list[str]) -> dict[str, np.ndarray]:
    """{target: (n_starts, n_points)} predictions with `inp` moved over its grid, all else fixed."""
    grid = sweep_grid(inp)
    n, k = len(starts), len(grid)
    rep = starts.loc[starts.index.repeat(k), ["climate_key", *FE.DESIGN_INPUTS]].reset_index(drop=True)
    rep[inp] = np.tile(grid, n)
    pred = fm.predict(raw_frame(rep, rep["climate_key"]), targets)
    return {t: pred[t].to_numpy().reshape(n, k) for t in targets}


def run_sweeps(fm: FinalModels, starts: pd.DataFrame, heat: list[str], rmse: dict) -> pd.DataFrame:
    rows = []
    for inp in dict.fromkeys(e[0] for e in EXPECTATIONS):
        exps = [e for e in EXPECTATIONS if e[0] == inp]
        P = sweep_predictions(fm, starts, inp, [e[1] for e in exps])
        for _, t, direction, where in exps:
            keep = starts["climate_key"].isin(heat).to_numpy() if where == "heating" else np.ones(len(starts), bool)
            Y = P[t][keep]
            rev = reversal_size(Y, direction)
            s = starts[keep]
            rows.append(pd.DataFrame({"input": inp, "target": t, "direction": direction,
                                      "run_id": s["run_id"].to_numpy(), "climate_key": s["climate_key"].to_numpy(),
                                      "y_first": Y[:, 0], "y_last": Y[:, -1],
                                      "net_change": (Y[:, -1] - Y[:, 0]) * (1 if direction == "up" else -1),
                                      "reversal": rev, "rmse": rmse[t], "reverses": rev > rmse[t]}))
    return pd.concat(rows, ignore_index=True)


def s6(sw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pooled = (sw.groupby(["input", "target", "direction"], sort=False)
              .agg(n=("reverses", "size"), n_reversing=("reverses", "sum"), share_reversing=("reverses", "mean"),
                   max_reversal=("reversal", "max"), rmse=("rmse", "first"),
                   share_net_wrong=("net_change", lambda v: float((v < 0).mean())))
              .reset_index())
    pooled["warn"] = pooled["share_reversing"] > MAX_REVERSING
    per_c = (sw.groupby(["input", "target", "climate_key"], sort=False)
             .agg(n=("reverses", "size"), share_reversing=("reverses", "mean"), max_reversal=("reversal", "max"),
                  share_net_wrong=("net_change", lambda v: float((v < 0).mean())))
             .reset_index())
    flagged = pooled.loc[pooled["warn"], ["input", "target", "share_reversing"]].to_dict("records")
    return pooled, per_c, {"status": "WARN" if flagged else "OK", "flagged": flagged, "threshold": MAX_REVERSING}


def stage_sweeps(dev: bool, log=print):
    fm = final_models(dev)
    df, heat = load(dev)
    starts = sweep_starts(df, DEV_SWEEP_N if dev else SWEEP_N)
    rmse = pick_rmse(heat)
    t0 = time.time()
    sw = run_sweeps(fm, starts, heat, rmse)
    pooled, per_c, res = s6(sw)
    d = out_dir(dev)
    sw.to_csv(d / "sweeps.csv.gz", index=False)
    pooled.to_csv(d / "sweep_summary.csv", index=False)
    per_c.to_csv(d / "sweep_per_climate.csv", index=False)
    record_stage(d, "sweeps", {"S6": res["status"], "flagged": res["flagged"], "n_starts": len(starts),
                               "rmse_threshold": rmse, "seconds": round(time.time() - t0, 1),
                               "summary": pooled.to_dict("records")}, dev)
    log(f"sweeps: {len(starts)} start designs x {SWEEP_POINTS} points, {time.time() - t0:.0f} s")
    for _, r in pooled.iterrows():
        log(f"  {r['input']:13s} -> {r['target']:28s} {r['direction']:4s} reversing {r['share_reversing']:6.2%} "
            f"(max {r['max_reversal']:.3f} vs RMSE {r['rmse']:.3f})")
    log(f"  -> S6 {res['status']}" + (f" (adjudicate: {res['flagged']})" if res["flagged"] else ""))
    return res


# ── 4. zone-sum consistency (H8, report) ────────────────────────────────────────────────────────

def zone_areas(raw: pd.DataFrame) -> pd.DataFrame:
    """Zone floor areas per storey (m2): perimeter trapezoids of depth 4.57 m + core."""
    W = np.sqrt(raw["floor_area"].to_numpy(float) / raw["aspect_ratio"].to_numpy(float))
    L = raw["aspect_ratio"].to_numpy(float) * W
    d = PERIM
    return pd.DataFrame({"north": d * (L - d), "south": d * (L - d), "east": d * (W - d), "west": d * (W - d),
                         "core": (L - 2 * d) * (W - 2 * d)}, index=raw.index)


def zone_gap(cool, zones: pd.DataFrame, raw: pd.DataFrame) -> np.ndarray:
    """(sum(zone area x zone cooling) / floor area - cooling) / cooling."""
    a = zone_areas(raw)
    s = sum(a[z].to_numpy() * zones[ZONE_T[z]].to_numpy(float) for z in ZONES) / raw["floor_area"].to_numpy(float)
    cool = np.asarray(cool, dtype=float)
    return (s - cool) / cool


def lhs_designs(n_per_climate: int, climates: list[str], seed: int) -> pd.DataFrame:
    """Random designs inside design_space.yaml (scipy LHS; n_floors = floor of uniform [1, 6))."""
    from scipy.stats import qmc
    from surrogate.uncertainty import design_bounds
    b = design_bounds()
    parts = []
    for j, c in enumerate(sorted(climates)):
        u = qmc.LatinHypercube(d=len(FE.DESIGN_INPUTS), seed=seed + j).random(n_per_climate)
        frame = pd.DataFrame({inp: b.loc[inp, "min"] + u[:, k] * (b.loc[inp, "max"] - b.loc[inp, "min"])
                              for k, inp in enumerate(FE.DESIGN_INPUTS)})
        frame["n_floors"] = np.floor(1 + u[:, FE.DESIGN_INPUTS.index("n_floors")] * 5).clip(1, 5)
        frame["climate_key"] = c
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def _gap_rows(source: str, climate, gap, cool) -> pd.DataFrame:
    keep = np.asarray(cool, float) >= MIN_COOL
    return pd.DataFrame({"source": source, "climate_key": np.asarray(climate)[keep], "gap": np.asarray(gap)[keep]})


def stage_zones(dev: bool, log=print):
    from surrogate import uncertainty as U
    df, _ = load(dev)
    raw = df[["climate_key", *FE.RAW_INPUTS]]
    parts = [_gap_rows("energyplus", df["climate_key"], zone_gap(df[COOL], df, raw), df[COOL])]
    oof = {t: U.pick_oof(t, "grouped", dev=False).set_index("run_id")["yhat"] for t in [COOL, *ZONE_T.values()]}
    runs = df["run_id"].to_numpy()
    zones_hat = pd.DataFrame({t: oof[t].reindex(runs).to_numpy() for t in ZONE_T.values()})
    cool_hat = oof[COOL].reindex(runs).to_numpy()
    parts.append(_gap_rows("grouped_oof", df["climate_key"], zone_gap(cool_hat, zones_hat, raw.reset_index(drop=True)),
                           cool_hat))
    fm = final_models(dev)
    new = lhs_designs(10 if dev else int(_Z["new_designs"]["per_climate"]), sorted(df["climate_key"].unique()),
                      int(_Z["new_designs"]["seed"]))
    rnew = raw_frame(new, new["climate_key"])
    pred = fm.predict(rnew, [COOL, *ZONE_T.values()])
    parts.append(_gap_rows("final_models_new_designs", new["climate_key"], zone_gap(pred[COOL], pred, rnew), pred[COOL]))
    g = pd.concat(parts, ignore_index=True)
    g["abs_gap"] = g["gap"].abs()
    summ = lambda x: pd.Series({"n": len(x), "median_abs_gap": x["abs_gap"].median(),  # noqa: E731
                                "p95_abs_gap": x["abs_gap"].quantile(0.95), "mean_gap": x["gap"].mean()})
    per_c = g.groupby(["source", "climate_key"]).apply(summ, include_groups=False).reset_index()
    overall = g.groupby("source").apply(summ, include_groups=False).reset_index()
    d = out_dir(dev)
    per_c.to_csv(d / "zone_sum.csv", index=False)
    record_stage(d, "zones", {"overall": overall.to_dict("records")}, dev)
    for _, r in overall.iterrows():
        log(f"  zone sum {r['source']:26s} median |gap| {r['median_abs_gap']:.2%}, 95th pct {r['p95_abs_gap']:.2%}")
    return overall


# ── 5. SHAP in user inputs (H2, H3) ─────────────────────────────────────────────────────────────

def design_groups() -> dict[str, list[str]]:
    from eda import eda_lib as L
    g = {k: list(v) for k, v in L.FEATURE_GROUPS.items() if k != "climate"}
    flat = [c for v in g.values() for c in v]
    if sorted(flat) != sorted(FE.DESIGN_INPUTS):
        raise AssertionError("eda_lib design groups do not cover the 21 design inputs exactly once")
    return g


def shap_settings(dev: bool, fallback: bool) -> dict:
    if dev:
        return {**DEV_SHAP}
    s = {"rows_per_climate": int(_H["rows_per_climate"]), "background_overall": int(_H["background"]["overall"]),
         "background_per_climate": int(_H["background"]["per_climate"]), "max_evals": MAX_EVALS}
    if fallback:
        s.update({k: int(v) for k, v in GUARD["fallback"].items()})
    return s


def sample_rows(df: pd.DataFrame, climates: list[str], n_per_climate: int, seed: int) -> pd.DataFrame:
    """Stratified random training rows (climates in sorted order, one RNG)."""
    return sweep_starts(df[df["climate_key"].isin(climates)], n_per_climate, seed)


def sample_background(df: pd.DataFrame, climates: list[str], n: int, seed: int) -> pd.DataFrame:
    g = df[df["climate_key"].isin(climates)]
    rng = np.random.default_rng(seed)
    return g.iloc[np.sort(rng.choice(len(g), size=min(n, len(g)), replace=False))].reset_index(drop=True)


class OverallModel:
    """f(Z) for the overall view: Z = 21 design inputs + one climate code (index into `climates`).

    Swapping the climate player swaps the whole climate (7 inputs + 7 indices) - never a mixed climate."""

    def __init__(self, fm: FinalModels, name: str, climates: list[str]):
        self.fm, self.name, self.climates = fm, name, list(climates)
        self.targets = object_targets(name)
        self.calls = 0

    def codes_to_keys(self, codes) -> np.ndarray:
        idx = np.rint(np.asarray(codes, dtype=float)).astype(int)
        if (np.abs(np.asarray(codes, float) - idx) > 1e-9).any() or idx.min() < 0 or idx.max() >= len(self.climates):
            raise AssertionError("climate player received a value that is not a training-climate code")
        return np.asarray(self.climates)[idx]

    def frame(self, Z) -> pd.DataFrame:
        Z = np.asarray(Z, dtype=float)
        designs = pd.DataFrame(Z[:, :len(FE.DESIGN_INPUTS)], columns=FE.DESIGN_INPUTS)
        designs["n_floors"] = np.rint(designs["n_floors"])
        return raw_frame(designs, self.codes_to_keys(Z[:, -1]))

    def __call__(self, Z):
        self.calls += 1
        return self.fm.predict_object(self.name, self.frame(Z)).to_numpy()

    def encode(self, rows: pd.DataFrame) -> np.ndarray:
        code = {c: i for i, c in enumerate(self.climates)}
        return np.column_stack([rows[FE.DESIGN_INPUTS].to_numpy(dtype=float),
                                rows["climate_key"].map(code).to_numpy(dtype=float)])


class ClimateModel:
    """f(Z) for the per-climate view: Z = 21 design inputs, climate fixed."""

    def __init__(self, fm: FinalModels, name: str, climate: str):
        self.fm, self.name, self.climate = fm, name, climate
        self.targets = object_targets(name)

    def __call__(self, Z):
        designs = pd.DataFrame(np.asarray(Z, dtype=float), columns=FE.DESIGN_INPUTS)
        designs["n_floors"] = np.rint(designs["n_floors"])
        return self.fm.predict_object(self.name, raw_frame(designs, self.climate)).to_numpy()

    @staticmethod
    def encode(rows: pd.DataFrame) -> np.ndarray:
        return rows[FE.DESIGN_INPUTS].to_numpy(dtype=float)


def permutation_shap(f, X_explain: np.ndarray, background: np.ndarray, max_evals: int, seed: int = SHAP_SEED):
    """shap.PermutationExplainer with an independent background masker -> (values (n, M, k), base (n, k))."""
    import shap
    masker = shap.maskers.Independent(background, max_samples=len(background))
    expl = shap.PermutationExplainer(f, masker, seed=seed)
    ex = expl(X_explain, max_evals=max_evals, silent=True)
    vals = np.asarray(ex.values, dtype=float)
    base = np.asarray(ex.base_values, dtype=float)
    if vals.ndim == 2:
        vals, base = vals[:, :, None], base.reshape(-1, 1)
    return vals, base.reshape(len(X_explain), -1)


def check_additivity(vals, base, pred) -> float:
    """Largest relative gap |base + sum(SHAP) - prediction| / max(|prediction|, 1)."""
    gap = np.abs(base + vals.sum(axis=1) - pred) / np.maximum(np.abs(pred), 1.0)
    return float(gap.max())


def object_climates(name: str, heat: list[str], all_climates: list[str]) -> list[str]:
    return sorted(heat) if any(t in HEATING_TARGETS for t in object_targets(name)) else sorted(all_climates)


def explain_object(fm: FinalModels, name: str, view: str, df: pd.DataFrame, heat: list[str], s: dict,
                   n_rows: int | None = None, log=print) -> tuple[pd.DataFrame, dict]:
    """SHAP for one fitted object and one view -> long frame (run_id, climate_key, target, player, shap,
    base, prediction) + timing/additivity info. n_rows: limit per climate (timing probe)."""
    all_c = sorted(df["climate_key"].astype(str).unique())
    climates = object_climates(name, heat, all_c)
    per = n_rows if n_rows is not None else s["rows_per_climate"]
    rows = sample_rows(df, climates, per, SHAP_SEED)
    out, t0, worst = [], time.perf_counter(), 0.0
    if view == "overall":
        f = OverallModel(fm, name, climates)
        bg = sample_background(df, climates, s["background_overall"], SHAP_SEED)
        groups = [(None, rows, f, f.encode(bg))]
        players = FE.DESIGN_INPUTS + [CLIMATE_PLAYER]
    else:
        groups = []
        for j, c in enumerate(climates):
            fc = ClimateModel(fm, name, c)
            bg = sample_background(df, [c], s["background_per_climate"], SHAP_SEED + 1 + j)
            groups.append((c, rows[rows["climate_key"] == c], fc, fc.encode(bg)))
        players = list(FE.DESIGN_INPUTS)
    for _, r, f, B in groups:
        if r.empty:
            continue
        Xe = f.encode(r)
        vals, base = permutation_shap(f, Xe, B, s["max_evals"])
        pred = f(Xe)
        worst = max(worst, check_additivity(vals, base, pred))
        for k, t in enumerate(f.targets):
            v = pd.DataFrame(vals[:, :, k], columns=players)
            v["run_id"] = r["run_id"].to_numpy()
            v["climate_key"] = r["climate_key"].to_numpy()
            v["target"], v["base"], v["prediction"] = t, base[:, k], pred[:, k]
            out.append(v)
    secs = time.perf_counter() - t0
    frame = pd.concat(out, ignore_index=True)
    frame.insert(0, "view", view)
    n = frame["run_id"].nunique()
    info = {"object": name, "view": view, "n_rows": int(n), "seconds": round(secs, 2),
            "seconds_per_row": secs / max(n, 1), "additivity_max_rel_gap": worst}
    if worst > ADD_TOL:
        raise AssertionError(f"SHAP additivity {name}/{view}: max relative gap {worst:.2e} > {ADD_TOL}")
    return frame, info


def stage_shap_probe(dev: bool, log=print) -> dict:
    """H2 runtime guard: time `probe_rows` rows per object and view, project the full run."""
    fm = final_models(dev)
    df, heat = load(dev)
    s = shap_settings(dev, fallback=False)
    probe_n = 1 if dev else int(GUARD["probe_rows"])
    proj = []
    for name in OBJECTS:
        for view in ("overall", "per_climate"):
            n_c = len(object_climates(name, heat, sorted(df["climate_key"].unique())))
            per = max(1, probe_n // n_c) if view == "overall" else 1
            _, info = explain_object(fm, name, view, df, heat, s, n_rows=per)
            total_rows = s["rows_per_climate"] * n_c
            info["projected_hours"] = info["seconds_per_row"] * total_rows / 3600.0
            proj.append(info)
            log(f"  probe {name:12s} {view:11s} {info['n_rows']:3d} rows, {info['seconds_per_row']:.2f} s/row "
                f"-> {info['projected_hours']:.2f} h for {total_rows} rows")
    worst = max(p["projected_hours"] for p in proj)
    fallback = (not dev) and worst > float(GUARD["max_hours_per_object_view"])
    res = {"fallback": bool(fallback), "max_projected_hours": worst, "projection": proj,
           "rule": f"fallback if any object/view > {GUARD['max_hours_per_object_view']} h",
           "settings_used": shap_settings(dev, fallback)}
    d = out_dir(dev)
    (d / "shap_timing.json").write_text(json.dumps(res, indent=2, default=_json_default), encoding="utf-8")
    record_stage(d, "shap_probe", {"fallback": res["fallback"], "max_projected_hours": worst}, dev)
    log(f"  -> fallback {'YES' if fallback else 'no'} (largest projection {worst:.2f} h); settings {res['settings_used']}")
    return res


def group_shares(frame: pd.DataFrame, players: list[str], by: list[str]) -> pd.DataFrame:
    """Share of mean |group SHAP| per group (group value = sum of its players per row), as 2.8."""
    groups = design_groups()
    if CLIMATE_PLAYER in players:
        groups = {CLIMATE_PLAYER: [CLIMATE_PLAYER], **groups}
    gv = pd.DataFrame({g: frame[cols].sum(axis=1) for g, cols in groups.items()})
    gv[by] = frame[by].to_numpy()
    imp = gv.groupby(by)[list(groups)].apply(lambda x: x.abs().mean())
    return imp.div(imp.sum(axis=1), axis=0).reset_index()


def importance(frame: pd.DataFrame, players: list[str], by: list[str]) -> pd.DataFrame:
    return frame.groupby(by)[players].apply(lambda x: x.abs().mean()).reset_index()


def compare_2_8(overall: pd.DataFrame, per_climate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(i) per-climate design-group shares vs 2.8; (ii) Spearman of the 21 design inputs vs 2.8."""
    from scipy.stats import spearmanr
    ref = json.loads((SIM_ROOT / "outputs" / "eda" / "eda_metrics.json").read_text(encoding="utf-8"))
    imp = importance(overall, FE.DESIGN_INPUTS + [CLIMATE_PLAYER], ["target"]).set_index("target")
    k = int(_H["compare_2_8"]["top_k"])
    rank_rows = []
    for t in imp.index:
        ours = imp.loc[t, FE.DESIGN_INPUTS].astype(float)
        theirs = pd.Series(ref["shap_importance"][t])[FE.DESIGN_INPUTS].astype(float)
        rho = float(spearmanr(ours.to_numpy(), theirs.to_numpy()).statistic)
        rank_rows.append({"target": t, "spearman_21_design_inputs": rho,
                          "top5_3_6": ", ".join(ours.sort_values(ascending=False).index[:k]),
                          "top5_2_8": ", ".join(theirs.sort_values(ascending=False).index[:k]),
                          "top5_overlap": len(set(ours.nlargest(k).index) & set(theirs.nlargest(k).index))})
    shares = group_shares(per_climate, list(FE.DESIGN_INPUTS), ["target", "climate_key"])
    grp_rows = []
    for _, r in shares.iterrows():
        old = ref["shap_group_share_by_climate"].get(r["target"], {}).get(r["climate_key"])
        if old is None:
            continue
        for g in design_groups():
            grp_rows.append({"target": r["target"], "climate_key": r["climate_key"], "group": g,
                             "share_3_6": float(r[g]), "share_2_8": float(old[g]),
                             "difference": float(r[g]) - float(old[g])})
    return pd.DataFrame(rank_rows), pd.DataFrame(grp_rows)


def phase5_timing(fm: FinalModels, df: pd.DataFrame, heat: list[str], dev: bool) -> dict:
    """Seconds to explain ONE design for all 10 targets (per-climate view, 50 background rows)."""
    p = _H["phase5_timing"]
    n_bg = 6 if dev else int(p["background"])
    n_designs = 1 if dev else int(p["designs"])
    rows = sample_rows(df, sorted(heat), 1, SHAP_SEED + 99).head(n_designs)
    times = []
    for _, r in rows.iterrows():
        t0 = time.perf_counter()
        c = r["climate_key"]
        bg_all = sample_background(df, [c], n_bg, SHAP_SEED)
        for name in OBJECTS:
            f = ClimateModel(fm, name, c)
            permutation_shap(f, f.encode(r.to_frame().T), f.encode(bg_all), DEV_SHAP["max_evals"] if dev else MAX_EVALS)
        times.append(time.perf_counter() - t0)
    return {"designs": len(times), "background": n_bg, "seconds_per_design_median": float(np.median(times)),
            "seconds_per_design_max": float(np.max(times))}


def stage_shap(dev: bool, log=print):
    d = out_dir(dev)
    tpath = d / "shap_timing.json"
    if not tpath.exists():
        raise SystemExit("run `explain.py --stage shap --probe` first (H2 runtime guard)")
    timing = json.loads(tpath.read_text(encoding="utf-8"))
    s = shap_settings(dev, bool(timing["fallback"]))
    fm = final_models(dev)
    df, heat = load(dev)
    frames, infos = {"overall": [], "per_climate": []}, []
    for name in OBJECTS:
        for view in ("overall", "per_climate"):
            fr, info = explain_object(fm, name, view, df, heat, s)
            frames[view].append(fr)
            infos.append(info)
            log(f"  shap {name:12s} {view:11s} {info['n_rows']:4d} rows {info['seconds']:8.1f} s "
                f"(additivity gap {info['additivity_max_rel_gap']:.1e})")
    overall = pd.concat(frames["overall"], ignore_index=True)
    per_c = pd.concat(frames["per_climate"], ignore_index=True)
    overall.to_csv(d / "shap_overall.csv.gz", index=False)
    per_c.to_csv(d / "shap_by_climate.csv.gz", index=False)
    players_o = FE.DESIGN_INPUTS + [CLIMATE_PLAYER]
    imp = pd.concat([importance(overall, players_o, ["target"]).assign(view="overall"),
                     importance(per_c, FE.DESIGN_INPUTS, ["target"]).assign(view="per_climate")], ignore_index=True)
    imp.to_csv(d / "shap_importance.csv", index=False)
    shares = pd.concat([group_shares(overall, players_o, ["target"]).assign(view="overall", climate_key="all"),
                        group_shares(per_c, FE.DESIGN_INPUTS, ["target", "climate_key"]).assign(view="per_climate")],
                       ignore_index=True)
    shares.to_csv(d / "shap_group_share.csv", index=False)
    ranks, groups_cmp = compare_2_8(overall, per_c)
    ranks.to_csv(d / "shap_vs_2p8.csv", index=False)
    groups_cmp.to_csv(d / "shap_groups_vs_2p8.csv", index=False)
    p5 = phase5_timing(fm, df, heat, dev)
    timing["phase5"] = p5
    timing["runs"] = infos
    tpath.write_text(json.dumps(timing, indent=2, default=_json_default), encoding="utf-8")
    record_stage(d, "shap", {"settings": s, "fallback": bool(timing["fallback"]), "runs": infos, "phase5": p5,
                             "spearman_vs_2_8": ranks.set_index("target")["spearman_21_design_inputs"].to_dict()}, dev)
    for _, r in ranks.iterrows():
        log(f"  vs 2.8 {r['target']:28s} Spearman {r['spearman_21_design_inputs']:.2f}, top-5 overlap {r['top5_overlap']}")
    log(f"  one design, all 10 targets (Phase 5): {p5['seconds_per_design_median']:.1f} s")
    return ranks


# ── report ──────────────────────────────────────────────────────────────────────────────────────

def write_report(dev: bool) -> Path:
    d = out_dir(dev)
    st = read_result(dev)
    L = [f"# Sub-task 3.6 - interpretability and physics consistency{' (DEV - not results)' if dev else ''}", ""]
    if "fit" in st:
        f = st["fit"]
        L += ["## Final models (H1)", "", f"Fitted on {f['n_rows']} training rows; files in `{f['models_dir']}`.", "",
              "| Object | SHA-256 (first 12) | Fit s |", "|---|---|---|"]
        L += [f"| {k} | {v['sha256'][:12]} | {v['fit_s']} |" for k, v in f["objects"].items()]
        L += ["", "GPR heating features: " + ", ".join(f.get("gpr_selected_features") or []), ""]
    if (d / "oat_summary.json").exists():
        r = json.loads((d / "oat_summary.json").read_text(encoding="utf-8"))
        L += ["## OAT test (gate S5)", "",
              f"**S5 {r['status']}** - {r['n_pairs_informative']} informative (parameter, bound) pairs "
              f"({r['n_dropped_duplicate_pairs']} duplicate of base dropped); {r['n_counted']} counted deltas "
              f"(|EnergyPlus change| > {OAT_REL:.0%} of base); sign-correct {r['share_sign_correct']:.1%} "
              f"(needs >= {MIN_SIGN_SHARE:.0%}); strong set {r['n_strong_sign_correct']}/{r['n_counted_strong']}; "
              f"R2 of surrogate vs EnergyPlus change {r['r2_delta']:.3f}.", "",
              "| Target | Correct / counted |", "|---|---|"]
        L += [f"| {t} | {v['correct']} / {v['counted']} |" for t, v in r["by_target"].items()]
        for key, title in (("strong_misses", "Strong-set misses"), ("other_misses", "Other misses"),
                           ("zero_responses", "Zero responses (surrogate does not move)")):
            L += ["", f"{title}: {len(r[key])}"]
            L += [f"- {m['parameter']} {m.get('bound', '')} / {m['climate_key']} / {m['target']}: EnergyPlus "
                  f"{m['d_ep']:+.3f}" + (f", surrogate {m['d_sur']:+.3f}" if "d_sur" in m else "") for m in r[key]]
        L += [""]
    if (d / "sweep_summary.csv").exists():
        s = pd.read_csv(d / "sweep_summary.csv")
        L += ["## Monotonic sweeps (gate S6)", "", f"**S6 {st.get('sweeps', {}).get('S6', '?')}** "
              f"(WARN if more than {MAX_REVERSING:.0%} of designs reverse by more than the CV RMSE).", "",
              "| Input | Target | Expected | Share reversing | Largest reversal | CV RMSE |", "|---|---|---|---|---|---|"]
        L += [f"| {r.input} | {r.target} | {r.direction} | {r.share_reversing:.2%} | {r.max_reversal:.3f} | {r.rmse:.3f} |"
              for r in s.itertuples()]
        L += [""]
    if "zones" in st:
        L += ["## Zone-sum consistency (report)", "", "| Source | Rows | Median abs gap | 95th pct |", "|---|---|---|---|"]
        L += [f"| {r['source']} | {int(r['n'])} | {r['median_abs_gap']:.3%} | {r['p95_abs_gap']:.3%} |"
              for r in st["zones"]["overall"]]
        L += [""]
    if (d / "shap_vs_2p8.csv").exists():
        r = pd.read_csv(d / "shap_vs_2p8.csv")
        sh = pd.read_csv(d / "shap_group_share.csv")
        L += ["## SHAP in user inputs", "", "Overall view: share of mean |SHAP| per group (climate = one block).", "",
              "| Target | " + " | ".join(c for c in sh.columns if c not in ("target", "view", "climate_key")) + " |",
              "|---|" + "---|" * (len(sh.columns) - 3)]
        for _, x in sh[sh["view"] == "overall"].iterrows():
            L.append(f"| {x['target']} | " + " | ".join(f"{x[c]:.1%}" if pd.notna(x[c]) else "-"
                                                    for c in sh.columns if c not in ("target", "view", "climate_key")) + " |")
        L += ["", "Comparison with the 2.8 RF SHAP (21 design inputs):", "",
              "| Target | Spearman | Top-5 overlap | Top 5 (3.6) | Top 5 (2.8) |", "|---|---|---|---|---|"]
        L += [f"| {x.target} | {x.spearman_21_design_inputs:.2f} | {x.top5_overlap} | {x.top5_3_6} | {x.top5_2_8} |"
              for x in r.itertuples()]
        if "shap" in st:
            p5 = st["shap"]["phase5"]
            L += ["", f"Explaining one design (all 10 targets, per-climate view, {p5['background']} background rows): "
                      f"median {p5['seconds_per_design_median']:.1f} s."]
    path = d / "explain_report.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


# ── CLI ─────────────────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sub-task 3.6: final models, physics checks, SHAP")
    ap.add_argument("--stage", required=True, choices=["fit", "oat", "sweeps", "zones", "shap", "figures", "report"])
    ap.add_argument("--dev", action="store_true", help="small data and models -> outputs/explain_dev, models_dev")
    ap.add_argument("--probe", action="store_true", help="with --stage shap: timing probe only (H2 guard)")
    a = ap.parse_args(argv)
    if a.dev:
        print("[DEV] small data and models; numbers are NOT results", flush=True)
    t0 = time.time()
    log = lambda m: print(m, flush=True)  # noqa: E731
    if a.stage == "fit":
        stage_fit(a.dev, log)
    elif a.stage == "oat":
        stage_oat(a.dev, log)
    elif a.stage == "sweeps":
        stage_sweeps(a.dev, log)
    elif a.stage == "zones":
        stage_zones(a.dev, log)
    elif a.stage == "shap":
        stage_shap_probe(a.dev, log) if a.probe else stage_shap(a.dev, log)
    elif a.stage == "figures":
        from surrogate import explain_figures
        for p in explain_figures.make_all(out_dir(a.dev)):
            log(f"  wrote {p.name}")
    if a.stage in ("oat", "sweeps", "zones", "shap", "report") and not a.probe:
        write_report(a.dev)
    log(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
