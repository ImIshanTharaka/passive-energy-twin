"""
eda_lib.py - Computation library for Sub-task 2.8 (EDA, Phase 3 go/no-go).

Pure functions only: no figures, no file writes. run_eda.py (CLI) and
eda_notebook.ipynb (narrative) call these; test_eda.py checks them.
Plan: claude/Phase2.8_Execution_Plan.md (decisions E1-E7).

Sections (plan §4)
    A  coverage       coverage_uniformity, input_correlation
    C  physics        PHYSICS_RULES, evaluate_physics, core_roof_interaction
    D  orientation    facade_azimuth, azimuth_bin, facade_table, facade_azimuth_check
    E  signal         climate_baseline, design_skill, regression_metrics,
                      cv_grouped, cv_random, evaluate_predictions, shap_* , top_residuals
       gate           Check, gate_status, signal_checks

Guards
    E1  Every function that fits or scores a model calls require_training(); a
        holdout frame raises HoldoutLeakError. The holdout is descriptive only.
    E2  cv_grouped() asserts that no design_id is in both train and test.
    E7  Data comes only through load_split() -> dataset.assemble.load_dataset()
        (sha256-verified). Nothing here writes to outputs/dataset/.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.assemble import load_dataset, OUT_DIR as DATASET_DIR, MANIFEST_NAME  # noqa: E402

DESIGN_SPACE = ROOT / "config" / "design_space.yaml"

# ── Targets and thresholds (E4: fixed before results are seen) ────────────────
PRIMARY_TARGETS = ["heating_demand_kwh_m2", "cooling_demand_kwh_m2", "eui_kwh_m2"]
SECONDARY_TARGETS = ["peak_heating_w_m2", "peak_cooling_w_m2",
                     "zone_cooling_core_kwh_m2", "zone_cooling_south_kwh_m2", "zone_cooling_east_kwh_m2",
                     "zone_cooling_north_kwh_m2", "zone_cooling_west_kwh_m2"]
HEATING_TARGETS = ("heating_demand_kwh_m2", "peak_heating_w_m2")   # skill only over heating climates
HEATING_CLIMATE_MIN_MEDIAN = 1.0       # kWh/m2: climate counts as "heating" if its median exceeds this
R2_POOLED_MIN = 0.85                   # R1 (primary, FAIL) / R3 (secondary, WARN)
DESIGN_SKILL_MIN = 0.70                # R2 (primary, FAIL)
KS_ALPHA = 0.01                        # C1
INPUT_CORR_MAX = 0.15                  # C2 (2.4 achieved <= 0.009 per slice)
PHYSICS_ALPHA = 0.01                   # P-rules significance
O1_MIN_LATITUDE = 25.0                 # facade check applies north of this latitude

COLD_CLIMATES = ("chicago", "denver", "minneapolis", "new_york")
HOT_CLIMATES = ("chennai", "colombo", "miami", "phoenix")

# ── E5: fixed, untuned baseline ───────────────────────────────────────────────
SEED = 42
RF_PARAMS = dict(n_estimators=300, min_samples_leaf=2, max_features=1 / 3, n_jobs=-1, random_state=SEED)
RF_PARAMS_FAST = dict(RF_PARAMS, n_estimators=50)
N_FOLDS = 5
SHAP_PER_CLIMATE = 100
SHAP_PER_CLIMATE_FAST = 20

# ── Facades (building-relative names, design_space.yaml orientation note) ─────
# Base azimuth = the direction the facade faces at orientation 0 (deg clockwise from true north).
# EnergyPlus Building North Axis rotates the building clockwise, so the true azimuth is
# (base + orientation) mod 360; e.g. "north" faces east (90 deg) at orientation 90.
FACADES = {
    "north": {"base": 0.0, "wwr": "wwr_north", "overhang": "overhang_depth_n", "zone": "zone_cooling_north_kwh_m2"},
    "east": {"base": 90.0, "wwr": "wwr_east", "overhang": "overhang_depth_e", "zone": "zone_cooling_east_kwh_m2"},
    "south": {"base": 180.0, "wwr": "wwr_south", "overhang": "overhang_depth_s", "zone": "zone_cooling_south_kwh_m2"},
    "west": {"base": 270.0, "wwr": "wwr_west", "overhang": "overhang_depth_w", "zone": "zone_cooling_west_kwh_m2"},
}
AZIMUTH_BINS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]   # 45-deg bins centred on each direction

# ── E5: SHAP feature groups (every model input in exactly one group) ─────────
FEATURE_GROUPS = {
    "climate": ["hdd18", "cdd10", "annual_ghi", "mean_dbt", "mean_rh", "latitude", "mean_wind"],
    "geometry": ["floor_area", "aspect_ratio", "n_floors", "floor_height"],
    "opaque_envelope": ["wall_r_value", "roof_r_value", "floor_r_value", "wall_thermal_mass", "infiltration_ach"],
    "glazing": ["glazing_u", "glazing_shgc"],
    "window_area": ["wwr_north", "wwr_east", "wwr_south", "wwr_west"],
    "shading": ["overhang_depth_n", "overhang_depth_e", "overhang_depth_s", "overhang_depth_w", "sidefin_depth"],
    "orientation": ["orientation"],
}


class HoldoutLeakError(RuntimeError):
    """A holdout row reached a function that fits or scores a model (E1)."""


# ═════════════════════════════════════════════════════════════════════════════
# Loading
# ═════════════════════════════════════════════════════════════════════════════

def load_split(split: str, dataset_dir: Path = DATASET_DIR) -> tuple[pd.DataFrame, dict, str]:
    """(frame, column roles, sha256) via the verified 2.7 loader (E7)."""
    df, cols = load_dataset(split, dataset_dir)
    entry = json.loads((Path(dataset_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))["files"][split]
    return df, cols, entry["sha256"]


def require_training(df: pd.DataFrame) -> None:
    """E1 guard: only the training split (batch 'main') may be fitted or scored."""
    if "batch" not in df.columns:
        raise HoldoutLeakError("frame has no 'batch' column; cannot prove it is training data")
    bad = set(df["batch"].astype(str).unique()) - {"main"}
    if bad:
        raise HoldoutLeakError(f"non-training rows passed to a model function (batch {sorted(bad)}); "
                               "the holdout is descriptive only in 2.8 (E1)")


def design_bounds(path: Path = DESIGN_SPACE) -> dict[str, dict]:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["continuous_parameters"]
    return {k: {"min": float(v["min"]), "max": float(v["max"]), "integer": v.get("type") == "integer"}
            for k, v in spec.items()}


def heating_climates(df: pd.DataFrame, target: str = "heating_demand_kwh_m2",
                     min_median: float = HEATING_CLIMATE_MIN_MEDIAN) -> list[str]:
    """Climates whose median heating demand exceeds min_median (E4)."""
    med = df.groupby("climate_key")[target].median()
    return sorted(med[med > min_median].index.astype(str))


def stratified_sample(df: pd.DataFrame, per_climate: int, seed: int = SEED) -> pd.DataFrame:
    """per_climate rows from every climate (fixed seed), original order kept."""
    parts = [g.sample(n=min(per_climate, len(g)), random_state=seed) for _, g in df.groupby("climate_key")]
    return pd.concat(parts).sort_index()


# ═════════════════════════════════════════════════════════════════════════════
# A. Coverage
# ═════════════════════════════════════════════════════════════════════════════

def coverage_uniformity(df: pd.DataFrame, inputs: list[str], bounds: dict[str, dict]) -> pd.DataFrame:
    """C1: per (input, climate) test against the sampled uniform distribution.

    Continuous: one-sample KS against U(min, max). Integer: chi-square of level counts.
    """
    from scipy import stats
    rows = []
    for climate, g in df.groupby("climate_key"):
        for col in inputs:
            b = bounds[col]
            x = g[col].to_numpy(float)
            if b["integer"]:
                levels = np.arange(int(b["min"]), int(b["max"]) + 1)
                counts = np.array([(x == lv).sum() for lv in levels])
                stat, p = stats.chisquare(counts)
                test = "chi2"
            else:
                stat, p = stats.kstest(x, "uniform", args=(b["min"], b["max"] - b["min"]))
                test = "ks"
            rows.append({"input": col, "climate_key": str(climate), "test": test,
                         "stat": float(stat), "p": float(p), "n": int(len(x))})
    out = pd.DataFrame(rows)
    out["flag"] = out["p"] < KS_ALPHA
    return out


def input_correlation(df: pd.DataFrame, inputs: list[str]) -> pd.DataFrame:
    """C2: largest pairwise |Spearman rho| among design inputs, per climate."""
    rows = []
    for climate, g in df.groupby("climate_key"):
        rho = g[inputs].rank().corr().to_numpy(copy=True)   # pandas 3 copy-on-write: to_numpy() may be read-only
        np.fill_diagonal(rho, 0.0)
        i, j = np.unravel_index(np.abs(rho).argmax(), rho.shape)
        rows.append({"climate_key": str(climate), "max_abs_rho": float(abs(rho[i, j])),
                     "pair": f"{inputs[i]} ~ {inputs[j]}"})
    out = pd.DataFrame(rows)
    out["flag"] = out["max_abs_rho"] > INPUT_CORR_MAX
    return out


# ═════════════════════════════════════════════════════════════════════════════
# C. Physics directions
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Rule:
    id: str
    x: str
    y: str
    sign: int            # +1: y rises with x; -1: y falls
    where: str           # "all" | "heating" | "cold"
    severity: str        # "FAIL" (strong effect) | "WARN" (weaker effect)
    note: str = ""


def _physics_rules() -> list[Rule]:
    r = [Rule("P1", "glazing_shgc", "cooling_demand_kwh_m2", +1, "all", "FAIL", "solar gain raises cooling"),
         Rule("P2", "glazing_u", "heating_demand_kwh_m2", +1, "heating", "FAIL", "conductive loss"),
         Rule("P3-wall", "wall_r_value", "heating_demand_kwh_m2", -1, "heating", "FAIL", "insulation"),
         Rule("P3-roof", "roof_r_value", "heating_demand_kwh_m2", -1, "heating", "FAIL", "insulation")]
    for name, f in FACADES.items():
        r.append(Rule(f"P4-{name}", f["wwr"], f["zone"], +1, "all", "FAIL", "facade WWR reaches its own zone"))
    for name, f in FACADES.items():
        r.append(Rule(f"P5-{name}", f["overhang"], f["zone"], -1, "all", "WARN", "overhang shades its own zone"))
    r += [Rule("P6", "infiltration_ach", "heating_demand_kwh_m2", +1, "heating", "WARN", "small effect expected"),
          Rule("P7", "glazing_shgc", "heating_demand_kwh_m2", -1, "cold", "WARN", "solar gain offsets heating")]
    return r


PHYSICS_RULES = _physics_rules()


def _rule_climates(rule: Rule, all_climates: list[str], heat: list[str]) -> list[str]:
    if rule.where == "all":
        return all_climates
    if rule.where == "heating":
        return heat
    if rule.where == "cold":
        return [c for c in all_climates if c in COLD_CLIMATES]
    raise ValueError(rule.where)


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from scipy import stats
    res = stats.spearmanr(x, y)
    return float(res.statistic), float(res.pvalue)


def evaluate_physics(df: pd.DataFrame, rules: list[Rule] = PHYSICS_RULES,
                     heat: list[str] | None = None, alpha: float = PHYSICS_ALPHA) -> pd.DataFrame:
    """Spearman rho per (rule, climate). verdict: confirmed | weak | opposite.

    The design is decorrelated (C2), so the marginal rho estimates each parameter's direction.
    """
    climates = sorted(df["climate_key"].astype(str).unique())
    heat = heating_climates(df) if heat is None else heat
    rows = []
    for rule in rules:
        for c in _rule_climates(rule, climates, heat):
            g = df[df["climate_key"] == c]
            rho, p = spearman(g[rule.x].to_numpy(float), g[rule.y].to_numpy(float))
            if p < alpha and np.sign(rho) == rule.sign:
                verdict = "confirmed"
            elif p < alpha:
                verdict = "opposite"
            else:
                verdict = "weak"
            rows.append({"rule": rule.id, "x": rule.x, "y": rule.y, "expected": rule.sign,
                         "climate_key": c, "rho": rho, "p": p, "verdict": verdict, "severity": rule.severity})
    return pd.DataFrame(rows)


def physics_rule_status(results: pd.DataFrame) -> pd.DataFrame:
    """One status per rule.

    FAIL-rule: any 'opposite' -> FAIL; any 'weak' -> WARN; else PASS.
    WARN-rule: any 'opposite' -> WARN; any 'weak' -> INFO; else PASS.
    """
    rows = []
    for rule_id, g in results.groupby("rule", sort=False):
        sev = g["severity"].iloc[0]
        opp = g.loc[g["verdict"] == "opposite", "climate_key"].tolist()
        weak = g.loc[g["verdict"] == "weak", "climate_key"].tolist()
        if opp:
            status = sev
        elif weak:
            status = "WARN" if sev == "FAIL" else "INFO"
        else:
            status = "PASS"
        rows.append({"rule": rule_id, "severity": sev, "status": status, "opposite": opp, "weak": weak,
                     "rho_min": float(g["rho"].min()), "rho_max": float(g["rho"].max())})
    return pd.DataFrame(rows)


def core_roof_interaction(df: pd.DataFrame, climates=HOT_CLIMATES) -> pd.DataFrame:
    """P8 (INFO): roof_r -> core cooling, single-storey vs multi-storey, in hot climates.

    With one floor the windowless core sits under the roof (2.7 T4 outliers), so the roof
    effect on core cooling should be much stronger for n_floors == 1.
    """
    rows = []
    for c in climates:
        g = df[df["climate_key"] == c]
        if g.empty:
            continue
        for label, sub in (("n_floors=1", g[g["n_floors"] == 1]), ("n_floors>=2", g[g["n_floors"] >= 2])):
            rho, p = spearman(sub["roof_r_value"].to_numpy(float), sub["zone_cooling_core_kwh_m2"].to_numpy(float))
            rows.append({"climate_key": c, "subset": label, "n": len(sub), "rho": rho, "p": p,
                         "core_cooling_median": float(sub["zone_cooling_core_kwh_m2"].median())})
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# D. Orientation / facades
# ═════════════════════════════════════════════════════════════════════════════

def facade_azimuth(facade: str, orientation) -> np.ndarray | float:
    """True azimuth (deg clockwise from north) that a building-relative facade faces."""
    return np.mod(FACADES[facade]["base"] + np.asarray(orientation, float), 360.0)


def azimuth_bin(azimuth) -> np.ndarray:
    """8 compass bins of 45 deg centred on N, NE, ... (N = 337.5-22.5)."""
    idx = np.floor(np.mod(np.asarray(azimuth, float) + 22.5, 360.0) / 45.0).astype(int) % 8
    return np.array(AZIMUTH_BINS, dtype=object)[idx]


def facade_table(df: pd.DataFrame) -> pd.DataFrame:
    """Long format: one row per (run, facade) with true azimuth and solar-aperture residual.

    residual: per climate, zone cooling minus an OLS fit on [1, wwr, wwr*shgc, overhang]; this
    removes window size and glass, so what remains by azimuth is the orientation effect.
    """
    parts = []
    for name, f in FACADES.items():
        parts.append(pd.DataFrame({
            "run_id": df["run_id"].to_numpy(), "climate_key": df["climate_key"].astype(str).to_numpy(),
            "latitude": df["latitude"].to_numpy(float), "facade": name,
            "azimuth": facade_azimuth(name, df["orientation"].to_numpy(float)),
            "wwr": df[f["wwr"]].to_numpy(float), "shgc": df["glazing_shgc"].to_numpy(float),
            "overhang": df[f["overhang"]].to_numpy(float), "zone_cooling": df[f["zone"]].to_numpy(float)}))
    t = pd.concat(parts, ignore_index=True)
    t["bin"] = azimuth_bin(t["azimuth"].to_numpy())
    t["residual"] = np.nan
    for c, idx in t.groupby("climate_key").groups.items():
        s = t.loc[idx]
        A = np.column_stack([np.ones(len(s)), s["wwr"], s["wwr"] * s["shgc"], s["overhang"]])
        coef, *_ = np.linalg.lstsq(A, s["zone_cooling"].to_numpy(), rcond=None)
        t.loc[idx, "residual"] = s["zone_cooling"].to_numpy() - A @ coef
    return t


def facade_azimuth_check(table: pd.DataFrame, min_latitude: float = O1_MIN_LATITUDE) -> pd.DataFrame:
    """O1: north of min_latitude, mean residual of W/SW-facing facades > N-facing facades."""
    rows = []
    for c, g in table.groupby("climate_key"):
        means = g.groupby("bin")["residual"].mean().reindex(AZIMUTH_BINS)
        lat = float(g["latitude"].iloc[0])
        west = float(np.nanmean(means[["SW", "W"]].to_numpy()))
        north = float(means["N"])
        applies = lat > min_latitude
        rows.append({"climate_key": str(c), "latitude": lat, "applies": applies,
                     "mean_resid_W_SW": west, "mean_resid_N": north, "difference": west - north,
                     "pass": (west > north) if applies else None,
                     **{f"bin_{b}": float(means[b]) for b in AZIMUTH_BINS}})
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# E. Signal: baseline, metrics, CV, SHAP
# ═════════════════════════════════════════════════════════════════════════════

def climate_baseline(y: pd.Series, climate: pd.Series) -> pd.Series:
    """The 'lazy' predictor: each row gets its climate's mean (E3)."""
    return y.groupby(climate.to_numpy()).transform("mean")


def r2(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    sst = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((y - yhat) ** 2) / sst) if sst > 0 else float("nan")


def design_skill(y, yhat, climate) -> float:
    """1 - SSE(model) / SSE(climate-mean predictor): the share of within-climate variance explained.

    0 for the climate-mean predictor itself, 1 for a perfect model, < 0 if worse than climate means.
    """
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    base = climate_baseline(pd.Series(y), pd.Series(np.asarray(climate))).to_numpy()
    sse_base = np.sum((y - base) ** 2)
    return float(1.0 - np.sum((y - yhat) ** 2) / sse_base) if sse_base > 0 else float("nan")


def regression_metrics(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return {"r2": r2(y, yhat), "mae": float(np.mean(np.abs(y - yhat))), "rmse": rmse,
            "cv_rmse": rmse / float(np.mean(y)) if np.mean(y) != 0 else float("nan"), "n": int(len(y))}


def _rf(params: dict):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(**params)


def cv_grouped(df: pd.DataFrame, inputs: list[str], targets: list[str], group: str = "design_id",
               params: dict = RF_PARAMS, n_folds: int = N_FOLDS, seed: int = SEED) -> pd.DataFrame:
    """E2: out-of-fold predictions, GroupKFold on design_id. One RF per target.

    Returns long frame: run_id, climate_key, target, fold, y, yhat.
    """
    from sklearn.model_selection import GroupKFold
    require_training(df)
    X = df[inputs].to_numpy(float)
    groups = df[group].astype(str).to_numpy()
    folds = list(GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X, groups=groups))
    for tr, te in folds:
        overlap = set(groups[tr]) & set(groups[te])
        if overlap:
            raise AssertionError(f"design_id in both train and test: {sorted(overlap)[:5]}")
    return _oof(df, X, targets, folds, params, scheme="grouped")


def cv_random(df: pd.DataFrame, inputs: list[str], targets: list[str], params: dict = RF_PARAMS,
              test_size: float = 0.2, seed: int = SEED) -> pd.DataFrame:
    """E2 comparison only: one random 80/20 row split (anchor designs can straddle it)."""
    from sklearn.model_selection import train_test_split
    require_training(df)
    X = df[inputs].to_numpy(float)
    tr, te = train_test_split(np.arange(len(df)), test_size=test_size, random_state=seed)
    return _oof(df, X, targets, [(tr, te)], params, scheme="random")


def _oof(df, X, targets, folds, params, scheme) -> pd.DataFrame:
    parts = []
    for t in targets:
        y = df[t].to_numpy(float)
        for k, (tr, te) in enumerate(folds):
            m = _rf(params).fit(X[tr], y[tr])
            parts.append(pd.DataFrame({"run_id": df["run_id"].to_numpy()[te],
                                       "climate_key": df["climate_key"].astype(str).to_numpy()[te],
                                       "target": t, "fold": k, "y": y[te], "yhat": m.predict(X[te])}))
    out = pd.concat(parts, ignore_index=True)
    out["scheme"] = scheme
    return out


def evaluate_predictions(oof: pd.DataFrame, heat: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-target summary, per-target-per-climate table).

    summary columns: r2 (pooled), mae, rmse, cv_rmse, design_skill, skill_climates,
                     climate_only_r2, within_r2_mean.
    Heating targets: design_skill and within_r2_mean over the heating climates only (E4).
    """
    summary, per_climate = [], []
    for t, g in oof.groupby("target", sort=False):
        m = regression_metrics(g["y"], g["yhat"])
        base = climate_baseline(g["y"].reset_index(drop=True), g["climate_key"].reset_index(drop=True))
        m["climate_only_r2"] = r2(g["y"], base)
        sk = g[g["climate_key"].isin(heat)] if t in HEATING_TARGETS else g
        m["design_skill"] = design_skill(sk["y"], sk["yhat"], sk["climate_key"])
        m["skill_climates"] = int(sk["climate_key"].nunique())
        within = []
        for c, gc in g.groupby("climate_key"):
            row = {"target": t, "climate_key": c, **regression_metrics(gc["y"], gc["yhat"])}
            row["in_skill_set"] = c in set(sk["climate_key"])
            per_climate.append(row)
            if row["in_skill_set"]:
                within.append(row["r2"])
        m["within_r2_mean"] = float(np.mean(within)) if within else float("nan")
        summary.append({"target": t, **m})
    return pd.DataFrame(summary), pd.DataFrame(per_climate)


def fit_full(df: pd.DataFrame, inputs: list[str], target: str, params: dict = RF_PARAMS):
    """RF on all training rows (for SHAP only; never scored)."""
    require_training(df)
    return _rf(params).fit(df[inputs].to_numpy(float), df[target].to_numpy(float))


def shap_values(model, X: pd.DataFrame) -> pd.DataFrame:
    """Exact TreeSHAP; returns a frame aligned with X (rows x inputs)."""
    import shap
    vals = shap.TreeExplainer(model).shap_values(X.to_numpy(float), check_additivity=False)
    return pd.DataFrame(vals, index=X.index, columns=X.columns)


def check_groups_cover(inputs: list[str], groups: dict = FEATURE_GROUPS) -> None:
    flat = [c for cols in groups.values() for c in cols]
    dup = sorted({c for c in flat if flat.count(c) > 1})
    missing, extra = sorted(set(inputs) - set(flat)), sorted(set(flat) - set(inputs))
    if dup or missing or extra:
        raise ValueError(f"feature groups: duplicated {dup}, missing {missing}, unknown {extra}")


def shap_grouped(sv: pd.DataFrame, groups: dict = FEATURE_GROUPS) -> pd.DataFrame:
    """Per-row group SHAP = sum of the group's feature SHAP values (additive, so exact)."""
    check_groups_cover(list(sv.columns), groups)
    return pd.DataFrame({g: sv[cols].sum(axis=1) for g, cols in groups.items()}, index=sv.index)


def shap_importance(sv: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per column, descending."""
    return sv.abs().mean().sort_values(ascending=False)


def shap_share(sv_grouped: pd.DataFrame) -> pd.Series:
    """Share of total mean |SHAP| per group (sums to 1)."""
    imp = sv_grouped.abs().mean()
    return (imp / imp.sum()).sort_values(ascending=False)


def top_residuals(oof: pd.DataFrame, diagnostics: pd.DataFrame | None, targets: list[str], n: int = 20) -> pd.DataFrame:
    """R4: largest |y - yhat| per target, joined to 2.7 diagnostics (outlier flags, warnings)."""
    g = oof[oof["target"].isin(targets)].copy()
    g["abs_err"] = (g["y"] - g["yhat"]).abs()
    top = g.sort_values("abs_err", ascending=False).groupby("target", sort=False).head(n)
    if diagnostics is not None:
        keep = [c for c in ("run_id", "outlier_targets", "err_warnings", "accepted_exception") if c in diagnostics]
        top = top.merge(diagnostics[keep], on="run_id", how="left")
    return top.reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# Gate
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Check:
    id: str
    name: str
    severity: str      # what a failure of this check means: FAIL | WARN | INFO
    status: str        # PASS | FAIL | WARN | INFO
    detail: str = ""
    data: dict = field(default_factory=dict)


def gate_status(checks: list[Check]) -> str:
    """NO-GO if any FAIL, REVIEW if any WARN, else GO (same convention as 2.5-2.7)."""
    statuses = {c.status for c in checks}
    if "FAIL" in statuses:
        return "NO-GO"
    if "WARN" in statuses:
        return "REVIEW"
    return "GO"


def check_rows(df: pd.DataFrame, holdout: pd.DataFrame) -> Check:
    ok = len(df) == 3000 and len(holdout) == 400 and set(df["batch"].astype(str)) == {"main"}
    return Check("G0", "data loaded via load_dataset (sha256 verified)", "FAIL", "PASS" if ok else "FAIL",
                 f"training {len(df)} rows, holdout {len(holdout)} rows")


def check_coverage(cov: pd.DataFrame) -> Check:
    flagged = cov[cov["flag"]]
    return Check("C1", "input uniformity per climate (KS / chi2)", "WARN", "WARN" if len(flagged) else "PASS",
                 f"{len(flagged)}/{len(cov)} (input, climate) pairs with p < {KS_ALPHA}; min p {cov['p'].min():.3g}",
                 {"flagged": flagged[["input", "climate_key", "p"]].to_dict("records")})


def check_correlation(cor: pd.DataFrame) -> Check:
    worst = cor.loc[cor["max_abs_rho"].idxmax()]
    return Check("C2", "design-input decorrelation per climate", "WARN", "WARN" if cor["flag"].any() else "PASS",
                 f"max |rho| {worst['max_abs_rho']:.3f} ({worst['pair']}, {worst['climate_key']}); limit {INPUT_CORR_MAX}")


def physics_checks(status: pd.DataFrame) -> list[Check]:
    out = []
    for _, r in status.iterrows():
        detail = f"rho {r['rho_min']:+.2f}..{r['rho_max']:+.2f}"
        if r["opposite"]:
            detail += f"; OPPOSITE sign in {r['opposite']}"
        if r["weak"]:
            detail += f"; not significant in {r['weak']}"
        out.append(Check(r["rule"], f"physics direction {r['rule']}", r["severity"], r["status"], detail))
    return out


def check_facades(fac: pd.DataFrame) -> Check:
    a = fac[fac["applies"]]
    failed = a.loc[~a["pass"].astype(bool), "climate_key"].tolist()
    return Check("O1", "W/SW-facing facades cool more than N-facing (lat > 25)", "WARN",
                 "WARN" if failed else "PASS",
                 f"{len(a) - len(failed)}/{len(a)} climates pass" + (f"; fail: {failed}" if failed else ""))


def signal_checks(summary: pd.DataFrame) -> list[Check]:
    """R1-R3 from the grouped-CV summary (E3/E4)."""
    s = summary.set_index("target")
    out = []
    for t in PRIMARY_TARGETS:
        r = s.loc[t]
        out.append(Check(f"R1:{t}", f"pooled R2 >= {R2_POOLED_MIN}", "FAIL",
                         "PASS" if r["r2"] >= R2_POOLED_MIN else "FAIL", f"R2 {r['r2']:.3f}"))
        out.append(Check(f"R2:{t}", f"design skill >= {DESIGN_SKILL_MIN}", "FAIL",
                         "PASS" if r["design_skill"] >= DESIGN_SKILL_MIN else "FAIL",
                         f"skill {r['design_skill']:.3f} over {int(r['skill_climates'])} climates"))
    for t in SECONDARY_TARGETS:
        if t in s.index:
            r = s.loc[t]
            out.append(Check(f"R3:{t}", f"pooled R2 >= {R2_POOLED_MIN}", "WARN",
                             "PASS" if r["r2"] >= R2_POOLED_MIN else "WARN",
                             f"R2 {r['r2']:.3f}, design skill {r['design_skill']:.3f}"))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Diagnosis and decision (Option C, agreed 17 Oct 2026, before the official run)
# ═════════════════════════════════════════════════════════════════════════════

LEARNING_FRACTIONS = (0.4, 0.7, 1.0)
DATA_CHECK_PREFIXES = ("G0", "C1", "C2", "P", "O1")      # failures here mean the DATA is suspect
MODEL_CHECK_PREFIXES = ("R1:", "R2:", "R3:")             # failures here can be the baseline MODEL


def learning_curve_skill(df: pd.DataFrame, inputs: list[str], targets: list[str], heat: list[str],
                         params: dict = RF_PARAMS, fractions=LEARNING_FRACTIONS, seed: int = SEED) -> pd.DataFrame:
    """Design skill (grouped CV) when training on a growing random share of the designs.

    Simulations are deterministic, so if skill still rises with more designs the shortfall
    is model capacity / sample size, not noise in the data.
    """
    require_training(df)
    designs = np.random.default_rng(seed).permutation(df["design_id"].astype(str).unique())
    rows = []
    for frac in fractions:
        keep = set(designs[: int(round(frac * len(designs)))])
        sub = df[df["design_id"].astype(str).isin(keep)].reset_index(drop=True)
        summ, _ = evaluate_predictions(cv_grouped(sub, inputs, targets, params=params, seed=seed), heat)
        for _, r in summ.iterrows():
            rows.append({"fraction": frac, "n_designs": len(keep), "n_rows": len(sub), "target": r["target"],
                         "design_skill": r["design_skill"], "r2": r["r2"]})
    return pd.DataFrame(rows)


def _is(check_id: str, prefixes) -> bool:
    return any(check_id.startswith(p) for p in prefixes)


def decide(checks: list[Check], learning: pd.DataFrame | None) -> dict:
    """Pre-registered decision rule (plan §5.1, Option C).

    GO / REVIEW follow the gate. On NO-GO:
      - any FAIL in a data check (G0, C*, P*, O1)          -> NO-GO: fix the data
      - only R1/R2 FAILs, and design skill rises with every
        increase in training designs for each failing target -> PROCEED (documented exception)
      - otherwise                                           -> NO-GO: investigate
    The gate itself is never altered; the decision is recorded next to it.
    """
    gate = gate_status(checks)
    failed = [c.id for c in checks if c.status == "FAIL"]
    out = {"gate": gate, "failed_checks": failed, "rule": "Option C (plan §5.1, agreed 17 Oct 2026)"}
    if gate == "GO":
        return {**out, "decision": "GO", "reason": "all checks pass"}
    if gate == "REVIEW":
        return {**out, "decision": "REVIEW", "reason": "warnings only; review them before Phase 3"}
    data_fail = [f for f in failed if _is(f, DATA_CHECK_PREFIXES)]
    if data_fail:
        return {**out, "decision": "NO-GO", "reason": f"data checks failed: {data_fail}; fix the data first"}
    model_fail = [f for f in failed if _is(f, ("R1:", "R2:"))]
    targets = sorted({f.split(":", 1)[1] for f in model_fail})
    rising = {}
    if learning is not None and len(learning):
        for t in targets:
            s = learning[learning["target"] == t].sort_values("fraction")["design_skill"].to_numpy()
            rising[t] = bool(len(s) >= 2 and np.all(np.diff(s) > 0))
    if targets and rising and all(rising.get(t, False) for t in targets):
        return {**out, "decision": "PROCEED (documented exception)", "rising_with_data": rising,
                "reason": ("all data checks pass; only baseline-model checks failed "
                           f"({model_fail}) and design skill still rises with training size, so the shortfall "
                           "is attributed to the untuned RF, not the dataset. Phase 3 must exceed the thresholds.")}
    return {**out, "decision": "NO-GO", "rising_with_data": rising,
            "reason": "model checks failed and skill does not rise with more designs; investigate before Phase 3"}
