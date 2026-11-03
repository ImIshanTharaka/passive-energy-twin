"""
features.py - physics-based feature engineering (Phase 3, Sub-task 3.2; plan M3).

One deterministic function turns the 28 model inputs (21 design + 7 climate) into the
engineered features used by every surrogate. Training (3.2-3.7), the Phase 4 optimiser and
the Phase 5 API all call `engineer()`, so there is no train/serve skew.

    from surrogate.features import engineer, FEATURE_SETS
    F = engineer(df)                     # DataFrame with the 28 inputs (extra columns ignored)
    X = F[FEATURE_SETS["S"]]             # one of the 8 ablation sets
    f1 = engineer({"floor_area": 5000.0, ...})   # a single design dict -> 1-row DataFrame

Every formula reproduces what the Phase 2 generator built (definitions and the generator facts
they rest on: claude/Phase3.2_Execution_Plan.md section 3). Features use ONLY the 28 inputs and
fixed archetype constants - never a target - so they cannot leak and cost nothing at inference.

Groups (plan D2):  G geometry (8) . S solar/orientation (24) . U heat loss (6) . C climate x building (5)
Sets:  R, G, S, U, C (cumulative ladder) and C-G, C-S, C-U (leave one group out of the full set).

Units: lengths m, areas m2; "per floor area" = per m2 of conditioned floor; UA terms in W/K per m2 floor.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

# Pure-data generator modules (no eppy). Importing them keeps the F-factor table and the shading
# threshold identical to Phase 2 by construction.
from idf_generator.construction.constructions import F_IP_TO_SI, FFACTOR_TABLE_IP, R_SI_TO_IP
from idf_generator.fenestration.fenestration import MIN_SHADING_DEPTH

# ── Inputs (dataset_manifest.json columns["inputs"], same order) ─────────────────────────────

DESIGN_INPUTS: list[str] = [
    "floor_area", "aspect_ratio", "n_floors", "floor_height",
    "wall_r_value", "roof_r_value", "floor_r_value", "wall_thermal_mass", "infiltration_ach",
    "wwr_north", "wwr_east", "wwr_south", "wwr_west", "glazing_u", "glazing_shgc",
    "overhang_depth_n", "overhang_depth_e", "overhang_depth_s", "overhang_depth_w",
    "sidefin_depth", "orientation",
]
CLIMATE_INPUTS: list[str] = ["hdd18", "cdd10", "annual_ghi", "mean_dbt", "mean_rh", "latitude", "mean_wind"]
RAW_INPUTS: list[str] = DESIGN_INPUTS + CLIMATE_INPUTS

# ── Fixed archetype / physics constants ─────────────────────────────────────────────────────
# Values that also live in the generator are checked against it in tests/test_features.py.

PERIM_DEPTH = 4.57                 # m, geometry.build_geometry default (DOE 15 ft)
ACH50_TO_NATURAL = 20.0            # generator.ACH50_TO_NATURAL_DIVISOR (imports eppy, so copied + tested)
R_FILMS_WALL = 0.17                # m2K/W, ISO 6946 Rsi 0.13 + Rse 0.04 (horizontal heat flow)   (F1)
R_FILMS_ROOF = 0.14                # m2K/W, ISO 6946 Rsi 0.10 + Rse 0.04 (upward heat flow)       (F1)
RHO_CP_AIR = 1.2 * 1006.0          # J/(m3 K), air volumetric heat capacity

# Building-relative facades: suffix, wwr column, overhang column, which footprint side is the facade
# length (N/S facades span the length L, E/W the width W - geometry.WALL_ADJACENCY) and the facade's
# outward azimuth relative to the building north axis.
FACADES: tuple[tuple[str, str, str, str, float], ...] = (
    ("n", "wwr_north", "overhang_depth_n", "L", 0.0),
    ("e", "wwr_east", "overhang_depth_e", "W", 90.0),
    ("s", "wwr_south", "overhang_depth_s", "L", 180.0),
    ("w", "wwr_west", "overhang_depth_w", "W", 270.0),
)
TRUE_DIRECTIONS: tuple[tuple[str, float], ...] = (("N", 0.0), ("E", 90.0), ("S", 180.0), ("W", 270.0))

# ── Feature names per group (column order of engineer()) ────────────────────────────────────

GEOMETRY: list[str] = [
    "geo_length", "geo_width", "geo_total_floor_area", "geo_wall_to_floor",
    "geo_perimeter_fraction", "geo_inv_floors", "geo_single_storey", "geo_compactness",
]
SOLAR: list[str] = (
    [f"sol_glazed_ratio_{f[0]}" for f in FACADES] + ["sol_glazed_ratio"]
    + [f"sol_aperture_{f[0]}" for f in FACADES] + ["sol_aperture"]
    + ["sol_orient_sin", "sol_orient_cos"]
    + [f"sol_aperture_true_{d}" for d, _ in TRUE_DIRECTIONS]
    + [f"sol_pf_overhang_{f[0]}" for f in FACADES]
    + [f"sol_pf_fin_{f[0]}" for f in FACADES]
)
HEAT_LOSS: list[str] = ["ua_wall", "ua_roof", "ua_glass", "ua_ground", "ua_infiltration", "ua_total"]
CLIMATE_X: list[str] = ["cx_ua_hdd", "cx_aperture_ghi", "cx_aperture_cdd", "cx_infil_hdd", "cx_infil_rh"]

GROUPS: dict[str, list[str]] = {"G": GEOMETRY, "S": SOLAR, "U": HEAT_LOSS, "C": CLIMATE_X}
ENGINEERED: list[str] = GEOMETRY + SOLAR + HEAT_LOSS + CLIMATE_X
ALL_FEATURES: list[str] = RAW_INPUTS + ENGINEERED

FEATURE_GROUPS: dict[str, str] = {**{c: "R" for c in RAW_INPUTS},
                                  **{c: g for g, cols in GROUPS.items() for c in cols}}


def _set(*groups: str) -> list[str]:
    return RAW_INPUTS + [c for g in groups for c in GROUPS[g]]


# Plan D2: forward ladder (cumulative) + leave-one-group-out from the full set. C-C equals U.
FEATURE_SETS: dict[str, list[str]] = {
    "R": _set(),
    "G": _set("G"),
    "S": _set("G", "S"),
    "U": _set("G", "S", "U"),
    "C": _set("G", "S", "U", "C"),
    "C-G": _set("S", "U", "C"),
    "C-S": _set("G", "U", "C"),
    "C-U": _set("G", "S", "C"),
}
LADDER: list[str] = ["R", "G", "S", "U", "C"]
LEAVE_ONE_OUT: list[str] = ["C-G", "C-S", "C-U"]

# ── Raw inputs behind each feature (for SHAP in user inputs, plan M9 / 3.6) ─────────────────

_FOOT = ("floor_area", "aspect_ratio")
_WWR = tuple(f[1] for f in FACADES)


def _parents() -> dict[str, tuple[str, ...]]:
    p: dict[str, set[str]] = {c: {c} for c in RAW_INPUTS}
    p["geo_length"] = p["geo_width"] = set(_FOOT)
    p["geo_total_floor_area"] = {"floor_area", "n_floors"}
    p["geo_wall_to_floor"] = {*_FOOT, "floor_height"}
    p["geo_perimeter_fraction"] = set(_FOOT)
    p["geo_inv_floors"] = p["geo_single_storey"] = {"n_floors"}
    p["geo_compactness"] = {*_FOOT, "n_floors", "floor_height"}
    for sfx, wwr, oh, _, _ in FACADES:
        p[f"sol_glazed_ratio_{sfx}"] = {*_FOOT, "floor_height", wwr}
        p[f"sol_aperture_{sfx}"] = p[f"sol_glazed_ratio_{sfx}"] | {"glazing_shgc"}
        p[f"sol_pf_overhang_{sfx}"] = {oh, wwr, "floor_height"}
        p[f"sol_pf_fin_{sfx}"] = {"sidefin_depth", wwr, *_FOOT}
    p["sol_glazed_ratio"] = {*_FOOT, "floor_height", *_WWR}
    p["sol_aperture"] = p["sol_glazed_ratio"] | {"glazing_shgc"}
    p["sol_orient_sin"] = p["sol_orient_cos"] = {"orientation"}
    for d, _ in TRUE_DIRECTIONS:
        p[f"sol_aperture_true_{d}"] = p["sol_aperture"] | {"orientation"}
    p["ua_wall"] = {"wall_r_value", *_FOOT, "floor_height", *_WWR}
    p["ua_roof"] = {"roof_r_value", "n_floors"}
    p["ua_glass"] = {"glazing_u", *_FOOT, "floor_height", *_WWR}
    p["ua_ground"] = {"floor_r_value", *_FOOT, "n_floors"}
    p["ua_infiltration"] = {"infiltration_ach", "floor_height"}
    p["ua_total"] = set().union(*(p[c] for c in HEAT_LOSS[:-1]))
    p["cx_ua_hdd"] = (p["ua_total"] - {"infiltration_ach"}) | {"hdd18"}
    p["cx_aperture_ghi"] = p["sol_aperture"] | {"annual_ghi"}
    p["cx_aperture_cdd"] = p["sol_aperture"] | {"cdd10"}
    p["cx_infil_hdd"] = p["ua_infiltration"] | {"hdd18"}
    p["cx_infil_rh"] = p["ua_infiltration"] | {"mean_rh"}
    return {c: tuple(r for r in RAW_INPUTS if r in p[c]) for c in ALL_FEATURES}


RAW_PARENTS: dict[str, tuple[str, ...]] = _parents()

# ── Physics helpers (vectorised) ────────────────────────────────────────────────────────────


def ffactor(floor_r_value: np.ndarray) -> np.ndarray:
    """Slab F-factor (W/m.K), vectorised copy of constructions.ffactor_for_floor_r.

    Linear in 1/F between the ASHRAE 90.1 Table A6.3.1 points (IP units); the last segment is
    extended beyond R-20 IP, exactly as the generator does. Equality is tested."""
    r_ip = np.asarray(floor_r_value, float) * R_SI_TO_IP
    r_tab = np.array([r for r, _ in FFACTOR_TABLE_IP])
    inv_tab = 1.0 / np.array([f for _, f in FFACTOR_TABLE_IP])
    # segment index: first segment whose upper end >= r_ip, else the last one (extrapolation)
    seg = np.clip(np.searchsorted(r_tab[1:], r_ip, side="left"), 0, len(r_tab) - 2)
    r0, r1 = r_tab[seg], r_tab[seg + 1]
    inv = inv_tab[seg] + (inv_tab[seg + 1] - inv_tab[seg]) * (r_ip - r0) / (r1 - r0)
    return F_IP_TO_SI / inv


def _as_frame(inputs: pd.DataFrame | Mapping[str, float] | Sequence[Mapping[str, float]]) -> pd.DataFrame:
    if isinstance(inputs, pd.DataFrame):
        return inputs
    if isinstance(inputs, Mapping):
        return pd.DataFrame([dict(inputs)])
    return pd.DataFrame([dict(r) for r in inputs])


def _validate(X: pd.DataFrame) -> None:
    missing = [c for c in RAW_INPUTS if c not in X.columns]
    if missing:
        raise KeyError(f"engineer(): missing input column(s) {missing}")
    vals = X[RAW_INPUTS].to_numpy(float)
    if not np.isfinite(vals).all():
        bad = sorted({RAW_INPUTS[j] for j in np.where(~np.isfinite(vals))[1]})
        raise ValueError(f"engineer(): non-finite values in {bad}")
    # Physical validity only (the design-space bounds are the applicability-domain check, 3.5/3.8).
    rules = {
        "floor_area": X["floor_area"] > 0, "aspect_ratio": X["aspect_ratio"] >= 1.0,
        "n_floors": (X["n_floors"] >= 1) & (X["n_floors"] == np.round(X["n_floors"])),
        "floor_height": X["floor_height"] > 0, "wall_r_value": X["wall_r_value"] > 0,
        "roof_r_value": X["roof_r_value"] > 0, "floor_r_value": X["floor_r_value"] > 0,
        "infiltration_ach": X["infiltration_ach"] >= 0, "glazing_u": X["glazing_u"] > 0,
        "glazing_shgc": (X["glazing_shgc"] > 0) & (X["glazing_shgc"] <= 1),
        "sidefin_depth": X["sidefin_depth"] >= 0,
    }
    for _, wwr, oh, _, _ in FACADES:
        rules[wwr] = (X[wwr] > 0) & (X[wwr] < 1)
        rules[oh] = X[oh] >= 0
    bad = [k for k, ok in rules.items() if not bool(np.all(ok))]
    if bad:
        raise ValueError(f"engineer(): physically invalid values in {bad}")
    width = np.sqrt(X["floor_area"] / X["aspect_ratio"])
    if bool(np.any(width - 2 * PERIM_DEPTH <= 0)):
        raise ValueError("engineer(): footprint too narrow for the 4.57 m perimeter zones "
                         "(geometry.compute_zone_footprints would fail)")


# ── Public API ──────────────────────────────────────────────────────────────────────────────


def engineer(inputs: pd.DataFrame | Mapping[str, float] | Sequence[Mapping[str, float]]) -> pd.DataFrame:
    """Return the 28 raw inputs followed by the 43 engineered features (ALL_FEATURES order).

    `inputs`: a DataFrame holding at least the 28 RAW_INPUTS (other columns are ignored and never
    read), one design dict, or a list of dicts. The index of a DataFrame input is preserved."""
    X = _as_frame(inputs)
    _validate(X)
    v = {c: X[c].to_numpy(float) for c in RAW_INPUTS}
    out: dict[str, np.ndarray] = dict(v)

    # G - geometry (geometry.compute_building_dimensions / compute_zone_footprints)
    fa, ar, n, h = v["floor_area"], v["aspect_ratio"], v["n_floors"], v["floor_height"]
    W = np.sqrt(fa / ar)
    L = ar * W
    side = {"L": L, "W": W}
    a_fl = L * W                                         # = floor_area (per storey)
    perim = 2.0 * (L + W)
    out["geo_length"], out["geo_width"] = L, W
    out["geo_total_floor_area"] = a_fl * n
    out["geo_wall_to_floor"] = perim * h / a_fl
    out["geo_perimeter_fraction"] = 1.0 - (L - 2 * PERIM_DEPTH) * (W - 2 * PERIM_DEPTH) / a_fl
    out["geo_inv_floors"] = 1.0 / n
    out["geo_single_storey"] = (n == 1).astype(float)
    out["geo_compactness"] = (perim * h * n + 2.0 * a_fl) / (a_fl * h * n)

    # S - solar and orientation (fenestration._place_window_on_wall: one centred window per wall,
    # W_win = L_wall*sqrt(wwr), H_win = h*sqrt(wwr), shading created only if depth >= 0.02 m)
    shgc, theta = v["glazing_shgc"], v["orientation"]
    glazed, aperture = {}, {}
    for sfx, wwr_c, oh_c, s, _ in FACADES:
        wwr = v[wwr_c]
        glazed[sfx] = wwr * side[s] * h / a_fl
        aperture[sfx] = shgc * glazed[sfx]
    for sfx, *_ in FACADES:
        out[f"sol_glazed_ratio_{sfx}"] = glazed[sfx]
    out["sol_glazed_ratio"] = sum(glazed.values())
    for sfx, *_ in FACADES:
        out[f"sol_aperture_{sfx}"] = aperture[sfx]
    out["sol_aperture"] = sum(aperture.values())
    rad = np.deg2rad(theta)
    out["sol_orient_sin"], out["sol_orient_cos"] = np.sin(rad), np.cos(rad)
    for d, az_dir in TRUE_DIRECTIONS:                    # North Axis is clockwise: true az = theta + rel
        out[f"sol_aperture_true_{d}"] = sum(
            aperture[sfx] * np.maximum(0.0, np.cos(np.deg2rad(theta + rel - az_dir)))
            for sfx, _, _, _, rel in FACADES)
    for sfx, wwr_c, oh_c, _, _ in FACADES:
        d_oh = v[oh_c]
        out[f"sol_pf_overhang_{sfx}"] = np.where(d_oh >= MIN_SHADING_DEPTH, d_oh / (h * np.sqrt(v[wwr_c])), 0.0)
    d_fin = v["sidefin_depth"]
    for sfx, wwr_c, _, s, _ in FACADES:
        out[f"sol_pf_fin_{sfx}"] = np.where(d_fin >= MIN_SHADING_DEPTH, d_fin / (side[s] * np.sqrt(v[wwr_c])), 0.0)

    # U - heat-loss coefficients per m2 floor (W/K/m2)
    u_wall = 1.0 / (v["wall_r_value"] + R_FILMS_WALL)
    u_roof = 1.0 / (v["roof_r_value"] + R_FILMS_ROOF)
    out["ua_wall"] = u_wall * (out["geo_wall_to_floor"] - out["sol_glazed_ratio"])
    out["ua_roof"] = u_roof / n
    out["ua_glass"] = v["glazing_u"] * out["sol_glazed_ratio"]
    out["ua_ground"] = ffactor(v["floor_r_value"]) * perim / (a_fl * n)
    out["ua_infiltration"] = RHO_CP_AIR * (v["infiltration_ach"] / ACH50_TO_NATURAL) * h / 3600.0
    out["ua_total"] = out["ua_wall"] + out["ua_roof"] + out["ua_glass"] + out["ua_ground"] + out["ua_infiltration"]

    # C - climate x building
    out["cx_ua_hdd"] = (out["ua_total"] - out["ua_infiltration"]) * v["hdd18"]
    out["cx_aperture_ghi"] = out["sol_aperture"] * v["annual_ghi"]
    out["cx_aperture_cdd"] = out["sol_aperture"] * v["cdd10"]
    out["cx_infil_hdd"] = out["ua_infiltration"] * v["hdd18"]
    out["cx_infil_rh"] = out["ua_infiltration"] * v["mean_rh"]

    return pd.DataFrame({c: out[c] for c in ALL_FEATURES}, index=X.index)


def feature_set(name: str) -> list[str]:
    """Column list of one ablation set (R, G, S, U, C, C-G, C-S, C-U)."""
    if name not in FEATURE_SETS:
        raise KeyError(f"unknown feature set {name!r}; have {list(FEATURE_SETS)}")
    return list(FEATURE_SETS[name])
