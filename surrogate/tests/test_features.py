"""
test_features.py - checks for surrogate/features.py (Sub-task 3.2 step 3).

The key checks rebuild real designs with the Phase 2 generator's own spec modules
(geometry, fenestration, constructions) and require every engineered quantity to equal what
EnergyPlus was actually given: window areas, window sizes, wall/roof/ground areas, exposed slab
perimeter, F-factor, layer R-values, the ACH50 divisor. Then: set definitions and counts,
orientation by hand, input handling, no NaN/inf at the design-space corners, and that the
feature -> raw-input map (used for SHAP in 3.6) is complete and exact.
Only the training split is read.

Run:  ./venv/bin/python surrogate/tests/test_features.py   (or all suites: surrogate/run_tests.py)
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np
import pandas as pd
import yaml

from idf_generator.construction.constructions import build_constructions, ffactor_for_floor_r
from idf_generator.fenestration.fenestration import MIN_SHADING_DEPTH, build_fenestration
from idf_generator.geometry.geometry import build_geometry
from surrogate import cv
from surrogate import features as FE
from surrogate.paths import DATASET_DIR, DESIGN_SPACE

N_DESIGNS = 20
TOL = 1e-9
_CACHE: dict = {}

DIR_SFX = {"North": "n", "East": "e", "South": "s", "West": "w"}


def _train():
    if "df" not in _CACHE:
        _CACHE["df"], _CACHE["cols"] = cv.load_training()
    return _CACHE["df"], _CACHE["cols"]


def _sample(n=N_DESIGNS, seed=3):
    """n distinct designs (one row each) from the training split, shading devices included."""
    df, _ = _train()
    one = df.drop_duplicates("design_id")
    return one.sample(n=n, random_state=seed).reset_index(drop=True)


def _build(row):
    geom = build_geometry(float(row.floor_area), float(row.aspect_ratio), int(row.n_floors),
                          float(row.floor_height))
    fen = build_fenestration(
        geom, row.wwr_north, row.wwr_east, row.wwr_south, row.wwr_west,
        row.glazing_u, row.glazing_shgc, row.glazing_shgc,            # VT = SHGC (decision S2)
        row.overhang_depth_n, row.overhang_depth_e, row.overhang_depth_s, row.overhang_depth_w,
        row.sidefin_depth)
    cons = build_constructions(row.wall_r_value, row.roof_r_value, row.floor_r_value, row.wall_thermal_mass)
    return geom, fen, cons


def _close(a, b, what, rel=TOL):
    assert math.isclose(float(a), float(b), rel_tol=rel, abs_tol=1e-12), f"{what}: {a} != {b}"


def _bounds():
    spec = yaml.safe_load(DESIGN_SPACE.read_text(encoding="utf-8"))["continuous_parameters"]
    return {k: (float(v["min"]), float(v["max"])) for k, v in spec.items()}


# ── definitions ─────────────────────────────────────────────────────────────────────────────

def check_raw_inputs_match_manifest():
    man = json.loads((DATASET_DIR / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert FE.RAW_INPUTS == man["columns"]["inputs"], "RAW_INPUTS order differs from the manifest"
    assert FE.DESIGN_INPUTS == man["columns"]["design_inputs"]
    assert FE.CLIMATE_INPUTS == man["columns"]["climate_inputs"]
    assert set(FE.DESIGN_INPUTS) == set(_bounds()), "design inputs differ from design_space.yaml"


def check_group_and_set_sizes():
    assert [len(FE.GROUPS[g]) for g in "GSUC"] == [8, 24, 6, 5]
    assert len(FE.ENGINEERED) == 43 and len(FE.ALL_FEATURES) == 71
    assert len(set(FE.ALL_FEATURES)) == 71, "duplicate feature names"
    sizes = {k: len(v) for k, v in FE.FEATURE_SETS.items()}
    assert sizes == {"R": 28, "G": 36, "S": 60, "U": 66, "C": 71, "C-G": 63, "C-S": 47, "C-U": 65}, sizes


def check_ladder_and_leave_one_out_structure():
    sets = {k: set(v) for k, v in FE.FEATURE_SETS.items()}
    for a, b in zip(FE.LADDER, FE.LADDER[1:]):
        assert sets[a] < sets[b], f"{a} is not a strict subset of {b}"
    for g in "GSU":
        assert sets[f"C-{g}"] == sets["C"] - set(FE.GROUPS[g]), f"C-{g} wrong"
    assert sets["C"] - set(FE.GROUPS["C"]) == sets["U"], "C minus climate group must equal U"
    for k, cols in FE.FEATURE_SETS.items():
        assert cols == [c for c in FE.ALL_FEATURES if c in set(cols)], f"set {k} not in canonical order"
    assert FE.feature_set("S") == FE.FEATURE_SETS["S"]


def check_constants_match_generator():
    import inspect
    from idf_generator.generator.generator import ACH50_TO_NATURAL_DIVISOR      # imports eppy
    assert FE.ACH50_TO_NATURAL == ACH50_TO_NATURAL_DIVISOR
    assert FE.PERIM_DEPTH == inspect.signature(build_geometry).parameters["perim_depth"].default
    assert MIN_SHADING_DEPTH == 0.02


# ── against the generator (20 real designs) ─────────────────────────────────────────────────

def check_geometry_matches_generator():
    S = _sample()
    F = FE.engineer(S)
    for i, row in S.iterrows():
        geom, _, _ = _build(row)
        f = F.loc[i]
        _close(f.geo_length, geom.length, "length")
        _close(f.geo_width, geom.width, "width")
        _close(f.geo_total_floor_area, geom.total_floor_area, "total floor area", rel=1e-9)
        walls = [s for z in geom.zones for s in z.exterior_walls]
        _close(f.geo_wall_to_floor, sum(w.area for w in walls) / geom.total_floor_area, "wall/floor")
        perim_area = sum(z.floor_area for z in geom.zones if z.role != "Core")
        _close(f.geo_perimeter_fraction, perim_area / geom.total_floor_area, "perimeter fraction")
        roofs = [s for z in geom.zones for s in z.surfaces if s.surface_type == "Roof"]
        grounds = [s for z in geom.zones for s in z.surfaces if s.surface_type == "Floor"
                   and s.boundary_condition == "Ground"]
        _close(f.geo_inv_floors, sum(r.area for r in roofs) / geom.total_floor_area, "roof/floor")
        volume = sum(z.volume for z in geom.zones)
        env = sum(w.area for w in walls) + sum(r.area for r in roofs) + sum(g.area for g in grounds)
        _close(f.geo_compactness, env / volume, "compactness")
        _close(f.geo_single_storey, float(geom.n_floors == 1), "single storey")


def check_windows_and_shading_match_generator():
    S = _sample()
    F = FE.engineer(S)
    n_shaded = 0
    for i, row in S.iterrows():
        geom, fen, _ = _build(row)
        f = F.loc[i]
        wall_dir = {s.name: s.direction for z in geom.zones for s in z.exterior_walls}
        wall_len = {s.name: math.dist(s.vertices[1], s.vertices[2]) for z in geom.zones for s in z.exterior_walls}
        area = {k: 0.0 for k in DIR_SFX.values()}
        for win in fen.windows:
            sfx = DIR_SFX[wall_dir[win.parent_wall_name]]
            area[sfx] += win.area
            # window size -> projection factors
            UL, LL, LR, _ = win.vertices
            h_win, w_win = math.dist(UL, LL), math.dist(LL, LR)
            assert w_win < wall_len[win.parent_wall_name] + 1e-9
            oh = getattr(row, f"overhang_depth_{sfx}")
            exp_oh = oh / h_win if oh >= MIN_SHADING_DEPTH else 0.0
            exp_fin = row.sidefin_depth / w_win if row.sidefin_depth >= MIN_SHADING_DEPTH else 0.0
            _close(f[f"sol_pf_overhang_{sfx}"], exp_oh, f"overhang PF {sfx}")
            _close(f[f"sol_pf_fin_{sfx}"], exp_fin, f"fin PF {sfx}")
        for sfx, a in area.items():
            _close(f[f"sol_glazed_ratio_{sfx}"], a / geom.total_floor_area, f"glazed ratio {sfx}")
            _close(f[f"sol_aperture_{sfx}"], row.glazing_shgc * a / geom.total_floor_area, f"aperture {sfx}")
        _close(f.sol_glazed_ratio, sum(area.values()) / geom.total_floor_area, "glazed ratio total")
        # a shading device exists in the IDF exactly when its projection factor is > 0
        kinds = {(DIR_SFX[wall_dir[s.base_surface_name]], s.kind) for s in fen.shading}
        for sfx in DIR_SFX.values():
            assert ((sfx, "Overhang") in kinds) == (f[f"sol_pf_overhang_{sfx}"] > 0), f"overhang {sfx}"
            assert ((sfx, "SideFinLeft") in kinds) == (f[f"sol_pf_fin_{sfx}"] > 0), f"fin {sfx}"
        n_shaded += len(fen.shading)
    assert n_shaded > 0, "sample contains no shading devices - test is vacuous"


def check_heat_loss_terms_match_generator():
    S = _sample()
    F = FE.engineer(S)
    for i, row in S.iterrows():
        geom, fen, cons = _build(row)
        f = F.loc[i]
        A = geom.total_floor_area
        # layer R written to EnergyPlus equals the sampled R (films are added by features, F1)
        _close(cons.total_r_value(cons.exterior_wall), row.wall_r_value, "wall layer R", rel=1e-9)
        _close(cons.total_r_value(cons.roof), row.roof_r_value, "roof layer R", rel=1e-9)
        walls = [s for z in geom.zones for s in z.exterior_walls]
        opaque = sum(w.area for w in walls) - sum(win.area for win in fen.windows)
        _close(f.ua_wall, opaque / (row.wall_r_value + FE.R_FILMS_WALL) / A, "ua_wall")
        roofs = sum(s.area for z in geom.zones for s in z.surfaces if s.surface_type == "Roof")
        _close(f.ua_roof, roofs / (row.roof_r_value + FE.R_FILMS_ROOF) / A, "ua_roof")
        _close(f.ua_glass, row.glazing_u * sum(w.area for w in fen.windows) / A, "ua_glass")
        exposed = sum(z.exposed_perimeter for z in geom.zones if z.floor_index == 0)
        _close(f.ua_ground, cons.ground_floor_ffactor * exposed / A, "ua_ground")
        vol = sum(z.volume for z in geom.zones)
        exp_inf = FE.RHO_CP_AIR * row.infiltration_ach / FE.ACH50_TO_NATURAL * vol / 3600.0 / A
        _close(f.ua_infiltration, exp_inf, "ua_infiltration")
        _close(f.ua_total, f.ua_wall + f.ua_roof + f.ua_glass + f.ua_ground + f.ua_infiltration, "ua_total")


def check_ffactor_vectorised_equals_generator():
    r = np.concatenate([np.linspace(0.01, 6.0, 997), [0.5, 0.880560, 1.320840, 1.761120, 2.641680, 3.522240, 5.0]])
    exp = np.array([ffactor_for_floor_r(x) for x in r])
    got = FE.ffactor(r)
    assert np.allclose(got, exp, rtol=1e-12, atol=0), f"max diff {np.max(np.abs(got - exp))}"


def check_climate_cross_terms():
    df, _ = _train()
    F = FE.engineer(df)
    assert np.allclose(F.cx_ua_hdd, (F.ua_total - F.ua_infiltration) * df.hdd18)
    assert np.allclose(F.cx_aperture_ghi, F.sol_aperture * df.annual_ghi)
    assert np.allclose(F.cx_aperture_cdd, F.sol_aperture * df.cdd10)
    assert np.allclose(F.cx_infil_hdd, F.ua_infiltration * df.hdd18)
    assert np.allclose(F.cx_infil_rh, F.ua_infiltration * df.mean_rh)
    # zero-HDD climates (Chennai, Colombo) have zero heating cross-terms
    hot = df.hdd18.to_numpy() == 0
    assert hot.any() and np.all(F.cx_ua_hdd.to_numpy()[hot] == 0)


# ── orientation by hand ─────────────────────────────────────────────────────────────────────

def _one(**kw):
    base = {c: float(np.mean(b)) for c, b in _bounds().items()}
    base.update(n_floors=2.0, hdd18=1000.0, cdd10=3000.0, annual_ghi=1700.0, mean_dbt=20.0,
                mean_rh=65.0, latitude=30.0, mean_wind=3.0)
    base.update(kw)
    return base


def check_true_direction_aperture_by_hand():
    d = _one(wwr_north=0.2, wwr_east=0.3, wwr_south=0.5, wwr_west=0.4)
    for theta in (0.0, 90.0, 45.0, 180.0, 123.4):
        f = FE.engineer({**d, "orientation": theta}).iloc[0]
        ap = {s: f[f"sol_aperture_{s}"] for s in "nesw"}
        if theta == 0.0:
            exp = {"N": ap["n"], "E": ap["e"], "S": ap["s"], "W": ap["w"]}
        elif theta == 90.0:                        # building "north" faces true east
            exp = {"N": ap["w"], "E": ap["n"], "S": ap["e"], "W": ap["s"]}
        elif theta == 180.0:
            exp = {"N": ap["s"], "E": ap["w"], "S": ap["n"], "W": ap["e"]}
        elif theta == 45.0:                        # north facade at 45 deg: half-way N/E
            c = math.cos(math.radians(45))
            exp = {"N": c * (ap["n"] + ap["w"]), "E": c * (ap["n"] + ap["e"]),
                   "S": c * (ap["e"] + ap["s"]), "W": c * (ap["s"] + ap["w"])}
        else:
            exp = None
        if exp:
            for k, val in exp.items():
                assert abs(f[f"sol_aperture_true_{k}"] - val) < 1e-12, f"theta={theta} true_{k}"
        # each facade contributes cos-weighted to at most two directions; weights sum to cos+sin >= 1
        tot = sum(f[f"sol_aperture_true_{k}"] for k in "NESW")
        assert f.sol_aperture - 1e-12 <= tot <= math.sqrt(2) * f.sol_aperture + 1e-12
    assert abs(FE.engineer({**d, "orientation": 30.0}).iloc[0].sol_orient_sin - 0.5) < 1e-12


# ── input handling ──────────────────────────────────────────────────────────────────────────

def check_dict_list_and_frame_agree():
    S = _sample(5)
    F = FE.engineer(S)
    rows = [{c: float(r[c]) for c in FE.RAW_INPUTS} for _, r in S.iterrows()]
    pd.testing.assert_frame_equal(FE.engineer(rows), F.reset_index(drop=True))
    pd.testing.assert_frame_equal(FE.engineer(rows[2]), F.iloc[[2]].reset_index(drop=True))


def check_only_raw_inputs_are_read_and_index_kept():
    df, cols = _train()
    S = df.sample(50, random_state=1)
    F = FE.engineer(S)
    assert list(F.columns) == FE.ALL_FEATURES and F.index.equals(S.index)
    junk = S.copy()
    for t in cols["targets"]:
        junk[t] = np.nan                          # targets unreadable
    junk["extra"] = "x"
    junk = junk[list(reversed(junk.columns))]     # column order irrelevant
    pd.testing.assert_frame_equal(FE.engineer(junk), F)
    pd.testing.assert_frame_equal(F[FE.RAW_INPUTS], S[FE.RAW_INPUTS].astype(float))
    pd.testing.assert_frame_equal(FE.engineer(S), F)            # deterministic


def check_invalid_inputs_raise():
    d = _one(orientation=10.0)
    cases = [({"n_floors": 2.5}, ValueError), ({"wwr_east": 0.0}, ValueError), ({"floor_area": np.nan}, ValueError),
             ({"aspect_ratio": 0.8}, ValueError), ({"glazing_shgc": 1.2}, ValueError),
             ({"floor_area": 200.0, "aspect_ratio": 3.0}, ValueError)]      # W = 8.2 m < 2 x 4.57 m: no core
    for change, exc in cases:
        try:
            FE.engineer({**d, **change})
        except exc:
            continue
        raise AssertionError(f"{change} accepted")
    try:
        FE.engineer({k: v for k, v in d.items() if k != "mean_rh"})
    except KeyError:
        pass
    else:
        raise AssertionError("missing column accepted")
    try:
        FE.feature_set("X")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown set accepted")


def check_finite_on_training_set_and_design_space_corners():
    df, _ = _train()
    assert np.isfinite(FE.engineer(df).to_numpy()).all(), "NaN/inf on the training set"
    b = _bounds()
    rng = np.random.default_rng(0)
    clim = df[FE.CLIMATE_INPUTS].drop_duplicates().to_numpy()
    n = 4000
    corners = {k: np.where(rng.random(n) < 0.5, lo, hi) for k, (lo, hi) in b.items()}
    C = pd.DataFrame(corners)
    C[FE.CLIMATE_INPUTS] = clim[rng.integers(0, len(clim), n)]
    F = FE.engineer(C)
    assert np.isfinite(F.to_numpy()).all(), "NaN/inf at a design-space corner"
    assert (F[FE.HEAT_LOSS].to_numpy() >= 0).all() and (F[["sol_glazed_ratio", "sol_aperture"]].to_numpy() > 0).all()
    assert (F.ua_wall > 0).all(), "glazing exceeded the wall area"


def check_raw_parents_complete_and_exact():
    """Each feature changes when (and only when) one of its listed raw parents changes."""
    assert set(FE.RAW_PARENTS) == set(FE.ALL_FEATURES)
    base = _sample(30, seed=11)
    # full circle: with the sampled 0-180 deg range some facades can never face some true directions
    # (e.g. the east facade never faces true north), but the map must hold for any angle.
    base["orientation"] = np.random.default_rng(2).uniform(0, 360, len(base))
    F0 = FE.engineer(base)
    b = _bounds()
    rng = np.random.default_rng(5)
    moved: dict[str, set] = {c: set() for c in FE.ALL_FEATURES}
    for raw in FE.RAW_INPUTS:
        P = base.copy()
        if raw in b:
            lo, hi = b[raw]
            new = rng.uniform(lo, hi, len(P))
            if raw == "n_floors":
                new = np.where(P[raw] == 1, 3.0, 1.0)
            if raw == "orientation":
                new = rng.uniform(0, 360, len(P))
            P[raw] = new
        else:
            P[raw] = P[raw] * 1.37 + 11.0
        F1 = FE.engineer(P)
        for c in FE.ALL_FEATURES:
            if not np.allclose(F1[c].to_numpy(), F0[c].to_numpy(), rtol=1e-12, atol=0):
                moved[c].add(raw)
    for c in FE.ALL_FEATURES:
        assert moved[c] <= set(FE.RAW_PARENTS[c]), f"{c} depends on unlisted {moved[c] - set(FE.RAW_PARENTS[c])}"
        assert moved[c] == set(FE.RAW_PARENTS[c]), f"{c}: listed parents never move it: {set(FE.RAW_PARENTS[c]) - moved[c]}"


def check_physics_directions():
    d = _one(orientation=20.0)
    lo, hi = FE.engineer({**d, "glazing_u": 1.0}).iloc[0], FE.engineer({**d, "glazing_u": 5.0}).iloc[0]
    assert hi.ua_glass > lo.ua_glass and hi.ua_total > lo.ua_total
    lo, hi = FE.engineer({**d, "glazing_shgc": 0.2}).iloc[0], FE.engineer({**d, "glazing_shgc": 0.7}).iloc[0]
    assert hi.sol_aperture > lo.sol_aperture and hi.ua_glass == lo.ua_glass
    lo, hi = FE.engineer({**d, "wall_r_value": 1.0}).iloc[0], FE.engineer({**d, "wall_r_value": 8.0}).iloc[0]
    assert hi.ua_wall < lo.ua_wall
    lo, hi = FE.engineer({**d, "wwr_south": 0.2}).iloc[0], FE.engineer({**d, "wwr_south": 0.7}).iloc[0]
    assert hi.ua_wall < lo.ua_wall and hi.ua_glass > lo.ua_glass and hi.sol_pf_overhang_s < lo.sol_pf_overhang_s
    one, five = FE.engineer({**d, "n_floors": 1.0}).iloc[0], FE.engineer({**d, "n_floors": 5.0}).iloc[0]
    assert five.ua_roof == one.ua_roof / 5 and five.ua_ground < one.ua_ground and five.geo_single_storey == 0


ALL_CHECKS = [check_raw_inputs_match_manifest, check_group_and_set_sizes, check_ladder_and_leave_one_out_structure,
              check_constants_match_generator, check_geometry_matches_generator,
              check_windows_and_shading_match_generator, check_heat_loss_terms_match_generator,
              check_ffactor_vectorised_equals_generator, check_climate_cross_terms,
              check_true_direction_aperture_by_hand, check_dict_list_and_frame_agree,
              check_only_raw_inputs_are_read_and_index_kept, check_invalid_inputs_raise,
              check_finite_on_training_set_and_design_space_corners, check_raw_parents_complete_and_exact,
              check_physics_directions]


def main():
    total = failed = 0
    print("=" * 72)
    print("SURROGATE FEATURES (3.2) - CHECKS")
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
