"""
test_ablation.py - checks for surrogate/ablation.py (Sub-task 3.2 step 4).

The pre-registered selection rule (D3/D4) on hand-made tables with known answers, the
config settings, and a small real run (one set, LightGBM, grouped CV) for output shape.
Only the training split is read.

Run:  ./venv/bin/python surrogate/tests/test_ablation.py   (or all suites: surrogate/run_tests.py)
"""

from __future__ import annotations

import sys

import lightgbm
import numpy as np
import pandas as pd

from surrogate import ablation as A
from surrogate import features as FE
from surrogate.config import CONFIG, PRIMARY_TARGETS


def _table(skills: dict[str, float], spread: float = 0.0) -> pd.DataFrame:
    """One row per set x model x target; mean skill per set as given (spread around it)."""
    rows = []
    for fs, s in skills.items():
        offs = np.linspace(-spread, spread, 6)
        k = 0
        for m in ("rf", "lgbm"):
            for t in PRIMARY_TARGETS:
                rows.append({"feature_set": fs, "model": m, "target": t,
                             "n_features": len(FE.FEATURE_SETS[fs]), "grouped_design_skill": s + offs[k]})
                k += 1
    return pd.DataFrame(rows)


def check_best_score_wins_when_clear():
    scores, w = A.choose_winner(_table({"R": 0.80, "S": 0.90, "C": 0.95}, spread=0.02))
    assert w == "C" and bool(scores.loc[scores.feature_set == "C", "winner"].iloc[0])
    assert abs(float(scores.loc[scores.feature_set == "C", "score"].iloc[0]) - 0.95) < 1e-12


def check_tie_goes_to_fewest_features():
    # best 0.950 (C, 71); U 0.946 (66) and C-S 0.9451 (47) are within 0.005 -> C-S
    _, w = A.choose_winner(_table({"S": 0.930, "U": 0.946, "C": 0.950, "C-S": 0.9451}))
    assert w == "C-S", w
    # exactly at the boundary counts as tied
    _, w = A.choose_winner(_table({"C": 0.950, "U": 0.945}))
    assert w == "U", w
    # just outside the boundary does not
    _, w = A.choose_winner(_table({"C": 0.950, "U": 0.9449}))
    assert w == "C", w


def check_score_is_mean_over_targets_and_models():
    t = _table({"G": 0.9, "S": 0.9})
    t.loc[(t.feature_set == "S") & (t.model == "rf") & (t.target == PRIMARY_TARGETS[0]), "grouped_design_skill"] = 0.96
    scores, w = A.choose_winner(t, tie=0.0)
    assert w == "S" and abs(float(scores.loc[scores.feature_set == "S", "score"].iloc[0]) - 0.91) < 1e-12


def check_config_settings():
    c = CONFIG["ablation_3_2"]
    assert c["feature_sets"] == list(FE.FEATURE_SETS) and c["models"] == ["rf", "lgbm"]
    assert c["selection"]["tie_skill"] == 0.005 and c["selection"]["tie_break"] == "fewest_features"
    assert A.model_params("rf") == CONFIG["reference_2_8"]["rf_params"]
    defaults = lightgbm.LGBMRegressor().get_params()
    for k in ("n_estimators", "learning_rate", "num_leaves", "max_depth", "min_child_samples",
              "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"):
        assert A.model_params("lgbm")[k] == defaults[k], f"lgbm {k} is not the library default"
    assert A.model_params("rf", dev=True)["n_estimators"] == 30


def check_small_real_run_shape():
    _, heat, table, pc = A.run(["R"], ["lgbm"], ["grouped"], log=lambda s: None)
    assert len(heat) == 7
    assert list(table.target) == PRIMARY_TARGETS and (table.n_features == 28).all()
    assert table["grouped_design_skill"].between(0.5, 1.0).all()
    assert set(pc.scheme) == {"grouped"}
    # deterministic: the same run gives the same numbers
    _, _, again, _ = A.run(["R"], ["lgbm"], ["grouped"], log=lambda s: None)
    assert np.allclose(table["grouped_design_skill"], again["grouped_design_skill"], rtol=0, atol=0)


ALL_CHECKS = [check_best_score_wins_when_clear, check_tie_goes_to_fewest_features,
              check_score_is_mean_over_targets_and_models, check_config_settings, check_small_real_run_shape]


def main():
    total = failed = 0
    print("=" * 72)
    print("SURROGATE ABLATION (3.2) - CHECKS")
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
