"""
test_mlp.py - checks for surrogate/mlp.py (Sub-task 3.3 step 2).

Synthetic data with known answers: output shapes, reproducibility, the grouped early-stopping
split, learning a known function, early stopping and restoring the best weights, bad input.

Run:  ./venv/bin/python surrogate/tests/test_mlp.py   (or all suites: surrogate/run_tests.py)
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from surrogate.mlp import MLPRegressor, inner_split


SCALE = np.array([1, 10, 100, 0.1, 1])
W = np.random.default_rng(99).normal(size=(5, 3))


def truth(X):
    return (X / SCALE) @ W * 50 + 200


def _data(n=600, k=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5)) * SCALE                                   # very different scales
    Y = truth(X)[:, :k] + rng.normal(0, 0.5, (n, k))
    groups = np.repeat(np.arange(n // 3), 3).astype(str)                  # 3 rows per "design"
    return X, Y, groups


FAST = dict(n_layers=2, width=64, max_epochs=300, patience=20, lr=3e-3)


def check_shapes_multi_and_single():
    X, Y, g = _data()
    net = MLPRegressor(**FAST).fit(X, Y, groups=g)
    assert net.predict(X[:9]).shape == (9, 3)
    net1 = MLPRegressor(**FAST).fit(X, Y[:, 0], groups=g)
    assert net1.predict(X[:9]).shape == (9,)


def check_same_seed_identical_other_seed_differs():
    X, Y, g = _data()
    a = MLPRegressor(**FAST, seed=1).fit(X, Y, groups=g).predict(X[:20])
    b = MLPRegressor(**FAST, seed=1).fit(X, Y, groups=g).predict(X[:20])
    c = MLPRegressor(**FAST, seed=2).fit(X, Y, groups=g).predict(X[:20])
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def check_inner_split_keeps_designs_apart():
    _, _, g = _data()
    tr, va = inner_split(len(g), g, 0.1, 42)
    assert set(g[tr]).isdisjoint(set(g[va]))
    assert len(tr) + len(va) == len(g) and len(np.intersect1d(tr, va)) == 0
    assert abs(len(set(g[va])) - 0.1 * len(set(g))) <= 1
    net = MLPRegressor(**FAST).fit(*_data()[:2], groups=g)
    assert set(g[net.inner_train_idx_]).isdisjoint(set(g[net.inner_val_idx_]))
    try:
        inner_split(5, ["a"] * 5, 0.1, 0)
    except ValueError:
        return
    raise AssertionError("one group accepted")


def check_learns_a_known_function():
    X, Y, g = _data(n=900)
    Xt, Yt, _ = _data(n=300, seed=1)
    net = MLPRegressor(n_layers=2, width=64, max_epochs=600, patience=40, lr=3e-3).fit(X, Y, groups=g)
    P, T = net.predict(Xt), truth(Xt)
    r2 = 1 - ((P - T) ** 2).sum(0) / ((T - T.mean(0)) ** 2).sum(0)
    assert (r2 > 0.98).all(), r2


def check_early_stopping_restores_best_weights():
    X, Y, g = _data()
    net = MLPRegressor(n_layers=4, width=256, max_epochs=400, patience=10, lr=3e-3).fit(X, Y, groups=g)
    assert net.n_epochs_ < 400, "never stopped early"
    assert net.n_epochs_ - net.best_epoch_ == 10
    assert abs(min(net.val_history_) - net.best_val_loss_) < 1e-12
    # the restored weights reproduce the best validation loss
    va = net.inner_val_idx_
    Ys = (Y[va] - net.y_mean_) / net.y_sd_
    Ps = (net.predict(X[va]) - net.y_mean_) / net.y_sd_
    assert abs(float(np.mean((Ps - Ys) ** 2)) - net.best_val_loss_) < 1e-4


def check_dataframe_input_and_bad_values():
    X, Y, g = _data()
    Xd = pd.DataFrame(X, columns=list("abcde"))
    net = MLPRegressor(**FAST).fit(Xd, Y, groups=g)
    assert net.feature_names_in_ == list("abcde")
    assert np.array_equal(net.predict(Xd.iloc[:5]), net.predict(X[:5]))
    Xbad = X.copy()
    Xbad[0, 0] = np.nan
    for bad in (lambda: MLPRegressor(**FAST).fit(Xbad, Y, groups=g),
                lambda: MLPRegressor(device="mps"),
                lambda: MLPRegressor(dropout=1.0)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("bad input accepted")


def check_openmp_safe_order_and_single_thread():
    """macOS OpenMP rules (mlp._torch): in a FRESH process, training the MLP must load LightGBM and
    XGBoost before torch, use one thread, and still allow boosting models afterwards (the order
    that crashed with OMP Error #179 / deadlocked on 5 Nov 2026)."""
    import subprocess
    code = (
        "import sys\n"
        "from surrogate.mlp import MLPRegressor\n"
        "import numpy as np\n"
        "assert 'torch' not in sys.modules\n"
        "rng = np.random.default_rng(0); X = rng.normal(size=(120, 4)); Y = X @ rng.normal(size=(4, 2))\n"
        "g = np.repeat(np.arange(40), 3).astype(str)\n"
        "MLPRegressor(n_layers=2, width=16, max_epochs=5).fit(X, Y, groups=g)\n"
        "mods = list(sys.modules)\n"
        "assert mods.index('lightgbm') < mods.index('torch') and mods.index('xgboost') < mods.index('torch'), 'order'\n"
        "import torch, lightgbm, xgboost\n"
        "assert torch.get_num_threads() == 1\n"
        "lightgbm.LGBMRegressor(n_estimators=10, verbose=-1).fit(X, Y[:, 0])\n"
        "xgboost.XGBRegressor(n_estimators=10).fit(X, Y[:, 0])\n"
        "MLPRegressor(n_layers=2, width=16, max_epochs=5).fit(X, Y, groups=g)\n"
        "print('OK')\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "OK" in r.stdout, (r.stdout + r.stderr)[-600:]


ALL_CHECKS = [check_shapes_multi_and_single, check_same_seed_identical_other_seed_differs,
              check_inner_split_keeps_designs_apart, check_learns_a_known_function,
              check_early_stopping_restores_best_weights, check_dataframe_input_and_bad_values,
              check_openmp_safe_order_and_single_thread]


def main():
    total = failed = 0
    print("=" * 72)
    print("SURROGATE MLP (3.3) - CHECKS")
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
