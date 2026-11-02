"""
check_ml.py - Sub-task 3.1 step 4 check: Phase 3 ML libraries installed and working.

    ./venv/bin/python surrogate/check_ml.py           # check only
    ./venv/bin/python surrogate/check_ml.py --pin     # check, then write surrogate/requirements.txt

Each library is imported AND used on a tiny problem, so a missing system
library (for example OpenMP for LightGBM/XGBoost on macOS) shows up here,
not in the middle of a tuning run. Phase 2 versions must be unchanged.
"""

from __future__ import annotations

import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

SURROGATE = Path(__file__).resolve().parent
PHASE2_PINS = {"numpy": "2.5.3", "pandas": "3.0.6", "scikit-learn": "1.9.1", "scipy": "1.18.1", "shap": "0.52.0"}
PHASE3_LIBS = ["lightgbm", "xgboost", "optuna", "torch"]
REQUIREMENTS = ["numpy", "pandas", "scipy", "scikit-learn", "shap", "pyyaml", "matplotlib", *PHASE3_LIBS]

rng = np.random.default_rng(42)
X = rng.random((500, 8))
y = 3 * X[:, 0] + X[:, 1] ** 2 + 0.1 * rng.standard_normal(500)
results: list[tuple[str, bool, str]] = []


def check(name, fn):
    t0 = time.perf_counter()
    try:
        detail = fn()
        results.append((name, True, f"{detail}  ({time.perf_counter() - t0:.1f} s)"))
    except Exception as exc:  # noqa: BLE001
        results.append((name, False, f"{type(exc).__name__}: {exc}"))


def lightgbm_fit():
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=50, verbose=-1, n_jobs=-1).fit(X, y)
    r2 = 1 - np.mean((m.predict(X) - y) ** 2) / np.var(y)
    assert r2 > 0.9, f"train R2 {r2:.3f}"
    return f"{lgb.__version__}, multi-core fit OK"


def xgboost_fit():
    import xgboost as xgb
    m = xgb.XGBRegressor(n_estimators=50, n_jobs=-1).fit(X, y)
    r2 = 1 - np.mean((m.predict(X) - y) ** 2) / np.var(y)
    assert r2 > 0.9, f"train R2 {r2:.3f}"
    return f"{xgb.__version__}, multi-core fit OK"


def optuna_study():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    s = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))
    s.optimize(lambda t: (t.suggest_float("x", -5, 5) - 2) ** 2, n_trials=30)
    assert abs(s.best_params["x"] - 2) < 1.0
    return f"{optuna.__version__}, TPE search OK"


def torch_fit():
    import torch
    torch.manual_seed(42)
    net = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)[:, None]
    for _ in range(300):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(xt), yt)
        loss.backward()
        opt.step()
    assert loss.item() < 0.1 * float(np.var(y)), f"loss {loss.item():.3f}"
    return f"{torch.__version__}, CPU training OK, Apple GPU (MPS) available: {torch.backends.mps.is_available()}"


def phase2_unchanged():
    diff = {k: (version(k), v) for k, v in PHASE2_PINS.items() if version(k) != v}
    assert not diff, f"Phase 2 versions changed: {diff}"
    return ", ".join(f"{k} {v}" for k, v in PHASE2_PINS.items())


for name, fn in [("LightGBM", lightgbm_fit), ("XGBoost", xgboost_fit), ("Optuna", optuna_study),
                 ("PyTorch", torch_fit), ("Phase 2 versions unchanged", phase2_unchanged)]:
    check(name, fn)

width = max(len(n) for n, _, _ in results)
for name, ok, detail in results:
    print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
n_fail = sum(not ok for _, ok, _ in results)
print(f"\n{len(results) - n_fail}/{len(results)} checks passed")

if n_fail == 0 and "--pin" in sys.argv:
    lines = ["# surrogate/requirements.txt - Phase 3 direct dependencies, exact versions (Sub-task 3.1 step 4).",
             f"# Recorded {time.strftime('%Y-%m-%d')} on {platform.system()} {platform.machine()}, "
             f"Python {platform.python_version()}.",
             "# Full environment: pip install -r simulation/requirements.txt -c surrogate/phase2_lock.txt, then this file.",
             *[f"{pkg}=={version(pkg)}" for pkg in REQUIREMENTS]]
    (SURROGATE / "requirements.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote surrogate/requirements.txt")
sys.exit(1 if n_fail else 0)
