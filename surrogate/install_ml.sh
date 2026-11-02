#!/bin/bash
# install_ml.sh - Sub-task 3.1 step 4: add the Phase 3 ML libraries to Project/venv.
# Run on the Mac:  cd ~/Documents/Msc/Research/Project && bash surrogate/install_ml.sh
# The Phase 2 lock is used as a constraint, so no Phase 2 library can be upgraded by accident.
# LightGBM/XGBoost versions match the scoping probe (plan §2.3); Optuna/PyTorch take the latest release.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/3] Installing lightgbm, xgboost, optuna, torch (CPU)"
./venv/bin/python -m pip install --quiet -c surrogate/phase2_lock.txt \
  "lightgbm==4.7.0" "xgboost==3.4.1" optuna torch

echo "[2/3] Checking each library on a tiny problem, then pinning versions"
./venv/bin/python surrogate/check_ml.py --pin

echo "[3/3] Re-running the step-1 setup check"
./venv/bin/python surrogate/check_setup.py
