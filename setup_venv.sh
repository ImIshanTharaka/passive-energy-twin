#!/bin/bash
# setup_venv.sh - Sub-task 3.1 step 1: create the shared Phase 3+ venv at Project/venv (plan P-L4).
# Run on the Mac:  cd ~/Documents/Msc/Research/Project && bash setup_venv.sh
# Safe to re-run. Changes nothing inside simulation/ (simulation/venv keeps working).
set -euo pipefail
cd "$(dirname "$0")"
PROJECT="$(pwd)"
PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13   # same interpreter as simulation/venv

if [ ! -x "$PY" ]; then echo "Python 3.13 not found at $PY"; exit 1; fi

if [ -d venv ]; then
  echo "[1/4] venv/ already exists - reusing it"
else
  echo "[1/4] Creating venv/ with $("$PY" --version)"
  "$PY" -m venv venv
fi

echo "[2/4] Installing Phase 2 requirements at the exact Phase 2 versions"
./venv/bin/python -m pip install --quiet -r simulation/requirements.txt -c surrogate/phase2_lock.txt

echo "[3/4] Adding path files (simulation.pth, project.pth)"
SITE="$(./venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$PROJECT/simulation" > "$SITE/simulation.pth"
echo "$PROJECT"            > "$SITE/project.pth"

echo "[4/4] Checking the setup"
./venv/bin/python surrogate/check_setup.py
