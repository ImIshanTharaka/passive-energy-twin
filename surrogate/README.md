# surrogate/ — Phase 3: ML surrogate models

Fast ML models that imitate EnergyPlus for the 10 targets of the frozen Phase 2 dataset.
Plan: `claude/Phase3_Execution_Plan.md` (Rev 2). Reads Phase 2 artefacts read-only through
`paths.py`; all outputs go to `surrogate/outputs/`.

Status: Sub-task 3.1 step 5 (evaluation harness).

| File | Purpose |
|---|---|
| `paths.py` | the only place that knows where Phase 2 artefacts live |
| `check_setup.py` | verifies the shared venv, path files and Phase 2 imports (3.1 step 1) |
| `s0_check.py` | gate S0: the 2.8 RF re-run through the new harness must match 2.8 (3.1 step 6) |
| `phase2_lock.txt` | exact package versions of `simulation/venv`, used as pip constraints |
| `install_ml.sh` | installs LightGBM, XGBoost, Optuna, PyTorch into the shared venv (3.1 step 4) |
| `check_ml.py` | tests each ML library on a tiny problem; `--pin` writes `requirements.txt` |
| `requirements.txt` | exact Phase 3 library versions (written by `check_ml.py --pin`) |
| `config.yaml` / `config.py` | the frozen Phase 3 protocol, selection rule and gate (signed off 30 Oct 2026) |
| `cv.py` | grouped CV (2.8 folds) and leave-one-climate-out, model-agnostic, holdout-guarded |
| `metrics.py` | 2.8 metrics + ranking (Spearman, top-10 % hit rate), LOCO bias/MAPE, M7 trigger |
| `tests/` | test suites (`tests/test_cv.py`: 18 checks for the harness); `run_tests.py` runs them all |
