# surrogate/ — Phase 3: ML surrogate models

Fast ML models that imitate EnergyPlus for the 10 targets of the frozen Phase 2 dataset.
Plan: `claude/Phase3_Execution_Plan.md` (Rev 2). Reads Phase 2 artefacts read-only through
`paths.py`; all outputs go to `surrogate/outputs/`.

Status: Sub-task 3.1 step 1 (layout and shared venv).

| File | Purpose |
|---|---|
| `paths.py` | the only place that knows where Phase 2 artefacts live |
| `check_setup.py` | verifies the shared venv, path files and Phase 2 imports (3.1 step 1) |
| `phase2_lock.txt` | exact package versions of `simulation/venv`, used as pip constraints |
