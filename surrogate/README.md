# surrogate/ — Phase 3: ML surrogate models

Fast ML models that imitate EnergyPlus for the 10 targets of the frozen Phase 2 dataset
(annual heating and cooling demand, EUI, peak heating and cooling, cooling of the five zones).
Plan: `claude/Phase3_Execution_Plan.md` (Rev 2); status of record: `claude/Phase3_Status_Handoff.md`.
Reads Phase 2 artefacts read-only through `paths.py`; all outputs go to `surrogate/outputs/`.

Status: Sub-tasks 3.1–3.7 complete; 3.8 (packaging) in progress — the deployed surrogate is
`registry.py` (plan `claude/Phase3.8_Execution_Plan.md`, decisions J1–J7).

## How to use the surrogate

Phase 4 (optimiser) and Phase 5 (web service) use **only** `surrogate.registry`. Demo with outputs:
`surrogate_notebook.ipynb`.

```python
from surrogate.registry import Surrogate, Site, DesignError
from surrogate.paths import EPW_DIR

s = Surrogate.load()                         # reads outputs/models/registry.json; SHA-256 of every file checked
site = s.site("colombo")                     # one of the 10 training climates, by key
site = Site.from_epw(EPW_DIR / "london.epw", name="London")   # any site from its EPW file

res = s.predict(designs, site)               # designs: dict, list of dicts or DataFrame with the 21 inputs
res.values, res.lower, res.upper             # n x 10, original units (kWh/m², W/m²); 90 % ranges by default
res.interval_type                            # "standard" (training climate) or "new_site"
res.climate                                  # level (training / new / extrapolation), distance, nearest climate,
                                             #   out-of-range climate values (plain names)
res.message, res.heating_note                # the agreed user-facing text (config.yaml registry_3_8.messages)
res.flags                                    # per design: design_in_domain, design_violations
res.to_frame(), res.to_dict()                # one table (CSV) / JSON-ready (web service)

Y = s.predict_values(designs, site)          # fast path: NumPy array n x 10 in Surrogate.TARGETS order
s.bounds()                                   # the 21 inputs: min, max, integer — the optimiser's search box
s.info()                                     # what is deployed (models, files, hashes, results)
```

Rules:
- **Designs** use the 21 input names of `simulation/config/design_space.yaml`. A design outside the
  ranges (or a non-integer `n_floors`) is refused with `DesignError` (`strict=True`, default);
  `strict=False` predicts and flags it.
- **One site per call.** Ranges: `level=` 0.5, 0.8, 0.9 (default) or 0.95. Training climate → standard
  range; any other site → the wider new-site range, chosen automatically.
- **Climate levels:** `training`; `new` (close to the training climates and every value inside their range);
  `extrapolation` (further away or any value out of range — advisory warning, prediction still given).
- An EPW must parse with the Phase 2 parser and declare no holidays / daylight saving.
- The heating model hardly responds to shading and thermal mass (≤ 2 %): judge shading through cooling and EUI.
- **Speed** (Apple M4, 22 Nov 2026, `outputs/registry/registry_report.md`): one design 6.7 ms (values) /
  14.1 ms (with ranges); 31,507 designs/s in batches of 100,000. **Measure speed on a rested Mac**
  (plugged in, idle ~10 min): right after heavy work the fanless MacBook Air ran everything ~2× slower.
- The module never imports `final_test`, `explain` or `timing` and never reads holdout files.

Maintenance (Mac, from `Project/`):

```bash
./venv/bin/python surrogate/registry.py --stage build    # slim heating GPR + registry.json (after any model change)
./venv/bin/python surrogate/registry.py --stage check    # acceptance: hashes, self-checks, speed
./venv/bin/python surrogate/tests/test_registry.py       # 16 checks (API = tested 3.7 path, exactly)
```

## Files

| File | Purpose |
|---|---|
| `paths.py` | the only place that knows where Phase 2 artefacts live |
| `config.yaml` / `config.py` | the frozen Phase 3 protocol, selection rule, gate and per-sub-task blocks (append-only, each added before its runs) |
| `check_setup.py`, `install_ml.sh`, `check_ml.py` | shared venv and ML library checks (3.1) |
| `requirements.txt`, `phase2_lock.txt` | exact library versions (Phase 3; Phase 2 constraints) |
| `cv.py` | grouped CV (2.8 folds) and leave-one-climate-out, model-agnostic, holdout-guarded |
| `metrics.py` | 2.8 metrics + ranking (Spearman, top-10 % hit rate), LOCO bias/MAPE |
| `s0_check.py` | gate S0: the 2.8 random forest re-run through the harness (3.1) |
| `features.py` | physics-based features from the 28 raw inputs (3.2) |
| `ablation.py` | feature-set ablation (3.2) |
| `models.py`, `mlp.py` | model factories, target forms, decomposed EUI, PyTorch MLP (3.3–3.4) |
| `tune.py`, `benchmark.py` | Optuna tuning and the model benchmark (3.3) |
| `climate_indices.py`, `climate_gen.py` | weather-file climate indices; unseen-climate study (3.4) |
| `uncertainty.py`, `uncertainty_figures.py` | conformal prediction ranges, applicability domain (3.5) |
| `explain.py`, `explain_figures.py` | final model fit, SHAP, physics checks (3.6) |
| `final_test.py`, `timing.py`, `final_figures.py` | the one-time holdout test and speed test (3.7) |
| **`registry.py`** | **the deployed surrogate: `Surrogate`, `Site`, `Prediction`; build / check (3.8)** |
| `surrogate_notebook.ipynb` | demo of the deployed surrogate (3.8) |
| `tests/`, `run_tests.py` | test suites (14 suites); `run_tests.py` runs them all |
| `outputs/models/` | fitted models (3.6), interval models (3.7), slim heating GPR + `registry.json` (3.8); `gpr_heating.pkl` (72 MB) archived outside git |
