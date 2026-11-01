# Phase 2: Simulation-Based Data Generation

**Project:** A Web-Based Digital Twin Framework with AI Assistance for Passive Energy Optimization in Buildings  
**Researcher:** M.G.I. Tharaka (Ishan) — MSc Computer Science, University of Moratuwa  
**Simulation Engine:** EnergyPlus v24.1  
**Platform:** macOS (Apple Silicon M4)

---

## Overview

This directory contains all code for Phase 2 of the research methodology. It produces the EnergyPlus simulation dataset (~3,400 runs) that trains and validates the ML surrogate models in Phase 3.

**Output:** `outputs/dataset_training.csv` (~2,800 rows) and `outputs/dataset_holdout.csv` (~400 rows), each containing 22–25 design parameter inputs, 8 climate features, and 12 output variables.

---

## Quick Start

### 1. Create and activate a virtual environment

```bash
cd phase2_simulation
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify the environment

```bash
python setup.py
```

All checks should pass before proceeding. The script verifies Python version, all packages, EnergyPlus installation, and directory structure.

### 4. Download EPW weather files (Sub-task 2.3)

Place EPW files into `climate/epw_files/`. See `config/climate_config.yaml` for the required filenames and download URLs.

### 5. Run in order

```bash
# Sub-task 2.2 — test the IDF generator
python run_tests.py              # all test suites (generator + climate)

# Sub-task 2.4 — generate LHS samples
python sampling/sampler.py

# Sub-task 2.5 — pilot run (200 simulations)
python runner/batch_runner.py --mode pilot

# Sub-task 2.6 — full campaign
python runner/batch_runner.py --mode full

# Sub-task 2.8 — EDA
jupyter notebook eda/eda_notebook.ipynb
```

---

## Directory Structure

```
phase2_simulation/
├── config/
│   ├── design_space.yaml       # Parameter bounds for all 22 inputs
│   └── climate_config.yaml     # EPW paths and climate metadata
│
├── idf_generator/              # Sub-task 2.2
│   ├── __init__.py
│   ├── generator.py            # Main parametric IDF builder
│   ├── geometry.py             # 5-zone geometry computation
│   ├── constructions.py        # Envelope construction helpers
│   └── fenestration.py         # Glazing and shading helpers
│
├── sampling/                   # Sub-task 2.4
│   ├── __init__.py
│   ├── sampler.py              # LHS generation and constraint enforcement
│   ├── samples_pilot.csv       # 200-sample pilot batch (generated)
│   ├── samples_main.csv        # ~2,800 main batch (generated)
│   └── samples_holdout.csv     # ~400 holdout set (generated)
│
├── climate/                    # Sub-task 2.3
│   ├── __init__.py
│   ├── epw_files/              # EPW weather files (download manually)
│   ├── climate_features.py     # EPW parser and feature extractor
│   └── climate_metadata.csv   # Climate features table (generated)
│
├── runner/                     # Sub-tasks 2.5–2.6
│   ├── __init__.py
│   ├── batch_runner.py         # Parallel simulation engine
│   ├── output_parser.py        # EnergyPlus output extraction
│   └── run_log.csv             # Per-run status log (generated)
│
├── outputs/
│   ├── raw/                    # Temporary EnergyPlus output files
│   ├── dataset_training.csv    # Final training dataset (generated)
│   └── dataset_holdout.csv     # Final holdout dataset (generated)
│
├── eda/
│   └── eda_notebook.ipynb      # Sub-task 2.8: exploratory data analysis
│
├── setup.py                    # Environment verification script
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| IDF generation | Build from scratch | Geometry changes (area/aspect/floors) require it |
| HVAC | IdealLoadsAirSystem | Isolates passive performance from HVAC efficiency |
| Perimeter depth | 4.57 m fixed | DOE standard (15 ft) |
| Sampling | Stratified LHS, replicated per climate | Space-filling; prevents climate/design confounding |
| Glazing | SimpleGlazingSystem | Accepts U/SHGC/VT directly |
| Output format | EnergyPlus SQL | More robust than ESO text files for batch parsing |
| Parallelism | ProcessPoolExecutor (8 workers) | Apple M4 has 10 cores |

---

## Phase 2 → Phase 3 Handoff Criteria

Phase 2 is complete when:
- `dataset_training.csv` has ≥ 2,500 valid rows across ≥ 5 climate zones
- `dataset_holdout.csv` has ≥ 300 valid rows including ≥ 80 from London (held-out)
- EDA confirms physically plausible output distributions
- Quick RF fit shows R² ≥ 0.85 on 80/20 split

---

## EnergyPlus Installation

EnergyPlus 24.1.0 should be installed at `/Applications/EnergyPlus-24-1-0/`.  
Download: https://github.com/NREL/EnergyPlus/releases/tag/v24.1.0

Verify with:
```bash
/Applications/EnergyPlus-24-1-0/energyplus --version
```
