"""
paths.py - the single place that knows where Phase 2 artefacts live (Phase 3 plan §4.2).

Phase 2 (`simulation/`) is read-only for Phase 3. Every surrogate module takes its
paths from here; nothing else hard-codes a location.
"""

from __future__ import annotations

from pathlib import Path

SURROGATE_ROOT = Path(__file__).resolve().parent          # Project/surrogate
PROJECT_ROOT = SURROGATE_ROOT.parent                      # Project
SIM_ROOT = PROJECT_ROOT / "simulation"                    # Phase 2, frozen

# Phase 2 artefacts (read-only)
DATASET_DIR = SIM_ROOT / "outputs" / "dataset"            # frozen dataset + sha256 manifest
DESIGN_SPACE = SIM_ROOT / "config" / "design_space.yaml"
CLIMATE_CONFIG = SIM_ROOT / "config" / "climate_config.yaml"
EPW_DIR = SIM_ROOT / "climate" / "epw_files"              # <key>.epw, used by climate_indices.py (3.4)
OAT_DIR = SIM_ROOT / "outputs" / "oat"                    # 86 one-at-a-time EnergyPlus runs (3.6)

# Phase 3 outputs
OUT = SURROGATE_ROOT / "outputs"
