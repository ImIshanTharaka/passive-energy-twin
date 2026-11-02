"""
config.py - loads surrogate/config.yaml (the frozen Phase 3 protocol and gate).

    from surrogate.config import CONFIG
    CONFIG["protocol"]["seed"]   # 42
"""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


CONFIG = load_config()
PRIMARY_TARGETS: list[str] = CONFIG["targets"]["primary"]
SECONDARY_TARGETS: list[str] = CONFIG["targets"]["secondary"]
ALL_TARGETS: list[str] = PRIMARY_TARGETS + SECONDARY_TARGETS
SEED: int = CONFIG["protocol"]["seed"]
