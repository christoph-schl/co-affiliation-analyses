from __future__ import annotations

import enum
from pathlib import Path

DATA_CONFIG_FOLDER = "config"
DATA_CONFIG_FILE = "config.toml"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIGURATION_PATH = PROJECT_ROOT / DATA_CONFIG_FOLDER / DATA_CONFIG_FILE


class ProcessingStage(enum.Enum):
    PREPROCESSING = "network"
    GRAVITY = "gravity"
    POSTPROCESSING = "postprocessing"
