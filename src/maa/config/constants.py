# src/maa/config/constants.py
from __future__ import annotations

import enum
from pathlib import Path

DATA_CONFIG_FOLDER = "config"
DATA_CONFIG_FILE = "config.toml"

CONFIGURATION_PATH = Path(__file__).resolve().parents[3] / DATA_CONFIG_FOLDER / DATA_CONFIG_FILE


class ProcessingStage(enum.Enum):
    PREPROCESSING = "network"
    MODEL = "model"
    POSTPROCESSING = "postprocessing"
