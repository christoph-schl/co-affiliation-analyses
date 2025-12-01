# Copyright © 2025 Christoph Schlager, TU Wien

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass
from pathlib import Path

from maa.constants.constants import DEFAULT_VALHALLA_BASE_URL

DATA_CONFIG_FOLDER = "config"
DATA_CONFIG_FILE = "config.toml"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIGURATION_PATH = PROJECT_ROOT / DATA_CONFIG_FOLDER / DATA_CONFIG_FILE

# network
DEFAULT_ARTICLE_FILE = Path("data/scopus/article_2013-01-01_2022-12-31.parquet")
DEFAULT_AFFILIATION_FILE = Path("data/scopus/affiliation_2013-01-01_2022-12-31.gpkg")
DEFAULT_OUTPUT_DIR = Path("data/output")
DEFAULT_YEAR_GAP_STABLE = 2

# gravity
DEFAULT_ROUTES_FILE = Path("data/valhalla/enriched_edges.csv")
DEFAULT_FIT_MODELS = True

# plot
DEFAULT_IMPACT_FILE = Path("data/impact/impact_data.csv")
DEFAULT_MIN_SAMPLES = 300
DEFAULT_MAX_GROUP_SIZE = 10


@dataclass(frozen=True)
class NetworkDefaults:
    article_file_path: Path = DEFAULT_ARTICLE_FILE
    affiliation_file_path: Path = DEFAULT_AFFILIATION_FILE
    output_path: Path = DEFAULT_OUTPUT_DIR
    year_gap_stable_links: int = DEFAULT_YEAR_GAP_STABLE


@dataclass(frozen=True)
class GravityDefaults:
    routes_file_path: Path = DEFAULT_ROUTES_FILE
    fit_models: bool = DEFAULT_FIT_MODELS


@dataclass(frozen=True)
class ImpactDefaults:
    impact_file_path: Path = DEFAULT_IMPACT_FILE
    min_samples: int = DEFAULT_MIN_SAMPLES
    max_groups: int = DEFAULT_MAX_GROUP_SIZE


@dataclass(frozen=True)
class RoutingDefaults:
    output_file_path_routes: Path = DEFAULT_ROUTES_FILE
    valhalla_base_url: str = DEFAULT_VALHALLA_BASE_URL


DEFAULT_CONFIG_CONTENT = {
    "network": asdict(NetworkDefaults()),
    "gravity": asdict(GravityDefaults()),
    "impact": asdict(ImpactDefaults()),
    "routing": asdict(RoutingDefaults()),
}


class ProcessingStage(enum.Enum):
    PREPROCESSING = "network"
    GRAVITY = "gravity"
    IMPACT = "impact"
    ROUTING = "routing"
