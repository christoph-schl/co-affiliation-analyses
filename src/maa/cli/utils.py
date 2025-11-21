from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Union

import geopandas as gpd
import nx2vos
import pandas as pd
import structlog

from maa.config import load_config_for_stage
from maa.config.models import input_types
from maa.config.utils import configure_logging
from maa.constants.constants import (
    AFFILIATION_LINKS_PREFIX,
    GRAVITY_INPUT_DIR,
    GRAVITY_INPUT_PREFIX,
    GRAVITY_INTER_RESULTS_PREFIX,
    GRAVITY_INTRA_RESULTS_PREFIX,
    GRAVITY_OUTPUT_DIR,
    LINK_DIR,
    VOS_DIR,
    VOS_MAP_PREFIX,
    VOS_NETWORK_PREFIX,
)
from maa.znib.znib import ZINBModel

_logger = structlog.getLogger(__name__)


def load_inputs_from_config(
    config: Path,
    stage: str,
    validate_paths: bool,
    debug: bool,
) -> input_types:
    """
    Load all required inputs for a given stage using the provided configuration file.

    :param config:
        Path to the configuration file (YAML/TOML) containing the pipeline settings.
    :param stage:
        Name of the stage-group to load configuration for. Determines which sections
        of the config file are activated.
    :param validate_paths:
        Whether to verify that all configured file paths exist on disk. If True, an
        error will be raised for missing paths.
    :param debug:
        Enables verbose logging output when True.
    :return:
        A fully populated LoadedInputs object containing articles, affiliations,
        routes (if applicable), and references to the loaded configuration.
    """

    configure_logging(debug=debug)

    cfg = load_config_for_stage(
        config_file=config,
        stage_group=stage,
        validate_paths_exist=validate_paths,
    )

    _logger.info("config.loaded", config=str(config), stage=stage)
    return cfg.load_inputs()


@dataclass(frozen=True)
class YearGapEntry:
    """Defines a single year-gap variant (gap + name suffix)."""

    gap: int
    suffix: str


class OutputPaths:
    """
    Generates output file paths for a given suffix and ensures parent dirs exist.
    """

    def __init__(self, root: Path, suffix: str):
        self.root = root
        self.suffix = suffix

    def _path(self, subdir: Path, filename: str) -> Path:
        path = self.root / subdir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def links(self) -> Path:
        return self._path(LINK_DIR, f"{AFFILIATION_LINKS_PREFIX}_{self.suffix}.gpkg")

    @property
    def map(self) -> Path:
        return self._path(VOS_DIR, f"{VOS_MAP_PREFIX}_{self.suffix}.txt")

    @property
    def network(self) -> Path:
        return self._path(VOS_DIR, f"{VOS_NETWORK_PREFIX}_{self.suffix}.txt")

    @property
    def gravity(self) -> Path:
        return self._path(GRAVITY_INPUT_DIR, f"{GRAVITY_INPUT_PREFIX}_{self.suffix}.txt")

    @property
    def intra_result(self) -> Path:
        return self._path(GRAVITY_OUTPUT_DIR, f"{GRAVITY_INTRA_RESULTS_PREFIX}_{self.suffix}.txt")

    @property
    def inter_result(self) -> Path:
        return self._path(GRAVITY_OUTPUT_DIR, f"{GRAVITY_INTER_RESULTS_PREFIX}_{self.suffix}.txt")


@dataclass(frozen=True)
class NetworkResult:
    """Result object for a computed network variant."""

    suffix: str
    graph: Any
    link_gdf: gpd.GeoDataFrame

    def write(self, output_paths: "OutputPaths") -> None:
        """Write the components common to all network-like results."""
        self.link_gdf.to_file(output_paths.links)
        nx2vos.write_vos_map(G=self.graph.graph, fname=output_paths.map)
        nx2vos.write_vos_network(G=self.graph.graph, fname=output_paths.network)


def _write_model(model: Optional[ZINBModel], path: Path) -> None:
    if model is None:
        return
    df = getattr(model, "result_df", None)
    if df is not None:
        df.to_csv(path, index=False)


@dataclass(frozen=True)
class ZNIBGravityResult(NetworkResult):
    """Result of a gravity model run (extends a base network)."""

    znib_data: pd.DataFrame
    znib_intra_model: Optional[ZINBModel] = None
    znib_inter_model: Optional[ZINBModel] = None

    def write(self, output_paths: "OutputPaths") -> None:
        """Write base network outputs + gravity-specific outputs."""
        super().write(output_paths)

        self.znib_data.to_csv(output_paths.gravity, index=False)

        _write_model(self.znib_intra_model, output_paths.intra_result)
        _write_model(self.znib_inter_model, output_paths.inter_result)


def iter_year_gaps(stable_gap: int) -> Iterator[YearGapEntry]:
    """Yield configured year-gap variants."""
    yield YearGapEntry(gap=0, suffix="all")
    yield YearGapEntry(gap=stable_gap, suffix="stable")


def write_outputs(
    results: Iterable[Union[NetworkResult, ZNIBGravityResult]], output_path: Path, dry_run: bool
) -> None:
    """Write each year-gap result to disk."""
    for result in results:
        if dry_run:
            _logger.info("dry_run.write", suffix=result.suffix, output=str(output_path))
            continue

        paths = OutputPaths(output_path, result.suffix)

        try:
            result.write(paths)
        except Exception as exc:  # noqa intentionally broad
            _logger.error("write.failed", suffix=result.suffix, error=str(exc))
            raise
        else:
            _logger.info("write.success", suffix=result.suffix, output=str(output_path))
