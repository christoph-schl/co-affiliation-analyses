# Copyright © 2025 Christoph Schlager, TU Wien

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

import geopandas as gpd
import nx2vos
import pandas as pd
import structlog

from maa.constants.constants import (
    AFFILIATION_LINKS_PREFIX,
    BAR_PLOT_FILENAME,
    CO_AFF_ALL_DATASET_NAME,
    CO_AFF_STABLE_DATASET_NAME,
    GRAVITY_INPUT_DIR,
    GRAVITY_INPUT_PREFIX,
    GRAVITY_INTER_RESULTS_PREFIX,
    GRAVITY_INTRA_RESULTS_PREFIX,
    GRAVITY_OUTPUT_DIR,
    LINK_DIR,
    PLOTS_OUTPUT_DIR,
    TIMESERIES_PLOT_INSTITUTION_FILENAME,
    TIMESERIES_PLOT_ORG_TYPE_FILENAME,
    VIOLINE_PLOT_FILENAME,
    VOS_DIR,
    VOS_MAP_PREFIX,
    VOS_NETWORK_PREFIX,
)
from maa.network.container import AffiliationGraph
from maa.plot.grid import PlotGrid
from maa.znib.model import ZINBModel

_logger = structlog.getLogger(__name__)


@dataclass(frozen=True)
class YearGapEntry:
    """Defines a single year-gap variant (gap + name suffix)."""

    gap: int
    suffix: str


class OutputPaths:
    """
    Generates output file paths for a given suffix and ensures parent dirs exist.
    """

    def __init__(self, root: Path, suffix: Optional[str]):
        self.root: Path = root
        self.suffix: Optional[str] = suffix

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

    @property
    def violine_plot(self) -> Path:
        return self._path(PLOTS_OUTPUT_DIR, VIOLINE_PLOT_FILENAME)

    @property
    def bat_plot(self) -> Path:
        return self._path(PLOTS_OUTPUT_DIR, BAR_PLOT_FILENAME)

    @property
    def timeseries_plot_org_type(self) -> Path:
        return self._path(PLOTS_OUTPUT_DIR, TIMESERIES_PLOT_ORG_TYPE_FILENAME)

    @property
    def timeseries_plot_institution(self) -> Path:
        return self._path(PLOTS_OUTPUT_DIR, TIMESERIES_PLOT_INSTITUTION_FILENAME)


@dataclass(frozen=True)
class NetworkResult:
    """Result object for a computed network variant."""

    suffix: str
    graph: AffiliationGraph
    link_gdf: gpd.GeoDataFrame

    def write(self, output_paths: "OutputPaths") -> None:
        """Write the components common to all network-like results."""
        self.link_gdf.to_file(output_paths.links)
        nx2vos.write_vos_map(G=self.graph.graph, fname=output_paths.map)
        nx2vos.write_vos_network(G=self.graph.graph, fname=output_paths.network)


@dataclass(frozen=True)
class CoAffiliationNetworks:
    """
    Contains co-affiliation networks based on:
    - `all`:     unfiltered co-affiliations
    - `stable`:  filtered (stable) co-affiliations
    """

    all: NetworkResult
    stable: NetworkResult

    @classmethod
    def from_dict(cls, data: Dict[str, NetworkResult]) -> "CoAffiliationNetworks":
        return cls(
            all=data[CO_AFF_ALL_DATASET_NAME],
            stable=data[CO_AFF_STABLE_DATASET_NAME],
        )


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


@dataclass(frozen=True)
class GravityResultDatasets:
    """
    Container for gravity model results based on:
    - `all`:     unfiltered co-affiliations
    - `stable`:  filtered (stable) co-affiliations
    """

    all: ZNIBGravityResult
    stable: ZNIBGravityResult

    @classmethod
    def from_dict(cls, data: Dict[str, ZNIBGravityResult]) -> "GravityResultDatasets":
        return cls(
            all=data[CO_AFF_ALL_DATASET_NAME],
            stable=data[CO_AFF_STABLE_DATASET_NAME],
        )


@dataclass(frozen=True)
class PlotResult:
    """Result object for a plotting resuls."""

    violine_plot: PlotGrid
    bar_plot: PlotGrid
    timeseries_plot_org_type: PlotGrid
    timeseries_plot_institution: PlotGrid

    def write(self, output_paths: "OutputPaths") -> None:
        """Write the components common to all plot grids."""
        self.violine_plot.save(output_paths.violine_plot)
        self.bar_plot.save(output_paths.bat_plot)
        self.timeseries_plot_org_type.save(output_paths.timeseries_plot_org_type)
        self.timeseries_plot_institution.save(output_paths.timeseries_plot_institution)


def iter_year_gaps(stable_gap: int) -> Iterator[YearGapEntry]:
    """Yield configured year-gap variants."""
    yield YearGapEntry(gap=0, suffix=CO_AFF_ALL_DATASET_NAME)
    yield YearGapEntry(gap=stable_gap, suffix=CO_AFF_STABLE_DATASET_NAME)


def _resolve_paths(base: Path, result: Any) -> OutputPaths:
    suffix = getattr(result, "suffix", None)
    return OutputPaths(base, suffix)


def write_outputs(
    results: Union[Union[CoAffiliationNetworks, GravityResultDatasets], PlotResult],
    output_path: Path,
) -> None:

    # normalize input into an iterable of result objects
    if isinstance(results, PlotResult):
        results_iter = [results]
    else:
        results_iter = [
            getattr(results, CO_AFF_ALL_DATASET_NAME),
            getattr(results, CO_AFF_STABLE_DATASET_NAME),
        ]

    for result in results_iter:
        suffix = getattr(result, "suffix", None)

        paths = _resolve_paths(base=output_path, result=result)

        try:
            result.write(paths)
        except Exception as exc:  # noqa
            _logger.error("write.failed", suffix=suffix, error=str(exc))
            raise

    _logger.info("write.success", output=str(output_path))
