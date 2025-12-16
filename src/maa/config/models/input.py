# Copyright © 2025 Christoph Schlager, TU Wien

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import geopandas as gpd
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from maa.config.constants import PROJECT_ROOT
from maa.config.utils import download_zenodo_record
from maa.dataframe.models.affiliation import read_affiliations
from maa.dataframe.models.article import read_articles
from maa.dataframe.models.route import read_routes

_ZENODO_ARTICLE_RECORD_ID = "17952177"
_ZENODO_AFFILIATION_RECORD_ID = "17953806"
_ZENODO_ROUTES_RECORD_ID = "17954106"
_ZENODO_IMPACT_RECORD_ID = "-9999"


@dataclass(frozen=True)
class LoadedNetworkInputs:
    """Inputs required for network computation."""

    config: NetworkConfig
    articles: pd.DataFrame
    affiliations: gpd.GeoDataFrame


@dataclass(frozen=True)
class LoadedGravityInputs(LoadedNetworkInputs):
    """Inputs required for gravity computation (extends network inputs)."""

    routes: pd.DataFrame
    fit_models: bool


@dataclass(frozen=True)
class LoadedPlotInputs(LoadedNetworkInputs):
    """Inputs required for gravity computation (extends network inputs)."""

    impact: pd.DataFrame
    min_samples: int
    max_groups: int


@dataclass(frozen=True)
class LoadedRoutingInputs(LoadedNetworkInputs):
    """Inputs required for gravity computation (extends network inputs)."""

    output_file_path_routes: Path
    valhalla_base_url: str


input_types = Union[LoadedNetworkInputs, LoadedGravityInputs, LoadedPlotInputs, LoadedRoutingInputs]


def _expand_str(s: str) -> str:
    return os.path.expandvars(os.path.expanduser(s))


class BaseConfig(BaseModel, ABC):
    """
    Base config for all pipeline stages:
    - expands ~ and env vars
    - coerces string paths to Path
    - resolves relative paths against PROJECT_ROOT
    """

    data_root: Optional[Path] = Field(
        None, description="Optional root for resolving relative paths"
    )

    @abstractmethod
    def load_inputs(self, download_if_missing: bool = False) -> input_types:
        """Load all required inputs for this config type."""
        raise NotImplementedError

    @field_validator("data_root", mode="before")
    def _normalize_data_root(cls, v: Any) -> Optional[Path]:
        if v is None:
            return None
        p = Path(_expand_str(str(v)))
        return p.expanduser().resolve()

    @model_validator(mode="before")
    def _coerce_path_like_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert string fields ending with *_path or *_dir into absolute paths.
        """
        for name, val in list(values.items()):
            if not isinstance(val, str):
                continue

            is_path = name in cls.model_fields and (
                cls.model_fields[name].annotation is Path
                or Path in getattr(cls.model_fields[name].annotation, "__args__", ())
            )

            looks_like_path = "path" in name.lower() or name.lower().endswith("_dir")

            if is_path or looks_like_path:
                p = Path(_expand_str(val))
                values[name] = p if p.is_absolute() else (PROJECT_ROOT / p)

        return values

    def check_paths_exist(self) -> None:
        missing = [
            f"{k}: {v}" for k, v in self.__dict__.items() if isinstance(v, Path) and not v.exists()
        ]
        if missing:
            raise FileNotFoundError("Missing config paths:\n" + "\n".join(missing))


class NetworkConfig(BaseConfig):
    article_file_path: Path = Field(..., description="Parquet with article records")
    affiliation_file_path: Path = Field(..., description="Affiliation file")
    output_path: Path = Field(..., description="Directory or file to write outputs")
    year_gap_stable_links: int = Field(..., description="Year gap for stable links")
    download_if_missing: bool = Field(..., description="Flag whether to download missing files")

    def load_inputs(self, download_if_missing: bool = False) -> "LoadedNetworkInputs":
        if not self.article_file_path.exists() and download_if_missing:
            download_zenodo_record(
                record_id=_ZENODO_ARTICLE_RECORD_ID, output_file_path=self.article_file_path
            )
        articles = read_articles(self.article_file_path)

        if not self.affiliation_file_path.exists() and download_if_missing:
            download_zenodo_record(
                record_id=_ZENODO_AFFILIATION_RECORD_ID, output_file_path=self.affiliation_file_path
            )

        affiliations = read_affiliations(self.affiliation_file_path)

        return LoadedNetworkInputs(
            config=self,
            articles=articles,
            affiliations=affiliations,
        )


class GravityConfig(NetworkConfig):
    routes_file_path: Path = Field(
        ..., description="CSV with travel time information for gravity modelling"
    )
    fit_models: bool = Field(..., description="Whether to fit the gravity models.")

    def load_inputs(self, download_if_missing: bool = False) -> "LoadedGravityInputs":
        # Load the inherited inputs first
        base = super().load_inputs(download_if_missing=download_if_missing)

        # Load gravity-specific inputs
        if not self.routes_file_path.exists() and download_if_missing:
            download_zenodo_record(
                record_id=_ZENODO_ROUTES_RECORD_ID, output_file_path=self.routes_file_path
            )
        routes = read_routes(self.routes_file_path)

        return LoadedGravityInputs(
            config=self,
            articles=base.articles,
            affiliations=base.affiliations,
            routes=routes,
            fit_models=self.fit_models,
        )


class RoutingConfig(NetworkConfig):
    output_file_path_routes: Path = Field(
        ..., description="Output file path for CSV with travel time information"
    )
    valhalla_base_url: str = Field(..., description="The base URL for the Valhalla routing engine.")

    def load_inputs(self, download_if_missing: bool = False) -> "LoadedRoutingInputs":
        # Load the inherited inputs first
        base = super().load_inputs(download_if_missing=download_if_missing)

        return LoadedRoutingInputs(
            config=self,
            articles=base.articles,
            affiliations=base.affiliations,
            output_file_path_routes=self.output_file_path_routes,
            valhalla_base_url=self.valhalla_base_url,
        )


class ImpactConfig(NetworkConfig):
    impact_file_path: Path = Field(..., description="Parquet with impact data for article records")
    min_samples: int = Field(..., description="Minimum number of samples to include")
    max_groups: int = Field(..., description="Maximum number of groups to include")

    def load_inputs(self, download_if_missing: bool = False) -> "LoadedNetworkInputs":
        base = super().load_inputs(download_if_missing=download_if_missing)

        if not self.impact_file_path.exists() and download_if_missing:
            download_zenodo_record(
                record_id=_ZENODO_IMPACT_RECORD_ID, output_file_path=self.impact_file_path
            )

        impact = pd.read_csv(self.impact_file_path)

        return LoadedPlotInputs(
            config=self,
            articles=base.articles,
            affiliations=base.affiliations,
            impact=impact,
            min_samples=self.min_samples,
            max_groups=self.max_groups,
        )


config_types = Union[NetworkConfig, GravityConfig, ImpactConfig, RoutingConfig]
