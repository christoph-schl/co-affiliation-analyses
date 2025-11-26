from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import pandas as pd
import structlog

from maa.config.constants import ProcessingStage
from maa.config.loader import load_inputs_from_config
from maa.config.models.input import GravityConfig, LoadedGravityInputs
from maa.config.models.output import (
    GravityResultDatasets,
    ZNIBGravityResult,
    iter_year_gaps,
    write_outputs,
)
from maa.constants.constants import NETWORK_COUNTRY, ORG_TYPE_FILTER_LIST
from maa.dataframe.models.route import get_empty_routes_df
from maa.network.network import AffiliationNetworkProcessor
from maa.utils.wrappers import get_execution_time
from maa.znib.configuration import (
    ZINBConfig,
    config_inter_proximity,
    config_intra_proximity,
)
from maa.znib.model import ZINBModel
from maa.znib.utils import enrich_edges_with_org_info

_logger = structlog.getLogger(__name__)


@dataclass
class ZNIB(AffiliationNetworkProcessor):
    """
    A processor for preparing input data for, and performing modelling with, the
    **Zero-Inflated Negative Binomial (ZNIB)** model to estimate the degree of proximity and
    distance in inter-organisational collaboration.

    This class inherits from :class:`AffiliationNetworkProcessor` and extends its
    functionality to both construct and analyse edge-level features for modelling
    collaboration intensity between affiliations. It encapsulates the workflow
    required to prepare, enrich, and optionally fit ZNIB models that capture
    the probabilistic structure of multi-affiliation collaboration networks.

    Core responsibilities include:
      - Generating and enriching affiliation-to-affiliation edge data.
      - Integrating travel-time and proximity indicators between organisations.
      - Applying filtering by edge weight and organisation types.
      - Producing a feature-rich dataset suitable for ZNIB model fitting.
      - Supporting subsequent model estimation, diagnostics, and interpretation.

    Inherits from:
        AffiliationNetworkProcessor: Provides the base methods and attributes
        for managing the affiliation graph and edge data structures.
    """

    _enriched_edges: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    @property
    def enriched_edges(self) -> Optional[pd.DataFrame]:
        return self._enriched_edges

    @enriched_edges.setter
    def enriched_edges(self, value: Optional[pd.DataFrame]) -> None:
        self._enriched_edges = value

    @get_execution_time
    def enrich_edges_with_org_info(
        self,
        route_df: pd.DataFrame,
        min_edge_weight: Optional[int] = None,
        org_type_list: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Creates the base edge-level DataFrame if not cached and the input DataFrame for the
        ZNIB (Zero-Inflated Negative Binomial) model.

        This function prepares and enriches edge-level data to serve as input for
        a ZNIB gravity model of multi-affiliation collaboration intensity.
        It performs the following steps:

          1. Generates all affiliation-to-affiliation edge combinations.
          2. Merges travel-time information for each edge of the two affiliations (from → to).
          3. Applies a binary proximity indicator variable to represent organisational
             proximity or distance between affiliations.
          4. Adds gravity model–related variables such as log-transformed
             article counts and travel durations.

        :param route_df:
            DataFrame containing travel time information between affiliation pairs.
        :param min_edge_weight:
            The minimum link strength to retain. Edges with lower weights are removed from
            the graph. If ``None``, all edges are included.
        :param org_type_list:
            Optional list of organisation types used to filter the edges connecting organisations.

        :return:
            A fully enriched DataFrame containing:
              - all affiliation-to-affiliation edges
              - merged travel-time and route information
              - proximity dummy variables indicating organisational proximity or distance
              - gravity model features (log of article count products and travel durations)
            ready for input into a ZNIB regression model.
        """

        self._enrich_edges_with_org_info(
            route_df=route_df, org_type_list=org_type_list, min_edge_weight=min_edge_weight
        )
        return self.enriched_edges

    def _enrich_edges_with_org_info(
        self,
        route_df: pd.DataFrame,
        org_type_list: Optional[List[str]],
        min_edge_weight: Optional[int] = None,
    ) -> None:

        self._create_affiliation_graph(min_edge_weight=min_edge_weight)

        if self.enriched_edges is None:
            self.enriched_edges = enrich_edges_with_org_info(
                edge_gdf=self.edge.edge_gdf, route_df=route_df, org_type_list=org_type_list
            )

    @get_execution_time
    def fit_znib(
        self, config: ZINBConfig, enriched_edge_df: Optional[pd.DataFrame] = None
    ) -> ZINBModel:
        """
        Fits a Zero-Inflated Negative Binomial (ZINB) model on the enriched edge data.

        This method creates and fits a `ZINBModel` instance using either the internally stored
        enriched edges or an optionally provided edge DataFrame. It returns the fitted model
        object for further inspection or analysis.

        :param config:
            ZINB configuration object specifying the dependent variable, predictor variables,
            zero-inflation specification, and other model parameters to be used in fitting
            the model.
        :param enriched_edge_df:
            The enriched edge dataframe.
        :return:
            The fitted ZNIB model.
        """

        if enriched_edge_df is not None:
            self.enriched_edges = enriched_edge_df

        znib_model = ZINBModel(df=self.enriched_edges, config=config)
        znib_model.fit()
        return znib_model


def get_gravity_model_for_year_gaps(
    article_df: pd.DataFrame,
    affiliation_gdf: gpd.GeoDataFrame,
    routes_df: pd.DataFrame,
    gravity_cfg: GravityConfig,
) -> GravityResultDatasets:
    """
    Build ZNIB gravity model input for each configured year-gap variant.

    Generates results for:
      • the complete co-affiliation dataset ("all"), and
      • the stable co-affiliation variant ("stable"),
    as defined in the network configuration.

    :param article_df:
        DataFrame containing article metadata.
    :param affiliation_gdf:
        GeoDataFrame containing affiliation information.
    :param routes_df:
        DataFrame containing travel time information for each affiliation pair.
    :param gravity_cfg:
        GravityConfig object defining year-gap parameters and paths.
    :Yields:
        YearGapResult:
            An object containing:
                • suffix: the variant name ("all", "stable", ...)
                • graph: the constructed affiliation graph
                • link_gdf: the GeoDataFrame of computed affiliation links
    """

    znib = ZNIB(
        article_df=article_df, affiliation_gdf=affiliation_gdf, country_filter=NETWORK_COUNTRY
    )

    gravity_results: Dict[str, ZNIBGravityResult] = {}
    for yg in iter_year_gaps(gravity_cfg.year_gap_stable_links):
        znib.min_year_gap = yg.gap
        _logger.info("processing.year_gap", gap=yg.gap, suffix=yg.suffix)
        model_data = znib.enrich_edges_with_org_info(
            route_df=routes_df,
            org_type_list=ORG_TYPE_FILTER_LIST,
        )

        znib_intra_model = None
        znib_inter_model = None
        if gravity_cfg.fit_models:
            _logger.info(
                "fit intra organisational znib gravity model", gap=yg.gap, suffix=yg.suffix
            )
            znib_intra_model = znib.fit_znib(config=config_intra_proximity)
            _logger.info(znib_intra_model.summary)

            _logger.info(
                "fit inter organisational znib gravity model", gap=yg.gap, suffix=yg.suffix
            )
            znib_inter_model = znib.fit_znib(config=config_inter_proximity)
            _logger.info(znib_inter_model.summary)

        gravity_results[yg.suffix] = ZNIBGravityResult(
            suffix=yg.suffix,
            graph=znib.edge,
            link_gdf=znib.link,
            znib_data=model_data,
            znib_intra_model=znib_intra_model,
            znib_inter_model=znib_inter_model,
        )

    gravity_datasets = GravityResultDatasets.from_dict(data=gravity_results)
    return gravity_datasets


def create_znib_gravity_models_from_config(
    config_path: Path,
    validate_paths: bool = False,
    write_outputs_to_file: bool = False,
) -> GravityResultDatasets:
    """
    Build ZNIB gravity model inputs for each configured year-gap variant and fits models if
    specified in the configuration file.

    This function loads all required data from the given configuration file,
    constructs gravity-model inputs for each year-gap variant (including the
    complete dataset and the stable co-affiliation variant), and optionally
    writes the generated outputs to the configured output directory.

    :param config_path:
        Path to the configuration file specifying input data sources,
        processing parameters, and output paths.
    :param debug:
        Enable verbose logging for debugging or development purposes.
    :param validate_paths:
        Validate the existence of all required input and output paths
        before running the pipeline.
    :param write_outputs_to_file:
        If True, write the generated gravity-model artifacts to disk.
        If False, results are produced but not persisted.
    :Yields:
        ZNIBGravityResult:
            For each year-gap variant, an object containing:
                • suffix: the variant name ("all", "stable", ...)
                • data: the prepared gravity-model inputs
                • metadata: additional information required for modeling
    """

    input_data = load_inputs_from_config(
        config=config_path,
        stage=ProcessingStage.GRAVITY.value,
        validate_paths=validate_paths,
    )
    gravity_cfg = input_data.config
    article_df = input_data.articles
    affiliation_gdf = input_data.affiliations
    if isinstance(input_data, LoadedGravityInputs):
        routes_df = input_data.routes
    else:
        routes_df = get_empty_routes_df()

    # 🔥type narrowing for mypy:
    assert isinstance(gravity_cfg, GravityConfig)
    results = get_gravity_model_for_year_gaps(
        article_df=article_df,
        affiliation_gdf=affiliation_gdf,
        routes_df=routes_df,
        gravity_cfg=gravity_cfg,
    )

    if write_outputs_to_file:
        write_outputs(results=results, output_path=gravity_cfg.output_path)

    return results
