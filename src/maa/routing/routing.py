# Copyright © 2025 Christoph Schlager, TU Wien

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pandera.typing import DataFrame

from maa.config.constants import ProcessingStage
from maa.config.loader import load_inputs_from_config
from maa.config.models.input import LoadedRoutingInputs
from maa.config.utils import LogLevel
from maa.constants.constants import (
    CHUNK_SIZE_PARALLEL_PROCESSING,
    DEFAULT_MAX_WORKERS_PARALLEL_PROCESSING,
    DEFAULT_VALHALLA_BASE_URL,
    NETWORK_COUNTRY,
)
from maa.dataframe.models.route import RouteSchema
from maa.network.network import AffiliationNetworkProcessor
from maa.utils.geo_tools import add_geometry_to_edge_df, add_routes_to_edges
from maa.utils.wrappers import parallelize_dataframe
from maa.znib.utils import get_znib_edges

logger = logging.getLogger(__name__)


def add_routing_parameters_to_edges(
    edge_df: pd.DataFrame,
    affiliation_gdf: gpd.GeoDataFrame,
    valhalla_base_url: str = DEFAULT_VALHALLA_BASE_URL,
    log_level: LogLevel = LogLevel.INFO,
) -> DataFrame[RouteSchema]:
    """
    Merges affiliation locations into the edge DataFrame and enriches it with routing attributes
     such as `travel` time and `distance`.

    :param edge_df:
        DataFrame containing network edges.
    :param affiliation_gdf:
        GeoDataFrame containing affiliation information.
    :param valhalla_base_url:
        The Base URL of the Valhalla API.
    :param log_level:
        The logging level for parallel processing.
    :return:
        DataFrame with routing parameters added.
    """

    # add geometry as coordinates
    edge_df = add_geometry_to_edge_df(
        edge_df=edge_df, affiliation_gdf=affiliation_gdf, as_coordinates=True
    )

    max_workers = max(1, int(DEFAULT_MAX_WORKERS_PARALLEL_PROCESSING) // 2)
    logger.debug("Parallelizing add_routes_to_edges with max_workers=%d", max_workers)

    enriched_edges = parallelize_dataframe(
        input_function=add_routes_to_edges,
        df=edge_df,
        valhalla_base_url=valhalla_base_url,
        log_level=log_level,
        chunk_size=CHUNK_SIZE_PARALLEL_PROCESSING,
        max_workers=max_workers,
        verbose=True,
    )

    return enriched_edges


def create_and_enrich_edges_from_config(
    config_path: Path,
    validate_paths: bool = False,
    write_outputs_to_file: bool = False,
    override_edges: bool = False,
    log_level: LogLevel = LogLevel.INFO,
) -> pd.DataFrame:
    """
    Create co-affiliation edges and enrich them with travel data based on a configuration file.

    Loads inputs and settings from the given configuration file, constructs an
    unfiltered co-affiliation network, and enriches each edge with routing
    parameters using the Valhalla routing engine. Optionally writes the enriched
    edges to disk, overwriting existing output only when `override_edges` is True.

    :param config_path:
        Path to the configuration file defining inputs and routing settings.
    :param validate_paths:
        Validate file and directory paths defined in the configuration.
    :param write_outputs_to_file:
        Write the enriched edges to the output file specified in the config.
    :param override_edges:
        Overwrite the output file if it already exists.
    :param log_level:
        The log level for parallel processing.
    :return:
        A pandas DataFrame containing the enriched edges.
    """
    input_data = load_inputs_from_config(
        config=config_path,
        stage=ProcessingStage.ROUTING.value,
        validate_paths=validate_paths,
    )

    assert isinstance(input_data, LoadedRoutingInputs)

    edges = _get_network(article_df=input_data.articles, affiliation_gdf=input_data.affiliations)
    enriched = add_routing_parameters_to_edges(
        edge_df=edges,
        affiliation_gdf=input_data.affiliations,
        valhalla_base_url=input_data.valhalla_base_url,
        log_level=log_level,
    )

    if write_outputs_to_file:
        output_path = input_data.output_file_path_routes
        file_exists = output_path.exists()

        if override_edges or not file_exists:
            enriched.to_parquet(output_path)
            logger.info(f"Enriched edges written to: {output_path}")
        else:
            logger.info(
                f"Skipping write: --override-edges flag not set and file already exists at "
                f"{output_path}"
            )

    return enriched


def _get_network(article_df: pd.DataFrame, affiliation_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    processor = AffiliationNetworkProcessor(
        article_df=article_df,
        affiliation_gdf=affiliation_gdf,
        country_filter=NETWORK_COUNTRY,
    )
    graph = processor.get_affiliation_graph()
    znib_edges = get_znib_edges(edge_gdf=graph.edge_gdf)
    return znib_edges
