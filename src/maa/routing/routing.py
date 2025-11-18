import logging

import geopandas as gpd
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame

from src.maa.constants import (
    CHUNK_SIZE_PARALLEL_PROCESSING,
    DEFAULT_MAX_WORKERS_PARALLEL_PROCESSING,
)
from src.maa.dataframe.models.route import RouteSchema
from src.maa.utils.geo_tools import add_geometry_to_edge_df, add_routes_to_edges
from src.maa.utils.wrappers import parallelize_dataframe

logger = logging.getLogger(__name__)


@pa.check_types(lazy=True)
def add_routing_parameters_to_edges(
    edge_df: pd.DataFrame, affiliation_gdf: gpd.GeoDataFrame
) -> DataFrame[RouteSchema]:
    """
    Merges affiliation locations into the edge DataFrame and enriches it with routing attributes
     such as `travel` time and `distance`.

    :param edge_df:
        DataFrame containing network edges.
    :param affiliation_gdf:
        GeoDataFrame containing affiliation information.
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
        chunk_size=CHUNK_SIZE_PARALLEL_PROCESSING,
        max_workers=max_workers,
        verbose=True,
    )

    return enriched_edges
