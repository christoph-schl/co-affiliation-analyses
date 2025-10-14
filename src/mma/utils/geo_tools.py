from enum import Enum
from typing import List, Tuple, Union

import geopandas as gpd
import pandas as pd
import structlog
from geopandas import GeoDataFrame
from routingpy import Valhalla
from shapely import LineString

from src.mma.constants import (
    AFFILIATION_ID_COLUMN,
    AFFILIATION_ID_FROM_COLUMN,
    AFFILIATION_ID_TO_COLUMN,
    DEFAULT_VALHALLA_BASE_URL,
    DISTANCE_M_COLUMN,
    DURATION_S_COLUMN,
    FROM_NODE_COLUMN,
    GEOMETRY_COLUMN,
    TO_NODE_COLUMN,
)
from src.mma.utils.wrappers import df_split

_logger = structlog.getLogger()
_TEMPORARY_GEOMETRY_COLUMN = "temp_geom"
_COORDS_COLUMN = "coords"
_SOURCE_COLUMN = "source"
_DF_INDEX_COLUMN = "df_index"
_CHUNK_SIZE_ROUTING_MATRIX = 25


def drop_empty_geometries(gdf: GeoDataFrame) -> Tuple[GeoDataFrame, int]:
    """
    Removes rows with empty or missing geometries from a GeoDataFrame.

    :param gdf: The input GeoDataFrame.
    :return: A cleaned GeoDataFrame with only valid geometries and the number of rows removed.
    """
    empty_geometry_index = gdf[(gdf.geometry.isnull()) | (gdf.geometry.empty)].index
    if not empty_geometry_index.empty:
        gdf = gdf.drop(index=empty_geometry_index).reset_index(drop=True)
    return gdf, len(empty_geometry_index)


def create_line_geometries(
    df: Union[gpd.GeoDataFrame, pd.DataFrame], geometry_columns: List[str], drop_invalid_geoms: bool
) -> Union[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Create LineString geometries by connecting points from specified geometry columns.

    This function takes a DataFrame or GeoDataFrame with columns containing Point geometries,
    optionally removes invalid or empty geometries, and constructs a new LineString geometry
    for each row by connecting the points in the specified columns.

    :param df: Input DataFrame or GeoDataFrame containing Point geometries.
    :param geometry_columns: List of column names containing Point geometries to connect.
                            The points will be connected in the order they appear in this list.
    :param drop_invalid_geoms: If True, rows containing None, NaN, or empty geometries
                                       will be removed before creating LineStrings.
    :return:  A DataFrame or GeoDataFrame with a new column
            containing LineString geometries constructed from the specified point columns.
            The original geometry column (if present) will be replaced.
    """

    if drop_invalid_geoms:
        df = drop_invalid_geometries(df=df, geometry_columns=geometry_columns)

    df[_TEMPORARY_GEOMETRY_COLUMN] = df[geometry_columns].values.tolist()
    df = df.drop(columns=[GEOMETRY_COLUMN], errors="ignore")
    df[GEOMETRY_COLUMN] = df[_TEMPORARY_GEOMETRY_COLUMN].apply(LineString)
    df = df.drop(columns=_TEMPORARY_GEOMETRY_COLUMN)

    return df


def drop_invalid_geometries(
    df: Union[gpd.GeoDataFrame, pd.DataFrame], geometry_columns: List[str]
) -> Union[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Remove rows containing invalid, missing, or empty geometries from specified columns.

    This function iterates over the provided geometry columns and drops any row where
    the geometry is either None, NaN, or an empty Shapely geometry (e.g., <POINT EMPTY>).
    It resets the index of the resulting DataFrame and logs the number of rows dropped.

    :param df: Input DataFrame or GeoDataFrame containing geometries.
              geometry_columns (List[str]): List of column names containing geometries to validate.
    :param geometry_columns: List of column names containing geometries to validate.
    :return: A DataFrame or GeoDataFrame with all rows containing invalid or empty geometries
            removed.
    """

    for column in geometry_columns:
        # ~gpd.GeoSeries(gdf.geometry).is_empty is fast (vectorized), but misses some “empty”
        # geometries, because Shapely/GeoPandas treat <POINT EMPTY> inconsistently when they’re
        # stored as object dtype or mixed types
        df = df[
            (df[column].notna())
            & (df[column].notnull())
            & (~df[column].apply(lambda g: g.is_empty if g else True))
        ]

    df = df.reset_index(drop=True)
    return df


def add_geometry_to_edge_df(
    edge_df: pd.DataFrame, affiliation_gdf: gpd.GeoDataFrame, as_coordinates: bool = False
) -> pd.DataFrame:
    """
     Attach geometry for the 'from' and 'to' affiliation IDs to an edge dataframe.

     The function expects the following globals to be defined in the caller's module:
       - FROM_NODE_COLUMN (name of column in edge_df containing origin affiliation id)
       - TO_NODE_COLUMN   (name of column in edge_df containing destination affiliation id)
       - GEOMETRY_COLUMN  (name of the geometry column in affiliation_gdf, typically "geometry")
       - AFFILIATION_ID_COLUMN (name of the affiliation id column in affiliation_gdf)
       - _COORDS_COLUMN   (name you want for coordinate lists when as_coordinates=True)

     :param edge_df: The edge DataFrame.
     :param affiliation_gdf: The GeoDataFrame with affiliations
     :param as_coordinates: If True, geometry is converted to a list of floats [x, y] representing
                            coordinates.
    :return: The edge DataFrame with the original edge columns and geometry or coordinate columns
             added.
    """

    # column aliases
    from_col = FROM_NODE_COLUMN
    to_col = TO_NODE_COLUMN
    geom_col = GEOMETRY_COLUMN
    affiliation_id_col = AFFILIATION_ID_COLUMN
    coords_col = _COORDS_COLUMN

    edge_df = edge_df.copy()
    edge_df = (
        edge_df[[from_col, to_col]]
        .merge(
            right=affiliation_gdf[[affiliation_id_col, geom_col]],
            left_on=from_col,
            right_on=affiliation_id_col,
            how="left",
            suffixes=("", "_from"),
        )
        .merge(
            right=affiliation_gdf[[affiliation_id_col, geom_col]],
            left_on=to_col,
            right_on=affiliation_id_col,
            how="left",
            suffixes=("", "_to"),
        )
        .drop(columns=[affiliation_id_col, f"{affiliation_id_col}_to"])
    )

    if as_coordinates:
        edge_df[coords_col] = edge_df[geom_col].apply(lambda point: [point.x, point.y])
        edge_df[f"{coords_col}_to"] = edge_df[f"{geom_col}_to"].apply(
            lambda point: [point.x, point.y]
        )

        edge_df = edge_df.drop(columns=[GEOMETRY_COLUMN, f"{GEOMETRY_COLUMN}_to"])

    return edge_df


class ValhallaProfile(str, Enum):
    """Routing profiles supported by the Valhalla API."""

    AUTO = "auto"
    BICYCLE = "bicycle"
    MULTIMODAL = "multimodal"
    PEDESTRIAN = "pedestrian"

    def __str__(self) -> str:
        """Return the raw string value (e.g., 'auto')."""
        return self.value


def add_routes_to_edges(
    edge_df: pd.DataFrame, profile: str = ValhallaProfile.AUTO.value
) -> pd.DataFrame:
    """
    Enriches edges with routing metrics (distance and duration) using Valhalla matrix API.
    Uses the slower Valhalla directions API as a fallback for edges without a matrix route.

    :param edge_df:
        DataFrame containing at minimum `from_node`, `to_node` affiliation id columns
        and corresponding `coords` and `coords_to` coordinates expected by the helper
        functions `_get_nodes_from_edges` and `_add_node_index_to_edges`.
    :param profile:
        Valhalla profile (e.g. ValhallaProfile.AUTO.value).
    :return:
        Copy of edges with `distance_m` and `duration_s` filled.
    """

    edge_subsets = df_split(
        df=edge_df,
        chunks=int(len(edge_df) / _CHUNK_SIZE_ROUTING_MATRIX),
    )

    # initialize Valhalla client
    client = Valhalla(base_url=DEFAULT_VALHALLA_BASE_URL)

    edge_df_list = []
    for edge_subset in edge_subsets:

        nodes = _get_nodes_from_edges(edge_df=edge_subset)
        edges = _add_node_index_to_edges(edge_df=edge_subset, node_df=nodes)

        # coordinates for valhalla
        coordinates = nodes[_COORDS_COLUMN].to_list()

        # source and target indexes for valhalla
        source_index = nodes[nodes[_SOURCE_COLUMN]].index.tolist()
        target_index = nodes[~nodes[_SOURCE_COLUMN]].index.tolist()

        # make an API call
        matrix = client.matrix(
            locations=coordinates,
            profile=profile,
            preference="fastest",
            sources=source_index,  # Index of the origin
            targets=target_index,  # Indices of the destinations
        )

        # add results to the edge DataFrame
        edges[DISTANCE_M_COLUMN] = edges.apply(
            lambda x: matrix.distances[int(x[_DF_INDEX_COLUMN])][int(x[f"{_DF_INDEX_COLUMN}_to"])],
            axis=1,
        )
        edges[DURATION_S_COLUMN] = edges.apply(
            lambda x: matrix.durations[int(x[_DF_INDEX_COLUMN])][int(x[f"{_DF_INDEX_COLUMN}_to"])],
            axis=1,
        )
        edge_df_list.append(edges)

    edges = pd.concat(edge_df_list, ignore_index=True)
    edges = edges.drop(columns=[_DF_INDEX_COLUMN, f"{_DF_INDEX_COLUMN}_to"])

    _fill_unrouted_edges_with_valhalla_directions_api(
        edge_df=edges, valhalla=client, profile=profile
    )

    edges = edges.rename(
        columns={
            FROM_NODE_COLUMN: AFFILIATION_ID_FROM_COLUMN,
            TO_NODE_COLUMN: AFFILIATION_ID_TO_COLUMN,
        }
    )
    return edges


def _fill_unrouted_edges_with_valhalla_directions_api(
    edge_df: pd.DataFrame, valhalla: Valhalla, profile: str = ValhallaProfile.AUTO.value
) -> None:
    """
    Checks for empty `distance_m` and `duration_s` columns in the edge DataFrame and fills them
    using now the Valhalla directions API.
    :param edge_df: The edge DataFrame
    :param valhalla: The Valhalla API client
    """
    unrouted_edges = edge_df[
        (edge_df[DISTANCE_M_COLUMN].isnull()) | (edge_df[DURATION_S_COLUMN].isnull())
    ]
    if not unrouted_edges.empty:
        # Log a warning with the number of edges that will be routed using the slower API
        _logger.warning(
            "Routes not found for %d edge(s); falling back to the slower Valhalla directions API.",
            len(unrouted_edges),
        )
        # Fill distance and duration in the original DataFrame
        filled_df = (
            unrouted_edges[[_COORDS_COLUMN, f"{_COORDS_COLUMN}_to"]]
            .apply(
                lambda row: pd.Series(
                    _get_valhalla_metrics(
                        valhalla=valhalla,
                        coords_from=row[_COORDS_COLUMN],
                        coords_to=row[f"{_COORDS_COLUMN}_to"],
                        profile=profile,
                    )
                ),
                axis=1,
            )
            .rename(columns={0: DISTANCE_M_COLUMN, 1: DURATION_S_COLUMN})
        )
        edge_df.loc[filled_df.index, DISTANCE_M_COLUMN] = filled_df[DISTANCE_M_COLUMN].values
        edge_df.loc[filled_df.index, DURATION_S_COLUMN] = filled_df[DURATION_S_COLUMN].values


def _get_valhalla_metrics(
    valhalla: Valhalla,
    coords_from: List[float],
    coords_to: List[float],
    profile: str = ValhallaProfile.AUTO.value,
) -> Tuple[int, int]:
    """
    Call Valhalla directions API and extract distance/duration.

    :param valhalla: The valhalla client
    :param coords_from: The lat lon source coordinates
    :param coords_to: The lat lon target coordinates
    :param profile: The valhalla profile (e.g. ValhallaProfile.AUTO.value).

    :return: A tuple of (distance, duration).
    """
    result = valhalla.directions(
        locations=[coords_from, coords_to],
        profile=profile,
    )
    return result.distance, result.duration


def _get_nodes_from_edges(edge_df: pd.DataFrame) -> pd.DataFrame:
    # column aliases
    from_col = FROM_NODE_COLUMN
    to_col = TO_NODE_COLUMN
    affiliation_id_col = AFFILIATION_ID_COLUMN
    coords_col = _COORDS_COLUMN

    # from nodes (source is set to True)
    from_df = (
        edge_df[[from_col, coords_col]]
        .rename(columns={from_col: affiliation_id_col})
        .drop_duplicates(subset=affiliation_id_col, ignore_index=True)
    )
    from_df[_SOURCE_COLUMN] = True

    # to nodes (source is set to False)
    to_df = (
        edge_df[[to_col, f"{coords_col}_to"]]
        .rename(columns={f"{coords_col}_to": coords_col, to_col: affiliation_id_col})
        .drop_duplicates(subset=affiliation_id_col, ignore_index=True)
    )
    to_df[_SOURCE_COLUMN] = False

    # merge and add node index
    nodes = pd.concat([from_df, to_df], ignore_index=True)
    nodes[_DF_INDEX_COLUMN] = nodes.index.values
    return nodes


def _add_node_index_to_edges(edge_df: pd.DataFrame, node_df: pd.DataFrame) -> pd.DataFrame:
    edge_df = (
        edge_df.merge(
            right=node_df[node_df[_SOURCE_COLUMN]][[AFFILIATION_ID_COLUMN, _DF_INDEX_COLUMN]],
            left_on=FROM_NODE_COLUMN,
            right_on=AFFILIATION_ID_COLUMN,
            how="left",
            suffixes=("", "_x"),
        )
        .merge(
            right=node_df[~node_df[_SOURCE_COLUMN]][[AFFILIATION_ID_COLUMN, _DF_INDEX_COLUMN]],
            left_on=TO_NODE_COLUMN,
            right_on=AFFILIATION_ID_COLUMN,
            how="left",
            suffixes=("", "_y"),
        )
        .drop(columns=[AFFILIATION_ID_COLUMN, f"{AFFILIATION_ID_COLUMN}_y"])
        .rename(columns={f"{_DF_INDEX_COLUMN}_y": f"{_DF_INDEX_COLUMN}_to"})
    )
    return edge_df
