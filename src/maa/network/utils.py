from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import structlog
from networkx import Graph

from maa.constants.constants import (
    AFFILIATION_EDGE_COUNT_COLUMN,
    AFFILIATION_ID_COLUMN,
    ARTICLE_AFFILIATION_COUNT_COLUMN,
    ARTICLE_AFFILIATION_ID_COLUMN,
    ARTICLE_AFFILIATION_INDEX_COLUMN,
    ARTICLE_AUTHOR_ID_COLUMN,
    ARTICLE_COUNT_COLUMN,
    COVER_DATE_COLUMN,
    FROM_AFFILIATION_INDEX_COLUMN,
    FROM_NODE_COLUMN,
    GEOMETRY_COLUMN,
    ORG_TYPE_COLORS,
    ORGANISATION_TYPE_COLUMN,
    PREFERRED_AFFILIATION_NAME_COLUMN,
    TO_AFFILIATION_INDEX_COLUMN,
    TO_NODE_COLUMN,
    WGS84_EPSG,
)
from maa.utils.geo_tools import create_line_geometries
from maa.utils.utils import filter_links_by_country
from maa.utils.wrappers import get_execution_time, parallelize_dataframe

_logger = structlog.getLogger()

_EDGE_COLUMN = "edge"
_NO_DATA_STRING = "nodata"
_CHUNK_SIZE_PARALLEL_PROCESSING = 2000
_YEAR_DIFFERENCE_COLUMN = "year_diff"
_COVER_YEAR_COLUMN = "cover_year"
_CLUSTER_COLOR_COLUMN = "cluster"


def get_combination_list(input_list: List[int]) -> List[Tuple[int, int]]:
    combination_list = [(min(a, b), max(a, b)) for a, b in combinations(input_list, 2)]
    return list(np.asarray(combination_list, dtype=np.int64))


def _generate_edges(row: pd.Series) -> List[Tuple[int, int]]:
    nodes = list(map(np.int64, row.split("-")))  # Convert to int64
    return get_combination_list(input_list=nodes)


def _apply_affiliation_idx_mapping(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Get affiliation indices for the given affiliation edge combinations by applying a mapping
    dictionary.

    The mapping is stored in the `affiliation_idx` column, and the edge combination column
    is called `edge`.

    :param df: Input DataFrame containing at least the `edge_col` and `mapping_col` columns.
    :return: A pandas Series of lists of tuples, where each tuple contains the mapped
             affiliation indices.
    """

    edge_col = _EDGE_COLUMN
    mapping_col = ARTICLE_AFFILIATION_INDEX_COLUMN

    return df.apply(
        lambda row: [(row[mapping_col][i], row[mapping_col][j]) for i, j in row[edge_col]], axis=1
    )


def _filter_multi_affiliation(article_author_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters articles with multi-affiliated authors.
    :param article_author_df: The DataFrame with article and author information
    :return: The filtered DataFrame with multi-affiliated authors
    """
    df = article_author_df.copy()
    # count affiliations per article and keep only articles with > 1 affiliation
    df[ARTICLE_AFFILIATION_COUNT_COLUMN] = df[ARTICLE_AFFILIATION_ID_COLUMN].apply(len)
    multi_aff_mask = df[ARTICLE_AFFILIATION_COUNT_COLUMN] > 1
    if not multi_aff_mask.any():
        return pd.DataFrame({FROM_NODE_COLUMN: [], TO_NODE_COLUMN: []})
    df = df.loc[multi_aff_mask].reset_index(drop=True)
    return df


def _create_link_df(article_author_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates all possible unique link combinations between affiliations stored in the `author_afids`
    column.

    For example, if an author has affiliations `[1, 2, 3]`, links `(1, 2)`, `(1, 3)`, and `(2, 3)`
    will be created.

    :param article_author_df: The DataFrame containing article, author, and affiliation information.
                              Must include the column `author_afids` with a list of affiliation IDs
                              per author.
    :return: A DataFrame containing one row per unique affiliation link combination.
    """

    link_df = article_author_df.copy()

    # create edge combinations and the affiliation-index mapping
    link_df[_EDGE_COLUMN] = link_df[ARTICLE_AFFILIATION_ID_COLUMN].apply(get_combination_list)
    link_df[ARTICLE_AFFILIATION_INDEX_COLUMN] = _apply_affiliation_idx_mapping(df=link_df)
    link_df = link_df.explode(
        column=[_EDGE_COLUMN, ARTICLE_AFFILIATION_INDEX_COLUMN], ignore_index=True
    )
    link_df = link_df.drop(columns=[ARTICLE_AFFILIATION_ID_COLUMN])

    if not link_df.empty:
        link_df[[FROM_NODE_COLUMN, TO_NODE_COLUMN]] = pd.DataFrame(
            link_df[_EDGE_COLUMN].tolist(), index=link_df.index
        )
        link_df[[FROM_AFFILIATION_INDEX_COLUMN, TO_AFFILIATION_INDEX_COLUMN]] = pd.DataFrame(
            link_df[ARTICLE_AFFILIATION_INDEX_COLUMN].tolist(), index=link_df.index
        )
        link_df = link_df.drop(columns=[_EDGE_COLUMN, ARTICLE_AFFILIATION_INDEX_COLUMN])
    else:
        link_df = pd.DataFrame({FROM_NODE_COLUMN: [], TO_NODE_COLUMN: []})

    return link_df


def _create_link_gdf(link_df: pd.DataFrame, affiliation_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Creates a GeoDataFrame of links by merging affiliation attributes, including geometry,
    to the link DataFrame.

    Each row in the resulting GeoDataFrame represents a link between two affiliations with a
    line geometry.

    :param affiliation_gdf: The GeoDataFrame containing affiliation information, including geometry.
    :param link_df: The DataFrame containing links between affiliations.
                    Each row represents a connection defined by the `from_node` and `to_node`
                    columns, which correspond to affiliation IDs.
    :param verbose: If True, prints warnings for any affiliations with missing or invalid geometry.
    :return: A GeoDataFrame representing links between affiliations, with line geometries for each
             link.
    """
    if link_df.empty:
        _logger.warning("Input link DataFrame is empty. No geometries to process.")
        return link_df

    link_df = pd.merge(
        left=link_df,
        right=affiliation_gdf,
        left_on=FROM_NODE_COLUMN,
        right_on=AFFILIATION_ID_COLUMN,
        how="left",
    )
    link_df = pd.merge(
        left=link_df,
        right=affiliation_gdf,
        left_on=TO_NODE_COLUMN,
        right_on=AFFILIATION_ID_COLUMN,
        how="left",
        suffixes=("", "_to"),
    ).reset_index(drop=True)

    geom_cols = [GEOMETRY_COLUMN, f"{GEOMETRY_COLUMN}_to"]

    n_links = len(link_df)
    link_df = parallelize_dataframe(
        input_function=create_line_geometries,
        df=link_df,
        geometry_columns=geom_cols,
        drop_invalid_geoms=True,
        chunk_size=_CHUNK_SIZE_PARALLEL_PROCESSING,
    )
    n_dropped = n_links - len(link_df)
    if n_dropped > 0:
        _logger.warning(
            f"Dropped {n_dropped} affiliation link(s) from the DataFrame with no " f"coordinates."
        )

    link_df = link_df.drop(
        columns=[f"{GEOMETRY_COLUMN}_to", AFFILIATION_ID_COLUMN, f"{AFFILIATION_ID_COLUMN}_to"],
        axis=1,
    )
    link_gdf = gpd.GeoDataFrame(link_df, crs=WGS84_EPSG)
    return link_gdf


@get_execution_time
def create_affiliation_links(
    article_author_df: pd.DataFrame,
    affiliation_gdf: gpd.GeoDataFrame,
    country_filter: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Creates a GeoDataFrame with all unique link combinations for authors with multiple
    affiliations.

    For each author in `self.article_df`, all possible pairwise links between affiliations
    stored in the `author_afids` column are generated. For example, if an author has
    affiliations `[1, 2, 3]`, the resulting links will be `(1, 2)`, `(1, 3)`, and `(2, 3)`.
    Affiliation information, including geometry, is merged into the link DataFrame, resulting in
    a GeoDataFrame with line geometries representing each link.

    :return: A GeoDataFrame containing links between affiliations, with a line geometry for each
             link.
    """

    multi_affiliation_df = _filter_multi_affiliation(article_author_df=article_author_df)
    link_df = _create_link_df(article_author_df=multi_affiliation_df)
    link_gdf = _create_link_gdf(affiliation_gdf=affiliation_gdf, link_df=link_df)

    if country_filter is not None:
        link_gdf = filter_links_by_country(link_gdf=link_gdf, country_filter=country_filter)

    return link_gdf


def _create_graph_from_edges(
    edge_gdf: gpd.GeoDataFrame, from_node_column: str, to_node_column: str, weight_column: str
) -> Graph:
    """
    Create a weighted NetworkX graph from an edge GeoDataFrame.
    :param edge_gdf: GeoDataFrame containing edge data.
    :param from_node_column: Column name for the source node.
    :param to_node_column: Column name for the target node.
    :param weight_column: Column name for the edge weight.
    :return: A NetworkX graph with weighted edges.
    """

    G = nx.Graph()

    edge_list = list(
        zip(
            edge_gdf[from_node_column],
            edge_gdf[to_node_column],
            edge_gdf[weight_column],
        )
    )
    G.add_weighted_edges_from(edge_list)

    return G


def compute_edge_strengths(link_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compute the strength of links between nodes by counting
    the number of edges between each (from_node, to_node) pair.
    :param link_gdf: GeoDataFrame containing at least `from_node` and `to_node` column.
    :return: GeoDataFrame with one edge per (from_node, to_node) pair and the new column
            `affiliation_edge_count` representing edge strength.
    """

    edge_gdf = link_gdf.copy()

    edge_gdf[AFFILIATION_EDGE_COUNT_COLUMN] = edge_gdf.groupby(
        [FROM_NODE_COLUMN, TO_NODE_COLUMN]
    ).transform("size")
    edge_gdf = edge_gdf.drop_duplicates(subset=[FROM_NODE_COLUMN, TO_NODE_COLUMN])
    return edge_gdf


@dataclass
class Edge:
    """
    Dataclass contains edges as geopandas GeoDataFrame and as weighted networkx graph
    """

    gdf: gpd.GeoDataFrame
    graph: Graph


@get_execution_time
def create_graph_from_links(
    link_gdf: gpd.GeoDataFrame,
    min_weight: int = 0,
) -> Edge:
    """
    Creates an affiliation network graph from a GeoDataFrame of links between nodes (affiliations).

    This function aggregates link data between pairs of nodes (`from_node`, `to_node`), computes the
    connection strength for each (source, target) pair, and returns a weighted NetworkX graph.
    Optionally, weak connections (below a given weight threshold) can be filtered out.

    :param link_gdf: The GeoDataFrame containing link data.
    :param min_weight: The minimum link strength to retain. Edges with lower weights are removed
                       from the graph. If ``None``, all edges are included.

    :return: A pandas DataFrame and a weighted, undirected NetworkX graph where:

        - **Nodes** represent affiliations.
        - **Edges** connect affiliations based on authors with multiple affiliations.
        - **Edge weights** correspond to the strength or frequency of those links.
    """

    # node columns
    from_col = PREFERRED_AFFILIATION_NAME_COLUMN
    to_col = f"{PREFERRED_AFFILIATION_NAME_COLUMN}_to"

    link_gdf = link_gdf.copy()
    link_gdf = link_gdf.drop(columns=AFFILIATION_ID_COLUMN, errors="ignore")

    # compute aggregated edges
    edge_gdf = compute_edge_strengths(link_gdf=link_gdf)

    # apply minimum weight filter (if requested)
    if min_weight is not None:
        edge_gdf = edge_gdf[edge_gdf[AFFILIATION_EDGE_COUNT_COLUMN] >= min_weight]

    # replace missing node names with a sentinel
    edge_gdf[from_col] = edge_gdf[from_col].fillna(_NO_DATA_STRING)
    edge_gdf[to_col] = edge_gdf[to_col].fillna(_NO_DATA_STRING)

    affiliation_graph = _create_graph_from_edges(
        edge_gdf=edge_gdf,
        from_node_column=FROM_NODE_COLUMN,
        to_node_column=TO_NODE_COLUMN,
        weight_column=AFFILIATION_EDGE_COUNT_COLUMN,
    )

    edge_gdf = _add_article_count_to_edges(affiliation_graph=affiliation_graph, edge_gdf=edge_gdf)
    edge_output = Edge(gdf=edge_gdf, graph=affiliation_graph)

    return edge_output


def _add_article_count_to_edges(
    affiliation_graph: Graph, edge_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    weight_sums = {
        node: sum(data["weight"] for _, _, data in affiliation_graph.edges(node, data=True))
        for node in affiliation_graph.nodes()
    }
    article_count_df = pd.DataFrame(
        list(weight_sums.items()), columns=[AFFILIATION_ID_COLUMN, ARTICLE_COUNT_COLUMN]
    )
    edge_gdf = (
        edge_gdf.merge(
            right=article_count_df,
            left_on=FROM_NODE_COLUMN,
            right_on=AFFILIATION_ID_COLUMN,
            how="left",
        )
        .merge(
            right=article_count_df,
            left_on=TO_NODE_COLUMN,
            right_on=AFFILIATION_ID_COLUMN,
            how="left",
            suffixes=("", "_to"),
        )
        .drop(columns=[AFFILIATION_ID_COLUMN, f"{AFFILIATION_ID_COLUMN}_to"], errors="ignore")
    )

    return edge_gdf


@get_execution_time
def retain_affiliation_links_with_min_year_gap(
    link_gdf: gpd.GeoDataFrame, min_year_gap: int
) -> gpd.GeoDataFrame:
    """
    Keep only affiliation links where the same author (for the same from_node/to_node pair)
    has publications separated by at least `min_year_gap` years.

    :param link_gdf:
                GeoDataFrame with at least the columns named by `cover_darte`, `author_ids`,
                `from_node` and `to_node`.
    :param min_year_gap:
                Minimum required difference (in years) between earliest and latest publication
                years for an author at the same affiliation pair. If < 1, the original DataFrame
                is returned (copied).
    :return:
                Filtered GeoDataFrame (a copy).
    """

    # nothing to filter
    if min_year_gap < 1:
        return link_gdf.copy()

    df = link_gdf.copy()

    # extract year (coerce invalid dates to NaT -> will be dropped)
    df[_COVER_YEAR_COLUMN] = pd.to_datetime(df[COVER_DATE_COLUMN], errors="coerce").dt.year

    # drop rows with missing/invalid cover dates (can't compare years otherwise)
    if df[_COVER_YEAR_COLUMN].isna().any():
        invalid_count = df[_COVER_YEAR_COLUMN].isna().sum()
        _logger.warning(
            "Dropping %d link(s) because %s could not be parsed as a year",
            invalid_count,
            COVER_DATE_COLUMN,
        )
        df = df.dropna(subset=[_COVER_YEAR_COLUMN])

    # compute year-span for each (author, from_node, to_node) group
    df[_YEAR_DIFFERENCE_COLUMN] = df.groupby(
        [ARTICLE_AUTHOR_ID_COLUMN, FROM_NODE_COLUMN, TO_NODE_COLUMN]
    )[_COVER_YEAR_COLUMN].transform(lambda years: int(years.max() - years.min()))

    before = len(link_gdf)
    df = df[df[_YEAR_DIFFERENCE_COLUMN] >= int(min_year_gap)].drop(columns=[_COVER_YEAR_COLUMN])
    after = len(df)

    _logger.info(
        f"Dropped {before - after} affiliation link(s) (kept {after}) "
        f"with min_year_gap={min_year_gap}",
    )

    return df


def get_vos_cluster_numbers(affiliation_graph: Graph) -> pd.DataFrame:
    """
    Creates a DataFrame with RGB colors for each organisation type
    for an organisation type.

    :param affiliation_graph: The affiliation graph network
    :return: The VOS color map DataFrame
    """

    # extract unique organization types
    node_vals = affiliation_graph.nodes(data=ORGANISATION_TYPE_COLUMN)
    unique_vals = {val for _, val in node_vals}

    # assign cluster numbers starting at 1
    cluster_map = dict(zip(unique_vals, range(1, len(unique_vals) + 1), strict=True))

    # map cluster numbers to colors
    color_map = {number: ORG_TYPE_COLORS[org_type] for org_type, number in cluster_map.items()}

    # build the DataFrame
    color_df = pd.DataFrame.from_dict(color_map, orient="index")
    color_df.index.name = _CLUSTER_COLOR_COLUMN

    return color_df
