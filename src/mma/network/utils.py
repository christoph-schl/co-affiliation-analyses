from itertools import combinations
from typing import List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import structlog

from src.mma.constants import (
    AFFILIATION_ID_COLUMN,
    ARTICLE_AFFILIATION_COUNT_COLUMN,
    ARTICLE_AFFILIATION_ID_COLUMN,
    ARTICLE_AFFILIATION_INDEX_COLUMN,
    FROM_AFFILIATION_INDEX_COLUMN,
    FROM_NODE_COLUMN,
    GEOMETRY_COLUMN,
    TO_AFFILIATION_INDEX_COLUMN,
    TO_NODE_COLUMN,
    WGS84_EPSG,
)
from src.mma.utils.geo_tools import create_line_geom

_logger = structlog.getLogger()

_EDGE_COLUMN = "edge"


def _get_combination_list(input_list: List[int]) -> List[Tuple[int, int]]:
    combination_list = [(min(a, b), max(a, b)) for a, b in combinations(input_list, 2)]
    return list(np.asarray(combination_list, dtype=np.int64))


def _generate_edges(row: pd.Series):
    nodes = list(map(np.int64, row.split("-")))  # Convert to int64
    return _get_combination_list(input_list=nodes)


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
    link_df[_EDGE_COLUMN] = link_df[ARTICLE_AFFILIATION_ID_COLUMN].apply(_get_combination_list)
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


def _create_link_gdf(
    link_df: pd.DataFrame, affiliation_gdf: gpd.GeoDataFrame, verbose: bool = True
) -> gpd.GeoDataFrame:
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
    if not link_df.empty:
        link_df[GEOMETRY_COLUMN] = link_df.apply(
            lambda x: create_line_geom(point_a=x.geometry, point_b=x.geometry_to), axis=1
        )

        # Count how many rows will be dropped (those with missing geometry)
        n_dropped = link_df[GEOMETRY_COLUMN].isna().sum()

        link_df = link_df.dropna(subset=[GEOMETRY_COLUMN]).reset_index(drop=True)
        if verbose:
            if n_dropped > 0:
                _logger.warning(f"Dropped {n_dropped} affiliation links(s) with no coordinates.")

        link_df = link_df.drop(columns=[f"{GEOMETRY_COLUMN}_to"], axis=1)
        link_df = gpd.GeoDataFrame(link_df, crs=WGS84_EPSG)
    return link_df


def create_affiliation_links(
    article_author_df: pd.DataFrame, affiliation_gdf: gpd.GeoDataFrame, verbose: bool = True
) -> gpd.GeoDataFrame:

    multi_affiliation_df = _filter_multi_affiliation(article_author_df=article_author_df)
    link_df = _create_link_df(article_author_df=multi_affiliation_df)
    link_gdf = _create_link_gdf(affiliation_gdf=affiliation_gdf, link_df=link_df, verbose=verbose)

    return link_gdf
