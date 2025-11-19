from typing import Dict, Iterable, List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pycountry
import structlog

import maa.constants.constants as constants
from maa.utils.wrappers import get_execution_time

_logger = structlog.getLogger(__name__)

_ARTICLE_ID_COLUMN = "article_id"
_ARTICLE_AUTHOR_ID_COLUMN = "author_ids"
_AUTHOR_SEPARATOR = ";"
_TO_PREFIX = "to_"
_TO_SUFFIX = "_to"
_FROM_PREFIX = "from_"
_FROM_SUFFIX = "_from"

ISO3_CODES = [country.alpha_3 for country in pycountry.countries]


@get_execution_time
def get_articles_per_author(article_df: pd.DataFrame) -> pd.DataFrame:
    """
    Explodes the dataframe to get the articles for each author.
    :param article_df: The article DataFrame.
    :return: The DataFrame contains the articles for each author.
    """

    article_df = article_df.copy()
    if constants.AFFILIATION_ID_COLUMN in article_df:
        article_df = article_df.rename(
            columns={constants.AFFILIATION_ID_COLUMN: constants.ARTICLE_AFFILIATION_ID_COLUMN}
        )

    article_df[_ARTICLE_ID_COLUMN] = np.arange(len(article_df))

    explode_columns = [constants.ARTICLE_AFFILIATION_ID_COLUMN]
    if _ARTICLE_AUTHOR_ID_COLUMN in article_df.columns:
        explode_columns.append(_ARTICLE_AUTHOR_ID_COLUMN)

    for column in explode_columns:
        article_df.loc[:, column] = article_df[column].str.split(_AUTHOR_SEPARATOR)
    article_df = article_df.explode(column=explode_columns, ignore_index=True)
    article_df = article_df[article_df[constants.ARTICLE_AFFILIATION_ID_COLUMN] != ""]
    article_df = article_df[
        article_df[constants.ARTICLE_AFFILIATION_ID_COLUMN].notnull()
    ].reset_index(drop=True)

    return article_df


def get_affiliation_id_map(
    affiliation_gdf: Union[gpd.GeoDataFrame, pd.DataFrame],
) -> Dict[np.int64, np.int64]:
    """
    Creates a dictionary that maps each affiliation ID to the ID of its
    parent organization. Only rows where the parent affiliation ID is
    not null are included in the mapping.
    :param affiliation_gdf: The affiliation GeoDataFrame with affiliations.
    :return: A dictionary where the key is the affiliation ID and the value
             is the parent affiliation ID. Returns None if no valid mapping
             can be created. Only non-null parent relationships are included.
    """

    parent_col = constants.AFFILIATION_ID_PARENT_COLUMN
    id_col = constants.AFFILIATION_ID_COLUMN

    if parent_col not in affiliation_gdf.columns:
        return {}

    # Filter rows with a valid parent
    df_parent = affiliation_gdf.loc[affiliation_gdf[parent_col].notna(), [id_col, parent_col]]

    if df_parent.empty:
        return {}

    # Convert parent IDs to integer safely
    df_parent[parent_col] = df_parent[parent_col].astype("int64")

    # Build mapping
    return df_parent.set_index(id_col)[parent_col].to_dict()


def validate_country_code(code: Optional[str] = None) -> None:
    """
    Validate that a given ISO-3 country code exists.

    :param code: The ISO-3 country code to validate (e.g., 'USA', 'FRA').
    :return: The uppercase valid ISO-3 country code.
    :raises ValueError: If the provided code is not a valid ISO-3 country code.
    """

    if code is not None:
        code = code.strip().upper()
        if code not in ISO3_CODES:
            raise ValueError(
                f"Invalid country code '{code}'. "
                f"Must be one of {len(ISO3_CODES)} valid ISO3 "
                f"codes."
            )


def filter_links_by_country(link_gdf: gpd.GeoDataFrame, country_filter: str) -> gpd.GeoDataFrame:
    """
    Filters links by ISO3 country code.
    :param link_gdf: The link GeoDataFrame.
    :param country_filter: The given filter
    :return: The filtered GeoDataFrame.
    """

    link_gdf = link_gdf[
        (link_gdf[constants.ISO3_COUNTRY_CODE_COLUMN] == country_filter)
        & (link_gdf[f"{constants.ISO3_COUNTRY_CODE_COLUMN}_to"] == country_filter)
    ].reset_index(drop=True)
    return link_gdf


def filter_organization_types(
    df: Union[pd.DataFrame, gpd.GeoDataFrame], org_types: List[str]
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Keep only rows where BOTH organization type columns (`org_typ` and `org_type_to`) are in the
    provided list.
    :param df: The edge or link Geo- (DataFrame).
    :param org_types: The given list of organization types.
    :return: The filtered GeoDataFrame.
    """

    mask = (df[constants.ORGANISATION_TYPE_COLUMN].isin(org_types)) & (
        df[f"{constants.ORGANISATION_TYPE_COLUMN}_to"].isin(org_types)
    )
    filtered_gdf = df[mask].copy()

    kept_count = len(filtered_gdf)
    total_count = len(df)
    _logger.info(
        f"Kept {kept_count}/{total_count} edges where both organization types are"
        f" in {org_types}"
    )

    return filtered_gdf


def _select_by_affix(
    columns: Iterable[str], prefixes: Iterable[str] = (), suffixes: Iterable[str] = ()
) -> List[str]:
    """
    Return columns that start with any prefix or end with any suffix (preserve original order).
    """
    prefixes = tuple(prefixes)
    suffixes = tuple(suffixes)
    return [
        col
        for col in columns
        if any(col.startswith(p) for p in prefixes) or any(col.endswith(s) for s in suffixes)
    ]


def _strip_affixes(name: str, prefixes: Iterable[str] = (), suffixes: Iterable[str] = ()) -> str:
    """Remove the first matching prefix and/or suffix from a column name."""
    for p in prefixes:
        if name.startswith(p):
            name = name[len(p) :]  # noqa: E203
            break
    for s in suffixes:
        if name.endswith(s):
            name = name[: -len(s)]
            break
    return name


# build rename mappings (strip prefixes/suffixes)
def _make_rename_map(
    columns: Iterable[str], prefixes: Iterable[str], suffixes: Iterable[str]
) -> Dict[str, str]:
    return {col: _strip_affixes(col, prefixes=prefixes, suffixes=suffixes) for col in columns}


def get_link_nodes(link_gdf: Union[gpd.GeoDataFrame, pd.DataFrame]) -> pd.DataFrame:
    """
    Convert a 'link' GeoDataFrame that encodes 'from' and 'to' node columns into a stacked node
    DataFrame.

    :param link_gdf: The link GeoDataFrame.
    :return: The DataFrame with extracted 'from' and 'to' nodes.
    """
    cols = list(link_gdf.columns)

    # find to/from columns (preserve order)
    to_prefix_cols = _select_by_affix(cols, prefixes=[_TO_PREFIX])
    to_suffix_cols = _select_by_affix(cols, suffixes=[_TO_SUFFIX])
    to_columns = to_prefix_cols + [c for c in to_suffix_cols if c not in to_prefix_cols]

    from_columns = _select_by_affix(cols, prefixes=[_FROM_PREFIX], suffixes=[_FROM_SUFFIX])
    # add stripped versions of the to-suffix columns (original behavior)
    stripped_to_suffix_cols = [c.replace(_TO_SUFFIX, "") for c in to_suffix_cols]
    from_columns += [c for c in stripped_to_suffix_cols if c not in from_columns]

    # used columns set (for determining common columns)
    used = set(from_columns + to_columns)

    # columns that are neither from- nor to-specific -> common to both
    common_columns = [c for c in cols if c not in used]

    # final ordered column lists for each side
    from_cols_final = from_columns + common_columns
    to_cols_final = to_columns + common_columns

    # slice dataframes
    from_df = link_gdf.loc[:, from_cols_final].copy()
    to_df = link_gdf.loc[:, to_cols_final].copy()

    from_rename = _make_rename_map(
        columns=from_df.columns, prefixes=[_FROM_PREFIX], suffixes=[_FROM_SUFFIX]
    )
    to_rename = _make_rename_map(
        columns=to_df.columns, prefixes=[_TO_PREFIX], suffixes=[_TO_SUFFIX]
    )

    from_df = from_df.rename(columns=from_rename)
    to_df = to_df.rename(columns=to_rename)

    # concat, drop geometry, sort and reset index (keeps original behavior)
    node_df = pd.concat([from_df, to_df], ignore_index=True)

    if constants.GEOMETRY_COLUMN in node_df.columns:
        node_df = node_df.drop(columns=[constants.GEOMETRY_COLUMN])

    sort_columns = [constants.EID_COLUMN, constants.ARTICLE_AUTHOR_ID_COLUMN]
    if constants.ARTICLE_AFFILIATION_INDEX_COLUMN in node_df.columns:
        sort_columns.append(constants.ARTICLE_AFFILIATION_INDEX_COLUMN)

    node_df = node_df.sort_values(by=sort_columns).reset_index(drop=True)

    return node_df
