from typing import Dict, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import pycountry

import src.mma.constants as constants

_ARTICLE_ID_COLUMN = "article_id"
_ARTICLE_AUTHOR_ID_COLUMN = "author_ids"
_AUTHOR_SEPARATOR = ";"

ISO3_CODES = [country.alpha_3 for country in pycountry.countries]


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
    affiliation_gdf: gpd.GeoDataFrame,
) -> Optional[Dict[np.int64, np.int64]]:
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
        return None

    # Filter rows with a valid parent
    df_parent = affiliation_gdf.loc[affiliation_gdf[parent_col].notna(), [id_col, parent_col]]

    if df_parent.empty:
        return None

    # Convert parent IDs to integer safely
    df_parent[parent_col] = df_parent[parent_col].astype("int64")

    # Build mapping
    return df_parent.set_index(id_col)[parent_col].to_dict()


def validate_country_code(code: str) -> str:
    """
    Validate that a given ISO-3 country code exists.

    :param code: The ISO-3 country code to validate (e.g., 'USA', 'FRA').
    :return: The uppercase valid ISO-3 country code.
    :raises ValueError: If the provided code is not a valid ISO-3 country code.
    """
    code = code.strip().upper()
    if code not in ISO3_CODES:
        raise ValueError(
            f"Invalid country code '{code}'. " f"Must be one of {len(ISO3_CODES)} valid ISO3 codes."
        )
    return code


def filter_links_by_country(link_gdf: gpd.GeoDataFrame, country_filter: str) -> gpd.GeoDataFrame:
    """
    Filters links by ISO3 country code.
    :param link_gdf: The link GeoDataFrame.
    :param country_filter: The given filter
    :return: The filtered GeoDataFrame.
    """

    link_gdf = link_gdf[
        (link_gdf[constants.ISO3_COUNTRY_CODE_COLUMN] == country_filter)
        & (link_gdf[constants.ISO3_COUNTRY_CODE_COLUMN] == country_filter)
    ].reset_index(drop=True)
    return link_gdf
