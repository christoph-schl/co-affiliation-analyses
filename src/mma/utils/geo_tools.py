from typing import List, Tuple, Union

import geopandas as gpd
import pandas as pd
import structlog
from geopandas import GeoDataFrame
from shapely import LineString

from src.mma.constants import GEOMETRY_COLUMN

_logger = structlog.getLogger()
_TEMPORARY_GEOMETRY_COLUMN = "temp_geom"


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
