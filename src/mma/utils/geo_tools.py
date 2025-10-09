from typing import Optional, Tuple

from geopandas import GeoDataFrame
from shapely import LineString, Point


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


def create_line_geom(point_a: Point, point_b: Point) -> Optional[LineString]:
    if point_a is None or point_b is None:
        return None

    if point_a.is_empty or point_b.is_empty:
        return None

    line = LineString([point_a, point_b])
    return line
