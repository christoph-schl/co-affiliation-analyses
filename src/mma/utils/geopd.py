import enum
from typing import Tuple

import geopandas as gpd
import numpy as np
import numpy.typing as npt


class SupportedOverlayPredicates(enum.Enum):
    NONE = None
    INTERSECTS = "intersects"
    WITHIN = "within"
    CONTAINS = "contains"
    CONTAINS_PROPERLY = "contains_properly"
    COVERED_BY = "covered_by"
    COVERS = "covers"
    CROSSES = "crosses"
    OVERLAPS = "overlaps"
    TOUCHES = "touches"


def get_gdf_overlay_index_based_on_predicate(
    to_overlay_gdf: gpd.GeoDataFrame,
    overlay_gdf: gpd.GeoDataFrame,
    predicate: SupportedOverlayPredicates,
    inverse: bool = False,
    unique: bool = True,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """
    Returns index values of the 'to_overlay_gdf' and the 'overlay_gdf' GeoDataFrame for geometries
    for which the given predicate (e.g. 'intersects') is fulfilled

        :param to_overlay_gdf:
            GeoDataFrame that should be overlapped with. i.e. a vector data vector_mask
        :param overlay_gdf:
            GeoDataFrame to use as overlay (clipping vector_mask)
        :param predicate:
            If predicate is provided, the input geometries are tested using the predicate function
        :param inverse:
            Whether to inverse the predicate (keep all inside, or outside). The inverse
            to_overlay_gdf Index is returned
        :param unique:
            Whether unique index values are returned

        :return:
            Index values of the overlay and to_overlay GeoDataFrame with which the predicate
             applies between the 'overlay_gdf' and the `to_overlay_gdf`
    """

    overlay_gdf_idx, to_overlay_gdf_idx = to_overlay_gdf.sindex.query(
        overlay_gdf.geometry, predicate=predicate.value
    )
    if unique:
        overlay_gdf_index = overlay_gdf.index.values[np.unique(overlay_gdf_idx)]
        to_overlay_gdf_index = to_overlay_gdf.index.values[np.unique(to_overlay_gdf_idx)]
    else:
        overlay_gdf_index = overlay_gdf.index.values[overlay_gdf_idx]
        to_overlay_gdf_index = to_overlay_gdf.index.values[to_overlay_gdf_idx]

    if inverse:
        inverse_to_overlay_gdf_index = to_overlay_gdf.index.values[
            ~np.isin(to_overlay_gdf.index.values, to_overlay_gdf_index)
        ]
        return inverse_to_overlay_gdf_index, overlay_gdf_index

    return to_overlay_gdf_index, overlay_gdf_index
