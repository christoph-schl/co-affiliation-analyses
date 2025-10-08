from pathlib import Path
from typing import Optional, Any

import geopandas as gpd
import numpy as np
import pandera.pandas as pa
from pandera.typing import Series
from pandera.typing.geopandas import GeoDataFrame, GeoSeries


class AffiliationSchema(pa.DataFrameModel):
    """Model for pandas affiliation DataFrame."""

    affiliation_id: Series[np.int64] = pa.Field(nullable=False)
    name: Optional[Series[str]] = pa.Field(nullable=True)
    preferred_name: Series[str] = pa.Field(nullable=True)
    latitude: Optional[Series[float]] = pa.Field(nullable=True)
    longitude: Optional[Series[float]] = pa.Field(nullable=True)
    type: Optional[Series[str]] = pa.Field(nullable=True)
    parent: Optional[Series[str]] = pa.Field(nullable=True)
    name_variants: Optional[Series[Any]] = pa.Field(nullable=True)
    address_part: Optional[Series[str]] = pa.Field(nullable=True)
    city: Optional[Series[str]] = pa.Field(nullable=True)
    org_type: Series[str] = pa.Field(nullable=True)
    affiliation_id_parent: Optional[Series[Any]] = pa.Field(nullable=True)
    parent_preferred_name: Optional[Series[str]] = pa.Field(nullable=True)
    umbrella_id: Optional[Series[float]] = pa.Field(nullable=True)
    iso3_code: Optional[Series[str]] = pa.Field(nullable=True)
    country: Optional[Series[str]] = pa.Field(nullable=True)
    geometry: GeoSeries = pa.Field(nullable=True)


@pa.check_types(lazy=True)
def read_affiliations(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    gdf = GeoDataFrame[AffiliationSchema](gdf)
    return gdf
