from dataclasses import dataclass

import geopandas as gpd


@dataclass
class Impact:
    link_gdf: gpd.GeoDataFrame

    def __post_init__(self) -> None:
        print()
