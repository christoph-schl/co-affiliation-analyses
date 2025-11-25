from dataclasses import dataclass

import geopandas as gpd
from networkx import Graph


@dataclass
class AffiliationGraph:
    """
    Dataclass contains edges as geopandas GeoDataFrame and as weighted networkx graph
    """

    edge_gdf: gpd.GeoDataFrame
    graph: Graph
