from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.mma.network.network import AffiliationNetworkProcessor
from src.mma.utils.wrappers import get_execution_time
from src.mma.znib.utils import create_znib_model_input


@dataclass
class ZNIB(AffiliationNetworkProcessor):

    pass

    @get_execution_time
    def get_znib_model_input(
        self, route_df: pd.DataFrame, min_edge_weight: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Creates the base edge-level DataFrame if not cached and the input DataFrame for the
        ZNIB (Zero-Inflated Negative Binomial) model.

        This function prepares and enriches edge-level data to serve as input for
        a ZNIB gravity model of multi-affiliation collaboration intensity.
        It performs the following steps:

          1. Generates all affiliation-to-affiliation edge combinations.
          2. Merges travel-time information for each edge of the two affiliations (from → to).
          3. Applies a binary proximity indicator variable to represent organisational
             proximity or distance between affiliations.
          4. Adds gravity model–related variables such as log-transformed
             article counts and travel durations.

        :param route_df:
            DataFrame containing travel time information between affiliation pairs.
        :param min_edge_weight:
            The minimum link strength to retain. Edges with lower weights are removed from
            the graph. If ``None``, all edges are included.

        :return:
            A fully enriched DataFrame containing:
              - all affiliation-to-affiliation edges
              - merged travel-time and route information
              - proximity dummy variables indicating organisational proximity or distance
              - gravity model features (log of article count products and travel durations)
            ready for input into a ZNIB regression model.
        """

        self._create_affiliation_graph(min_edge_weight=min_edge_weight)

        edges = create_znib_model_input(edge_gdf=self.edge.gdf, route_df=route_df)
        return edges
