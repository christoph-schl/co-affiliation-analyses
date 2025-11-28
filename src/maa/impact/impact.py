# Copyright © 2025 Christoph Schlager, TU Wien

from dataclasses import dataclass, field
from typing import List, Optional, Union

import geopandas as gpd
import pandas as pd

from maa.constants.constants import (
    DEFAULT_MAX_WORKERS_PARALLEL_PROCESSING,
)
from maa.impact.utils import (
    compute_rolling_mwpr,
    get_mean_weighted_percentile_ranks,
    merge_impact_measures_to_nodes,
)
from maa.utils.utils import filter_organization_types, get_link_nodes


@dataclass
class Impact:
    """
    Used to compute node-level citation impact metrics and to aggregate them into
    Mean Weighted Percentile Ranks (mwPR) across user-defined groups, based on
    Formula (14) from Bornmann & Williams (2020).

    This class performs two main steps automatically upon initialization:
      1. Converts link-based geospatial data (`link_gdf`) into a node-level
         representation using `get_link_nodes()`.
      2. Merges citation percentile information from `impact_df` into the
         node DataFrame using `merge_impact_measures_to_nodes()`.

    The resulting node-level DataFrame (`_node_df`) contains one row per node
    with associated percentile scores and metadata, and serves as the input for
    group-based mwPR computations.

    Scientific Basis
    ----------------
    The Mean Weighted Percentile Rank (mwPR) is calculated using the definition
    in Formula (14) of:

        Bornmann, L., & Williams, R. (2020).
        *An evaluation of percentile measures of citation impact, and a proposal for
        making them better*. Scientometrics, 124, 1691–1709.
        https://doi.org/10.1007/s11192-020-03512-7

    The mwPR for a group F (e.g. institution, field, region) is defined as:

        mwPR(F) = ((wPR₁ × FR₁) + (wPR₂ × FR₂) + ... + (wPRᵧ × FRᵧ)) / Σᵢ FRᵢ

    where:
      • wPRᵢ  : the percentile rank (e.g. Hazen percentile) of node i
      • FRᵢ   : the fractional contribution of node i, defined here as:
                FRᵢ = 1 / (number of nodes in the same group)
                ensuring each node contributes equally within its group
      • y     : the total number of nodes in the group
    """

    link_gdf: Union[gpd.GeoDataFrame, pd.DataFrame]
    impact_df: pd.DataFrame
    allowed_org_types: Optional[List[str]] = None
    _node_df: Optional[pd.DataFrame] = field(default=None, init=False)

    def __post_init__(self) -> None:

        # filter organisation types
        if self.allowed_org_types is not None:
            self.link_gdf = filter_organization_types(
                df=self.link_gdf, org_types=self.allowed_org_types
            )

        self._node_df = get_link_nodes(link_gdf=self.link_gdf)
        self._node_df = merge_impact_measures_to_nodes(
            node_df=self._node_df, impact_df=self.impact_df
        )

    @property
    def node_df(self) -> Optional[pd.DataFrame]:
        return self._node_df

    def get_mwpr(self, group_column: str, min_samples: int = 0) -> pd.DataFrame:
        """
        Computes the Mean Weighted Percentile Rank (mwPR) for each group in the
        internal node DataFrame.

        This method applies the mwPR definition described in the class docstring,
        using fractional weights computed as 1 / (number of nodes per group).

        :param group_column:
            The column name of the node-level citation impact metric column.
        :param min_samples:
            The minimum number of samples for a group to be included.
        :return:
            The DataFrame with mwPRs for each group.
        """

        mwpr_df = get_mean_weighted_percentile_ranks(
            df=self._node_df, group_column=group_column, min_samples=min_samples
        )
        return mwpr_df

    def get_rolling_mwpr(
        self,
        group_column: str,
        time_window: str,
        max_workers: int = DEFAULT_MAX_WORKERS_PARALLEL_PROCESSING,
    ) -> pd.DataFrame:
        """
        Performs a rolling mwPR computation of the hazen percentiles on each time interval for a
        specified time window.

        :param time_window:
            A pandas-compatible rolling window specification (e.g., '365D' for a 365-day window).
            The window is centered to capture temporal context around each observation.
        :param group_column:
            Column name in `nodes_df` that defines the grouping or affiliation for computing
            mean weighted percentile ranks.
        :param max_workers:
             Number of worker processes to use for parallel processing. Defaults to
            `DEFAULT_MAX_WORKERS_PARALLEL_PROCESSING`.
        :return:
            A DataFrame containing rolling MWPR values. Each row represents an index position
            corresponding to the center of a rolling time window, with the computed MWPR and
            associated cover date.
        """

        return compute_rolling_mwpr(
            nodes_df=self._node_df,
            time_window=time_window,
            group_column=group_column,
            max_workers=max_workers,
        )
