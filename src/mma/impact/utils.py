import enum
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import structlog

from src.mma.constants import (
    AFFILIATION_CLASS_COLUMN,
    CLASS_NAME_COLUMN,
    COVER_DATE_COLUMN,
    EID_COLUMN,
    HAZEN_PERCENTILE_COLUMN,
    ITEM_ID_COLUMN,
    MWPR_COLUMN,
)

_logger = structlog.getLogger(__name__)

_SAMPLES_COLUMN = "samples"
_WEIGHT_COLUMN = "weight"
_GROUP_COUNT_COLUMN = "n_groups"


class AffiliationType(enum.Enum):
    ALL = "All"
    FIRST = "First"
    LAST = "Last"


def merge_impact_measures_to_nodes(
    node_df: pd.DataFrame,
    impact_df: pd.DataFrame,
    drop_null_impacts: bool = True,
) -> pd.DataFrame:
    """
    Merge impact measures into `node_df` by matching `node_df.eid` to `impact_df[ITEM_ID_COLUMN]`.
    - Ensures HAZEN_PERCENTILE_COLUMN is numeric (converts commas to dots and coerces errors).
    - Optionally drops rows with missing hazen percentiles and logs how many were dropped.

    :param node_df: The node dataframe.
    :param impact_df: The impact dataframe.
    :param drop_null_impacts: If True, rows with missing hazen percentiles are dropped.
    :return: The merged impact-node dataframe.
    """

    # convert to decimals
    if not np.issubdtype(impact_df[HAZEN_PERCENTILE_COLUMN].dtype, np.number):
        impact_df[HAZEN_PERCENTILE_COLUMN] = pd.to_numeric(
            impact_df[HAZEN_PERCENTILE_COLUMN].str.replace(",", ".", regex=False), errors="coerce"
        )

    node_df = node_df.merge(
        right=impact_df[[ITEM_ID_COLUMN, HAZEN_PERCENTILE_COLUMN, CLASS_NAME_COLUMN]],
        left_on=EID_COLUMN,
        right_on=ITEM_ID_COLUMN,
        how="left",
    )

    if drop_null_impacts:
        null_index = node_df[node_df[HAZEN_PERCENTILE_COLUMN].isnull()].index
        node_df = node_df.drop(null_index).reset_index(drop=True)
        if len(null_index) > 0:
            _logger.warning(f"dropped {len(null_index)} nodes with missing hazen percentiles")

    # tag affiliation type
    first_affiliation_idx = node_df.affiliation_idx == 0
    last_affiliation_idx = node_df.affiliation_idx == node_df.affiliation_count - 1
    node_df.loc[first_affiliation_idx, AFFILIATION_CLASS_COLUMN] = AffiliationType.FIRST.value
    node_df.loc[last_affiliation_idx, AFFILIATION_CLASS_COLUMN] = AffiliationType.LAST.value

    return node_df


def add_weights(df: pd.DataFrame, group_column: str) -> None:
    """
    Adds fractional contributions (`weights`) to the pandas DataFrame.
    :param df: The pandas DataFrame.
    :param group_column: The column name of the `group_column`.
    :return: The pandas DataFrame with `weights` added.
    """
    df[_GROUP_COUNT_COLUMN] = df.groupby([group_column])[group_column].transform(len)
    df[_WEIGHT_COLUMN] = 1 / df[_GROUP_COUNT_COLUMN]


def get_mean_weighted_percentile_ranks(
    df: pd.DataFrame, group_column: str, min_samples: int = 0
) -> pd.DataFrame:
    """
    Computes the Mean Weighted Percentile Rank (mwPR) for each group (`group column`e) using
    the definition given in Formula (14) of Bornmann & Williams (2020),
    *“An evaluation of percentile measures of citation impact, and a proposal for
    making them better”*, Scientometrics, 124(2), 1691–1709.
    https://doi.org/10.1007/s11192-020-03512-7

    The mwPR for a field or unit F is defined as:

        mwPR(F) = ((wPR₁ × FR₁) + (wPR₂ × FR₂) + ... + (wPRᵧ × FRᵧ))
                  ---------------------------------------------------
                                Σᵢ FRᵢ (i = 1 to y)

    where:
      • wPRᵢ  : weighted percentile rank of paper i (e.g. Hazen percentile)
      • FRᵢ   : Fractional contribution of paper i, defined here as:
                FRᵢ = 1 / (number of papers in the same group defined by `group_column`)
                This means each paper contributes equally within its group, and the total
                fractional contribution across all papers in a group sums to 1.
      • y     : total number of papers within the group

      - Only groups with at least `min_samples` papers are included

    :param df:
        The DataFrame containing at least the columns:
              - `eid` column (entity identifier),
              - `group_column` (used to compute weights),
              - `hazen_perc_med` column (percentile measure for each row).
    :param group_column:
        The name of the column in `df` that contains the group.

    :param min_samples:
        The minimum number of samples for a group to be included.
    :return:
        The DataFrame with mwPRs for each group.
    """

    # add weights to DataFrame
    add_weights(df=df, group_column=group_column)

    df[_SAMPLES_COLUMN] = df.groupby(group_column)[group_column].transform("count")
    df = df[df[_SAMPLES_COLUMN] >= min_samples].reset_index(drop=True)

    grouped = df.groupby(group_column, group_keys=False)
    mean_weighted_pr = grouped.apply(
        lambda g: pd.Series(
            {
                MWPR_COLUMN: (g[HAZEN_PERCENTILE_COLUMN] * g[_WEIGHT_COLUMN]).sum()
                / g[_WEIGHT_COLUMN].sum(),
                _SAMPLES_COLUMN: g[_SAMPLES_COLUMN].iloc[0],
            }
        ),
        include_groups=False,
    ).reset_index()
    mean_weighted_pr[_SAMPLES_COLUMN] = mean_weighted_pr[_SAMPLES_COLUMN].astype("int64")

    mean_weighted_pr = mean_weighted_pr.sort_values(by=[MWPR_COLUMN], ascending=False).reset_index(
        drop=True
    )
    return mean_weighted_pr


def _compute_mwpr_for_class(
    node_df: pd.DataFrame, aff_class: str, filter_names: list[str], group_column: str
) -> pd.DataFrame:
    """
    Private helper to compute mwPR for a specific affiliation class and filter
    groups to match the given list of affiliation names.

    :param aff_class: AffiliationType value (e.g., FIRST or LAST)
    :param filter_names: list of affiliation names to retain
    :param group_column: The group column
    :return: filtered mwPR DataFrame
    """
    df_class = node_df[node_df[AFFILIATION_CLASS_COLUMN] == aff_class].copy()
    mwpr_class = get_mean_weighted_percentile_ranks(
        df=df_class, group_column=group_column, min_samples=0
    )
    return mwpr_class[mwpr_class[group_column].isin(filter_names)].reset_index(drop=True)


@dataclass
class AffiliationMwpr:
    """
    Container for mwPR results computed by affiliation group.
    """

    all: pd.DataFrame
    first: pd.DataFrame
    last: pd.DataFrame

    @property
    def groups(self) -> List[str]:
        return self.all.iloc[:, 0].tolist()

    def to_concatenated(self) -> pd.DataFrame:
        """
        Concatenate all mwPR DataFrames into a single DataFrame with an additional column
        indicating the group source ('all', 'first', or 'last').

        :return: A concatenated pandas DataFrame with a `group_type` column.
        """
        return pd.concat(
            [
                self.all,
                self.first,
                self.last,
            ],
            ignore_index=True,
        ).round(decimals=2)


def compute_mwpr_for_affiliation_class(
    node_df: pd.DataFrame,
    n_groups: int,
    min_samples: int = 0,
    group_column: str = "preferred_name",
) -> AffiliationMwpr:
    """
    Compute mwPR for all affiliations as well as for first and last author affiliations.

    :param node_df: DataFrame containing node-level data
    :param n_groups: number of top groups to retain for 'ALL' affiliation class
    :param min_samples: minimum sample size for mwPR computation
    :param group_column: The group column for which to compute mwPR
    :return: The DataFrame with 'all', 'first', 'last' mapping to corresponding mwPR DataFrames
    """

    # compute mwPR for all affiliations
    mwpr_all = get_mean_weighted_percentile_ranks(
        df=node_df, group_column=group_column, min_samples=min_samples
    ).head(n=n_groups)
    mwpr_all[AFFILIATION_CLASS_COLUMN] = AffiliationType.ALL.value

    # Keep list of affiliation names for filtering
    affiliation_names = mwpr_all[group_column].tolist()

    # --- Compute mwPR for first and last authors ---
    mwpr_first = _compute_mwpr_for_class(
        node_df=node_df,
        aff_class=AffiliationType.FIRST.value,
        filter_names=affiliation_names,
        group_column=group_column,
    )
    mwpr_first[AFFILIATION_CLASS_COLUMN] = AffiliationType.FIRST.value

    mwpr_last = _compute_mwpr_for_class(
        node_df=node_df,
        aff_class=AffiliationType.LAST.value,
        filter_names=affiliation_names,
        group_column=group_column,
    )
    mwpr_last[AFFILIATION_CLASS_COLUMN] = AffiliationType.LAST.value

    mwpr = AffiliationMwpr(all=mwpr_all, first=mwpr_first, last=mwpr_last)

    return mwpr


def aggregate_mwpr_over_time(
    node_df: pd.DataFrame,
    group_column: str,
    time_freq: int,
) -> pd.DataFrame:
    """
    Compute mean weighted percentile ranks (MWPR) over time intervals.

    Groups the input DataFrame by time periods of `time_freq` years,
    computes MWPR within each group, and assigns each period its midpoint date.

    :param node_df: Input DataFrame containing the COVER_DATE_COLUMN and group_column.
    :param group_column: Column name used for grouping in MWPR computation.
    :param time_freq: Time interval in years for grouping.
    :return: DataFrame of MWPR values with midpoints for each time interval.
    """

    df = node_df.copy()

    # Group by time intervals of given frequency starting each January
    grouped = df.groupby(
        pd.Grouper(
            key=COVER_DATE_COLUMN,
            freq=f"{time_freq}YS-JAN",
            origin="start",
        )
    )

    # Compute MWPR per time period
    result = grouped.apply(
        lambda g: get_mean_weighted_percentile_ranks(
            df=g,
            group_column=group_column,
            min_samples=0,
        ),
        include_groups=False,
    ).reset_index()

    # Shift date to the midpoint of the time interval
    midpoint_offset = pd.DateOffset(months=int(time_freq * 12 / 2))
    result[COVER_DATE_COLUMN] += midpoint_offset

    return result
