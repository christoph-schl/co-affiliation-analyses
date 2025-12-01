# Copyright © 2025 Christoph Schlager, TU Wien

import enum
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import structlog

from maa.constants.constants import (
    AFFILIATION_CLASS_COLUMN,
    AFFILIATION_ID_COLUMN,
    CLASS_NAME_COLUMN,
    COVER_DATE_COLUMN,
    DEFAULT_MAX_WORKERS_PARALLEL_PROCESSING,
    EID_COLUMN,
    HAZEN_PERCENTILE_COLUMN,
    ITEM_ID_COLUMN,
    MWPR_COLUMN,
    ORGANISATION_TYPE_COLUMN,
    SAMPLES_COLUMN,
)
from maa.utils.wrappers import get_execution_time, parallelize_dataframe

_logger = structlog.getLogger(__name__)

_WEIGHT_COLUMN = "weight"
_GROUP_COUNT_COLUMN = "n_groups"
_INDEX_COLUMN = "df_index"
_MIN_INDEX_COLUMN = "min_index"
_MAX_INDEX_COLUMN = "max_index"
_MWPR_INDEX_COLUMN = "mwpr_index"
_AFFILIATION_INDEX_COLUMN = "affiliation_idx"
_CHUNK_SIZE_PARALLEL_PROCESSING = 15
_NODE_COLUMN = "node"


class AffiliationType(enum.Enum):
    AA = "AA"
    FA = "FA"
    LA = "LA"


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
    if _AFFILIATION_INDEX_COLUMN in node_df.columns:
        first_affiliation_idx = node_df.affiliation_idx == 0
        last_affiliation_idx = node_df.affiliation_idx == node_df.affiliation_count - 1
        node_df.loc[first_affiliation_idx, AFFILIATION_CLASS_COLUMN] = AffiliationType.FA.value
        node_df.loc[last_affiliation_idx, AFFILIATION_CLASS_COLUMN] = AffiliationType.LA.value

    node_df = node_df.rename(columns={_NODE_COLUMN: AFFILIATION_ID_COLUMN}, errors="ignore")

    return node_df


def add_weights(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """
    Adds fractional contributions (`weights`) to the pandas DataFrame.
    :param df: The pandas DataFrame.
    :param group_column: The column name of the `group_column`.
    :return: The pandas DataFrame with `weights` added.
    """
    df[_GROUP_COUNT_COLUMN] = df.groupby([EID_COLUMN])[group_column].transform("nunique")
    df[_WEIGHT_COLUMN] = 1 / df[_GROUP_COUNT_COLUMN]

    df = df.drop_duplicates([EID_COLUMN, group_column]).reset_index(drop=True)
    return df


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

    df = df.copy()

    # add weights to DataFrame
    df = add_weights(df=df, group_column=group_column)

    df[SAMPLES_COLUMN] = df.groupby(group_column)[group_column].transform("count")
    df = df[df[SAMPLES_COLUMN] >= min_samples].reset_index(drop=True)

    grouped = df.groupby(group_column, group_keys=False)
    mean_weighted_pr = grouped.apply(
        lambda g: pd.Series(
            {
                MWPR_COLUMN: (g[HAZEN_PERCENTILE_COLUMN] * g[_WEIGHT_COLUMN]).sum()
                / g[_WEIGHT_COLUMN].sum(),
                SAMPLES_COLUMN: g[SAMPLES_COLUMN].iloc[0],
            }
        ),
        include_groups=False,
    ).reset_index()
    mean_weighted_pr[SAMPLES_COLUMN] = mean_weighted_pr[SAMPLES_COLUMN].astype("int64")

    # merge organisation type and affiliation id to df
    if (group_column != ORGANISATION_TYPE_COLUMN) and (ORGANISATION_TYPE_COLUMN in df.columns):
        mean_weighted_pr = mean_weighted_pr.merge(
            right=df[
                [group_column, ORGANISATION_TYPE_COLUMN, AFFILIATION_ID_COLUMN]
            ].drop_duplicates(subset=group_column),
            left_on=group_column,
            right_on=group_column,
            how="left",
        )

    if not mean_weighted_pr.empty:
        mean_weighted_pr = mean_weighted_pr.sort_values(
            by=[MWPR_COLUMN], ascending=False
        ).reset_index(drop=True)
        return mean_weighted_pr

    mean_weighted_pr[MWPR_COLUMN] = pd.Series(dtype=float)
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
    mwpr_all[AFFILIATION_CLASS_COLUMN] = AffiliationType.AA.value

    # Keep list of affiliation names for filtering
    affiliation_names = mwpr_all[group_column].tolist()

    # --- Compute mwPR for first and last authors ---
    mwpr_first = _compute_mwpr_for_class(
        node_df=node_df,
        aff_class=AffiliationType.FA.value,
        filter_names=affiliation_names,
        group_column=group_column,
    )
    mwpr_first[AFFILIATION_CLASS_COLUMN] = AffiliationType.FA.value

    mwpr_last = _compute_mwpr_for_class(
        node_df=node_df,
        aff_class=AffiliationType.LA.value,
        filter_names=affiliation_names,
        group_column=group_column,
    )
    mwpr_last[AFFILIATION_CLASS_COLUMN] = AffiliationType.LA.value

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


def _get_rolling_mwpr(
    index_df: pd.DataFrame, node_df: pd.DataFrame, group_column: str
) -> pd.DataFrame:

    node_df = node_df.loc[index_df.min_index.min() : index_df.max_index.max()].copy()  # noqa: E203
    index_df[_MWPR_INDEX_COLUMN] = index_df.apply(
        lambda x: list(range(x.min_index, x.max_index + 1)), axis=1
    )
    index_df = index_df.explode(column=_MWPR_INDEX_COLUMN, ignore_index=True)

    index_df = index_df.merge(
        right=node_df, left_on=_MWPR_INDEX_COLUMN, right_index=True, how="left", suffixes=("", "_n")
    ).drop(columns=[f"{_INDEX_COLUMN}_n", _MWPR_INDEX_COLUMN])

    grouped = index_df.groupby(_INDEX_COLUMN)
    index_df = grouped.apply(
        lambda g: get_mean_weighted_percentile_ranks(
            df=g, group_column=group_column, min_samples=0
        ).assign(
            df_index=g.name
        ),  # attach the group name
        include_groups=False,
    ).reset_index(drop=True)

    return index_df


@get_execution_time
def _get_dict_df(nodes_df: pd.DataFrame, time_window: str) -> pd.DataFrame:
    columns = [COVER_DATE_COLUMN, _INDEX_COLUMN]
    index_dict_list = []
    for idx, roll in enumerate(
        nodes_df[columns].rolling(window=time_window, on=COVER_DATE_COLUMN, center=True)
    ):
        subset_df = roll.copy()

        index_dict_list.append(
            {
                _INDEX_COLUMN: idx,
                _MIN_INDEX_COLUMN: subset_df.df_index.min(),
                _MAX_INDEX_COLUMN: subset_df.df_index.max(),
            }
        )
    dict_df = pd.DataFrame(index_dict_list)

    return dict_df


@get_execution_time
def compute_rolling_mwpr(
    nodes_df: pd.DataFrame,
    time_window: str,
    group_column: str,
    max_workers: int = DEFAULT_MAX_WORKERS_PARALLEL_PROCESSING,
    chunk_size: int = _CHUNK_SIZE_PARALLEL_PROCESSING,
) -> pd.DataFrame:
    """
    Performs a rolling mwPR computation of the hazen percentiles on each time interval for a
    specified time window.

    :param nodes_df:
        Input dataset containing time series records. Must include the column specified by
        `COVER_DATE_COLUMN` (representing the date of each record), as well as the column
        specified by `group_column`, which is used to compute both the Mean Weighted Percentile
        Rank (MWPR) and the `hazen_perc_med` (Hazen median percentile) values.
    :param time_window:
        A pandas-compatible rolling window specification (e.g., '365D' for a 365-day window).
        The window is centered to capture temporal context around each observation.
    :param group_column:
        Column name in `nodes_df` that defines the grouping or affiliation for computing
        mean weighted percentile ranks.
    :param max_workers:
         Number of worker processes to use for parallel processing. Defaults to
        `DEFAULT_MAX_WORKERS_PARALLEL_PROCESSING`.
    :param chunk_size:
        Number of rows to include in each chunk when distributing work across workers.
        Defaults to `_CHUNK_SIZE_PARALLEL_PROCESSING`.
    :return:
        A DataFrame containing rolling MWPR values. Each row represents an index position
        corresponding to the center of a rolling time window, with the computed MWPR and
        associated cover date.
    """

    nodes_df = nodes_df.sort_values(by=COVER_DATE_COLUMN).reset_index(drop=True)
    nodes_df = nodes_df.reset_index(drop=True)
    nodes_df[_INDEX_COLUMN] = nodes_df.index

    dict_df = _get_dict_df(nodes_df, time_window=time_window)

    mwpr_df = parallelize_dataframe(
        input_function=_get_rolling_mwpr,
        df=dict_df,
        node_df=nodes_df,
        group_column=group_column,
        max_workers=max_workers,
        chunk_size=chunk_size,
        verbose=True,
    )
    mwpr_df[COVER_DATE_COLUMN] = nodes_df.loc[mwpr_df.df_index, COVER_DATE_COLUMN].values

    return mwpr_df
