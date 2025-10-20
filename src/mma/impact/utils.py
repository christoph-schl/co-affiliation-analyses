import numpy as np
import pandas as pd
import structlog

from src.mma.constants import (
    CLASS_NAME_COLUMN,
    EID_COLUMN,
    HAZEN_PERCENTILE_COLUMN,
    ITEM_ID_COLUMN,
    MWPR_COLUMN,
    ORGANISATION_TYPE_COLUMN,
)

_logger = structlog.getLogger(__name__)

_SAMPLES_COLUMN = "samples"
_WEIGHT_COLUMN = "weight"
_GROUP_COUNT_COLUMN = "n_orgs"


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
    return node_df


def _add_weights(df: pd.DataFrame, group_column: str) -> None:
    df[_GROUP_COUNT_COLUMN] = df.groupby([group_column])[ORGANISATION_TYPE_COLUMN].transform(len)
    df[_WEIGHT_COLUMN] = 1 / df[_GROUP_COUNT_COLUMN]


def get_mean_weighted_percentile_ranks(
    df: pd.DataFrame, group_column: str, min_samples: int
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
    _add_weights(df=df, group_column=group_column)

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
