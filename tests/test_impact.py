from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from src.maa.constants import (
    AFFILIATION_CLASS_COLUMN,
    ARTICLE_AUTHOR_ID_COLUMN,
    EID_COLUMN,
    FROM_AFFILIATION_INDEX_COLUMN,
    MWPR_COLUMN,
    ORGANISATION_TYPE_COLUMN,
    PREFERRED_AFFILIATION_NAME_COLUMN,
    TO_AFFILIATION_INDEX_COLUMN,
)
from src.maa.impact.impact import Impact
from src.maa.impact.utils import (
    AffiliationType,
    get_mean_weighted_percentile_ranks,
    merge_impact_measures_to_nodes,
)
from src.maa.network.network import AffiliationNetworkProcessor
from src.maa.utils.utils import get_link_nodes

_COUNTRY_COLUMN = "country"
_MAX_AFFILIATION_IDX_COLUMN = "max_affiliation_idx"


@pytest.mark.parametrize(
    "min_samples, expected_mwpr",
    [
        (0, [85.0, 82.0, 78.18]),
        (3, [85.0, 78.18]),
        (4, []),
    ],
)
def test_get_mean_weighted_percentile_ranks(
    impact_country_test_df: pd.DataFrame, min_samples: int, expected_mwpr: List[float]
) -> None:
    mwpr = get_mean_weighted_percentile_ranks(
        df=impact_country_test_df, group_column=_COUNTRY_COLUMN, min_samples=min_samples
    ).round(2)
    assert np.all(mwpr[MWPR_COLUMN] == expected_mwpr)


@pytest.mark.parametrize(
    "min_samples, expected_mwpr",
    [
        (0, [89.88, 86.41]),
        (1, [89.88, 86.41]),
        (3, []),
    ],
)
def test_get_mean_weighted_percentile_ranks_for_affiliation_links(
    link_df: pd.DataFrame, impact_df: pd.DataFrame, min_samples: int, expected_mwpr: List[float]
) -> None:
    impact = Impact(link_gdf=link_df, impact_df=impact_df)
    mwpr = impact.get_mwpr(group_column=ORGANISATION_TYPE_COLUMN, min_samples=min_samples).round(2)
    assert np.all(mwpr[MWPR_COLUMN] == expected_mwpr)


def test_get_mean_weighted_percentile_ranks_pipline(
    article_df: pd.DataFrame, affiliation_gdf: gpd.GeoDataFrame, impact_df: pd.DataFrame
) -> None:
    # create links
    processor = AffiliationNetworkProcessor(article_df=article_df, affiliation_gdf=affiliation_gdf)
    links = processor.get_affiliation_links()

    impact = Impact(link_gdf=links, impact_df=impact_df)
    mwpr = impact.get_mwpr(group_column=ORGANISATION_TYPE_COLUMN)
    assert mwpr[ORGANISATION_TYPE_COLUMN].tolist() == ["gov", "res", "uni"]


def test_merge_impact_measures_to_nodes(link_df: pd.DataFrame, impact_df: pd.DataFrame) -> None:
    node_df = get_link_nodes(link_gdf=link_df)
    node_df = merge_impact_measures_to_nodes(node_df=node_df, impact_df=impact_df)

    _test_first_affiliation_classification(link_df=link_df, node_df=node_df)
    _test_last_affiliation_classification(link_df=link_df, node_df=node_df)


def _test_last_affiliation_classification(link_df: pd.DataFrame, node_df: pd.DataFrame) -> None:
    link_df[_MAX_AFFILIATION_IDX_COLUMN] = link_df.groupby([EID_COLUMN, ARTICLE_AUTHOR_ID_COLUMN])[
        TO_AFFILIATION_INDEX_COLUMN
    ].transform(max)
    la_preferred_name = link_df[
        link_df[TO_AFFILIATION_INDEX_COLUMN] == link_df[_MAX_AFFILIATION_IDX_COLUMN]
    ][f"{PREFERRED_AFFILIATION_NAME_COLUMN}_to"].tolist()
    node_la = node_df.loc[
        node_df[PREFERRED_AFFILIATION_NAME_COLUMN].isin(la_preferred_name), AFFILIATION_CLASS_COLUMN
    ]
    assert np.all(node_la == AffiliationType.LA.value)


def _test_first_affiliation_classification(link_df: pd.DataFrame, node_df: pd.DataFrame) -> None:
    fa_preferred_name = link_df[link_df[FROM_AFFILIATION_INDEX_COLUMN] == 0][
        PREFERRED_AFFILIATION_NAME_COLUMN
    ].tolist()
    fa_preferred_name.append(
        link_df[link_df[TO_AFFILIATION_INDEX_COLUMN] == 0][
            PREFERRED_AFFILIATION_NAME_COLUMN
        ].tolist()
    )
    node_fa = node_df.loc[
        node_df[PREFERRED_AFFILIATION_NAME_COLUMN].isin(fa_preferred_name), AFFILIATION_CLASS_COLUMN
    ]
    assert np.all(node_fa == AffiliationType.FA.value)
