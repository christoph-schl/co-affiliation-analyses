from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from src.mma.constants import MWPR_COLUMN, ORGANISATION_TYPE_COLUMN
from src.mma.impact.impact import Impact
from src.mma.impact.utils import get_mean_weighted_percentile_ranks
from src.mma.network.network import AffiliationNetworkProcessor

_COUNTRY_COLUMN = "country"


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


def test_get_mean_weighted_percentile_ranks_pipline(
    article_df: pd.DataFrame, affiliation_gdf: gpd.GeoDataFrame, impact_df: pd.DataFrame
) -> None:

    # create links
    processor = AffiliationNetworkProcessor(article_df=article_df, affiliation_gdf=affiliation_gdf)
    links = processor.get_affiliation_links()

    impact = Impact(link_gdf=links, impact_df=impact_df)
    mwpr = impact.get_mwpr(group_column=ORGANISATION_TYPE_COLUMN)
    assert mwpr[ORGANISATION_TYPE_COLUMN].tolist() == ["gov", "resi", "univ"]
