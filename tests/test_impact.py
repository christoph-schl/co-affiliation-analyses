from typing import List

import numpy as np
import pandas as pd
import pytest

from src.mma.constants import MWPR_COLUMN
from src.mma.impact.utils import get_mean_weighted_percentile_ranks

_COUNTRY_COLUMN = "country"


@pytest.mark.parametrize(
    "min_samples, expected_mwpr",
    [
        (0, [85.0, 82.5, 81.0]),
        (3, [85.0, 81.0]),
        (4, [81.0]),
    ],
)
def test_get_mean_weighted_percentile_ranks(
    impact_country_test_df: pd.DataFrame, min_samples: int, expected_mwpr: List[float]
) -> None:

    # add_weights(df=impact_country_test_df, group_column=_COUNTRY_COLUMN)
    mwpr = get_mean_weighted_percentile_ranks(
        df=impact_country_test_df, group_column=_COUNTRY_COLUMN, min_samples=min_samples
    )
    assert np.all(mwpr[MWPR_COLUMN] == expected_mwpr)
