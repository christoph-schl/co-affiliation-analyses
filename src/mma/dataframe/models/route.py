from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series


class RouteSchema(pa.DataFrameModel):
    """Schema for pandas article DataFrame."""

    affiliation_id_from: Series[np.int64] = pa.Field(nullable=False)
    affiliation_id_to: Series[np.int64] = pa.Field(nullable=False)
    duration_s: Series[float] = pa.Field(nullable=True)


@pa.check_types(lazy=True)
def read_routes(path: Path) -> DataFrame[RouteSchema]:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df
