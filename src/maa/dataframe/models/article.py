# Copyright © 2025 Christoph Schlager, TU Wien

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series


class ArticleSchema(pa.DataFrameModel):
    """Schema for pandas article DataFrame."""

    id: Optional[Series[np.int64]] = pa.Field(nullable=False)
    doi: Series[str] = pa.Field(nullable=False)
    eid: Series[str] = pa.Field(nullable=False)
    author_ids: Series[str] = pa.Field(nullable=False)
    author_afids: Series[str] = pa.Field(nullable=False)
    citedby_count: Optional[Series[int]] = pa.Field(nullable=False)
    cover_date: Series[datetime] = pa.Field(nullable=True, coerce=True)


@pa.check_types(lazy=True)
def read_articles(path: Path) -> DataFrame[ArticleSchema]:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df
