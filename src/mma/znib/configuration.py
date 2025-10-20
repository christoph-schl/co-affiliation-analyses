from dataclasses import dataclass
from typing import List, Union

from src.mma.constants import (
    DEFAULT_DISPERSION_POWER,
    DEFAULT_FIT_METHOD,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_REGULARIZATION_STRENGTH,
    DEFAULT_ZERO_INFLATION_MODEL,
)


@dataclass(frozen=True)
class ZINBConfig:
    dependent_var: str
    predictor_vars: List[str]
    inflation_var: Union[str, List[str]]
    zero_inflation_model: str = DEFAULT_ZERO_INFLATION_MODEL
    dispersion_power: int = DEFAULT_DISPERSION_POWER
    fit_method: str = DEFAULT_FIT_METHOD
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    alpha_reg: float = DEFAULT_REGULARIZATION_STRENGTH


# Define configuration
config = ZINBConfig(
    dependent_var="affiliation_edge_count",
    predictor_vars=["ln_prod_article_count", "ln_duration", "univ_univ", "hosp_hosp", "comp_comp"],
    inflation_var="ln_prod_article_count",
)
