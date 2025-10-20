from dataclasses import dataclass, field
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
    scale_vars: List[str] = field(default_factory=list)


config_same = ZINBConfig(
    dependent_var="affiliation_edge_count",
    predictor_vars=["ln_prod_article_count", "ln_duration", "same_org"],
    inflation_var="ln_prod_article_count",
    max_iterations=1000,
    alpha_reg=3.0,
    scale_vars=["ln_prod_article_count", "ln_duration"],
)


config_proximity = ZINBConfig(
    dependent_var="affiliation_edge_count",
    predictor_vars=[
        "ln_prod_article_count",
        "ln_duration",
        "univ_univ",
        "resi_resi",
        "hosp_hosp",
        "comp_comp",
        "coll_coll",
        "ngo_ngo",
        "gov_gov",
    ],
    inflation_var="ln_prod_article_count",
    max_iterations=1000,
    alpha_reg=3.0,
)

config_distance = ZINBConfig(
    dependent_var="affiliation_edge_count",
    predictor_vars=[
        "ln_prod_article_count",
        "ln_duration",
        "comp_univ",
        "resi_univ",
        "coll_univ",
        "hosp_univ",
        "ngo_univ",
        "gov_univ",
        "comp_resi",
        "coll_resi",
        "hosp_resi",
        "ngo_resi",
        "comp_hosp",
        "coll_hosp",
        "hosp_ngo",
        "gov_hosp",
        "coll_comp",
        "coll_ngo",
        "coll_gov",
    ],
    inflation_var="ln_prod_article_count",
    max_iterations=1000,
    alpha_reg=3.0,
    scale_vars=["ln_prod_article_count", "ln_duration"],
)
