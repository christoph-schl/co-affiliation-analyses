from dataclasses import dataclass, field
from typing import List, Union

from maa.constants.constants import (
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


config_intra_proximity = ZINBConfig(
    dependent_var="affiliation_edge_count",
    predictor_vars=[
        "ln_prod_article_count",
        "ln_duration",
        "uni_uni",
        "res_res",
        "med_med",
        "comp_comp",
        "coll_coll",
        "npo_npo",
        "gov_gov",
    ],
    inflation_var="ln_prod_article_count",
    max_iterations=1000,
    alpha_reg=3.0,
)

config_inter_proximity = ZINBConfig(
    dependent_var="affiliation_edge_count",
    predictor_vars=[
        "ln_prod_article_count",
        "ln_duration",
        "coll_comp",
        "coll_gov",
        "coll_med",
        "coll_npo",
        "coll_res",
        "coll_uni",
        "comp_gov",
        "comp_med",
        "comp_npo",
        "comp_res",
        "comp_uni",
        "gov_med",
        "gov_npo",
        "gov_res",
        "gov_uni",
        "med_npo",
        "med_res",
        "med_uni",
        "npo_res",
        "npo_uni",
        "res_uni",
    ],
    inflation_var="ln_prod_article_count",
    max_iterations=1000,
    alpha_reg=3.0,
    scale_vars=["ln_prod_article_count", "ln_duration"],
)
