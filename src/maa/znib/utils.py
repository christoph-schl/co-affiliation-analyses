from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Optional, Union

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
import structlog

from maa.constants.constants import (
    AFFILIATION_EDGE_COUNT_COLUMN,
    AFFILIATION_ID_COLUMN,
    ARTICLE_COUNT_COLUMN,
    DEFAULT_REGULARIZATION_STRENGTH,
    FROM_NODE_COLUMN,
    LN_DURATION_COLUMN,
    LN_PROD_ARTICLE_COUNT_COLUMN,
    ORGANISATION_TYPE_COLUMN,
    TO_NODE_COLUMN,
    TRAVEL_TIME_SEC_COLUMN,
)
from maa.network.utils import get_combination_list
from maa.utils.utils import filter_organization_types

_logger = structlog.getLogger(__name__)

_SAME_ORG_TYPE_COLUMN_NAME = "same_org"


def _build_proximity_dict(org_types: list[str]) -> dict[str, dict[str, str]]:
    """
    Build a proximity mapping for all pairs of organisation types.
    :param org_types: List of organisation type names.
    :return: A dict mapping e.g. 'univ_gov' → {'from': 'univ', 'to': 'gov'}
    """

    proximity_dict = {}
    for pair in combinations(org_types, 2):
        proximity_dict["_".join(pair)] = {"from": pair[0], "to": pair[1]}

    for org_type in org_types:
        proximity_dict[f"{org_type}_{org_type}"] = {"from": org_type, "to": org_type}

    return proximity_dict


def _classify_proximity(
    edge_df: pd.DataFrame,
    class_name: str,
    first_org_type: str,
    second_org_type: str,
) -> None:

    org_type_to = f"{ORGANISATION_TYPE_COLUMN}_to"
    edge_df[class_name] = 0

    edge_df.loc[
        (edge_df[ORGANISATION_TYPE_COLUMN] == first_org_type)
        & (edge_df[org_type_to] == second_org_type),
        class_name,
    ] = 1
    edge_df.loc[
        (edge_df[org_type_to] == first_org_type)
        & (edge_df[ORGANISATION_TYPE_COLUMN] == second_org_type),
        class_name,
    ] = 1


def _apply_proximity_dummy(edge_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds dummy variables to an edge DataFrame for a zero-inflated binomial regression (ZNIB) model,
    encoding organisational proximity and distance.

    Each dummy variable represents a specific pair of organisation types:
        - 'proximity': same organisation type (e.g., 'univ_univ')
        - 'distance': different organisation types (e.g., 'univ_resi')

    For each pair, a new column is added to `edge_df` with values:
        - 1 if the edge matches the pair,
        - 0 otherwise.

    :param edge_df: The edge DataFrame containing organisation type columns:
                    - `ORGANISATION_TYPE_COLUMN` for the "from" node
                    - `f'{ORGANISATION_TYPE_COLUMN}_to'` for the "to" node
    :return: The edge DataFrame with new columns representing each proximity (same org type)
             or distance (different org type) dummy variable. Values are 0 or 1.
             Column names follow the pattern '<from>_<to>' (e.g., 'univ_univ', 'univ_resi').

    Example:
        >>> edge_df = pd.DataFrame({
        ...     'from_org': ['univ', 'resi', 'univ'],
        ...     'to_org': ['univ', 'comp', 'resi']
        ... })
        >>> _apply_proximity_dummy(edge_df)
           from_org  to_org  univ_univ  univ_resi  resi_comp  ...
        0      univ    univ          1          0          0
        1      resi    comp          0          0          1
        2      univ    resi          0          1          0
    """

    org_type_list = list(
        np.unique(
            edge_df[ORGANISATION_TYPE_COLUMN].unique().tolist()
            + edge_df[f"{ORGANISATION_TYPE_COLUMN}_to"].unique().tolist()
        )
    )
    proximity_dict = _build_proximity_dict(org_types=org_type_list)

    for link_name, org_types in proximity_dict.items():
        _classify_proximity(
            edge_df=edge_df,
            class_name=link_name,
            first_org_type=org_types["from"],
            second_org_type=org_types["to"],
        )

    edge_df[_SAME_ORG_TYPE_COLUMN_NAME] = 0
    edge_df.loc[
        edge_df[ORGANISATION_TYPE_COLUMN] == edge_df[f"{ORGANISATION_TYPE_COLUMN}_to"],
        _SAME_ORG_TYPE_COLUMN_NAME,
    ] = 1


def get_znib_edges(edge_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Build a dataframe of all affiliation-to-affiliation combinations enriched with organisation
    types (for both sides) and the affiliation edge count.

    :param edge_gdf: The GeoDataFrame containing the edge data.
    :return: The DataFrame of all affiliation-to-affiliation combinations.
    """

    # column name aliases
    from_col = FROM_NODE_COLUMN
    to_col = TO_NODE_COLUMN
    org_col = ORGANISATION_TYPE_COLUMN
    count_col = AFFILIATION_EDGE_COUNT_COLUMN
    article_count_col = ARTICLE_COUNT_COLUMN
    aff_id = AFFILIATION_ID_COLUMN

    # gather unique affiliation ids (from both columns)
    if edge_gdf.empty:
        unique_affiliations = np.array([], dtype=object)
    else:
        unique_affiliations = pd.unique(edge_gdf[[from_col, to_col]].values.ravel())

    # all possible affiliation combinations
    combos = pd.DataFrame(
        get_combination_list(input_list=unique_affiliations), columns=[from_col, to_col]
    )

    # build affiliation -> organisation_type mapping (from both sides)
    edges_from = edge_gdf[[from_col, org_col, article_count_col]].rename(columns={from_col: aff_id})
    edges_to = edge_gdf[[to_col, f"{org_col}_to", f"{article_count_col}_to"]].rename(
        columns={
            to_col: aff_id,
            f"{org_col}_to": org_col,
            f"{article_count_col}_to": article_count_col,
        }
    )
    affiliation_map = pd.concat([edges_from, edges_to], ignore_index=True).drop_duplicates(
        subset=[aff_id]
    )

    # merge org types for both ends and merge the existing edge counts (if any)
    znib_edges = (
        combos.merge(affiliation_map, left_on=from_col, right_on=aff_id, how="left")
        .merge(affiliation_map, left_on=to_col, right_on=aff_id, how="left", suffixes=("", "_to"))
        .merge(edge_gdf[[from_col, to_col, count_col]], on=[from_col, to_col], how="left")
        .drop(columns=[aff_id, f"{aff_id}_to"])
    )

    # ensure counts are integers and missing counts are treated as 0
    if count_col in znib_edges.columns:
        znib_edges[count_col] = znib_edges[count_col].fillna(0).astype(int)
    else:
        znib_edges[count_col] = 0

    return znib_edges


def _merge_routes_to_edges(edge_df: pd.DataFrame, route_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge travel time information into the given edge dataframe.

    :param edge_df: The DataFrame containing edges
    :param route_df: The DataFrame containing route info between affiliations
    :return: The edge_df with route data merged in.
    """

    # Column name aliases
    from_col, to_col = FROM_NODE_COLUMN, TO_NODE_COLUMN
    aff_from, aff_to = f"{AFFILIATION_ID_COLUMN}_from", f"{AFFILIATION_ID_COLUMN}_to"
    duration_col = TRAVEL_TIME_SEC_COLUMN

    # merge routes into edges
    merged = edge_df.merge(
        route_df[[aff_from, aff_to, duration_col]],
        left_on=[from_col, to_col],
        right_on=[aff_from, aff_to],
        how="left",
    ).drop(columns=[aff_from, aff_to])

    # identify edges with missing route info
    missing_routes = merged[merged[duration_col].isna()]

    if not missing_routes.empty:
        _logger.warning("%d route(s) not found for edges", len(missing_routes))

    return merged


def enrich_edges_with_org_info(
    edge_gdf: gpd.GeoDataFrame, route_df: pd.DataFrame, org_type_list: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Creates the input DataFrame for the ZNIB (Zero-Inflated Negative Binomial) model.

    This function prepares and enriches edge-level data to serve as input for
    a ZNIB gravity model of multi-affiliation collaboration intensity.
    It performs the following steps:

      1. Generates all affiliation-to-affiliation edge combinations.
      2. Merges travel-time information for each edge of the two affiliations (from → to).
      3. Applies a binary proximity indicator variable to represent organisational
         proximity or distance between affiliations.
      4. Adds gravity model–related variables such as log-transformed
         article counts and travel durations.

    :param edge_gdf:
        Base edge-level DataFrame containing observed collaboration counts and
        affiliation-level metadata (e.g., organisation type, article counts).
    :param route_df:
        DataFrame containing travel time information between affiliation pairs.
    :param org_type_list:
        Optional list of organisation types used to filter the edges connecting organisations.
    :return:
        A fully enriched DataFrame containing:
          - all affiliation-to-affiliation edges
          - merged travel-time and route information
          - proximity dummy variables indicating organisational proximity or distance
          - gravity model features (log of article count products and travel durations)
        ready for input into a ZNIB regression model.
    """

    if org_type_list is not None:
        edge_gdf = filter_organization_types(df=edge_gdf, org_types=org_type_list)

    znib_edges = get_znib_edges(edge_gdf=edge_gdf)

    znib_edges = _merge_routes_to_edges(edge_df=znib_edges, route_df=route_df)

    _apply_proximity_dummy(edge_df=znib_edges)
    _add_gravity_model_variables(edge_df=znib_edges)

    return znib_edges


def _add_gravity_model_variables(edge_df: pd.DataFrame) -> None:
    """
    Adds gravity model–related variables to the given edge DataFrame.

        This function enriches the input `edge_df` with additional variables
        used in gravity model computations. Specifically:
          - Fills missing edge counts with 0.
          - Ensures the edge count column is of integer type.
          - Computes the natural log of the product of article counts
            (i.e., ln(article_count_from * article_count_to)).
          - Computes the natural log of travel duration (in seconds),
            offset by +1 to avoid log(0) errors.

    :param edge_df:
    :return:  The same DataFrame with new columns:
                - `ln_prod_article_count`
                - `ln_duration`
    """

    count_col = AFFILIATION_EDGE_COUNT_COLUMN
    duration_col = TRAVEL_TIME_SEC_COLUMN
    ln_article_col = LN_PROD_ARTICLE_COUNT_COLUMN
    article_count_col = ARTICLE_COUNT_COLUMN
    ln_dur_col = LN_DURATION_COLUMN

    edge_df.loc[edge_df[count_col].isnull(), count_col] = 0
    edge_df[count_col] = edge_df[count_col].astype(np.int64)
    edge_df[ln_article_col] = np.log(edge_df[article_count_col] * edge_df[article_count_col])
    edge_df[ln_dur_col] = np.log(edge_df[duration_col] + 1)


@dataclass
class ZNIBInput:
    """
    Container for Zero-Inflated Negative Binomial (ZINB) model input matrices
    and computed start parameters for optimization.
    """

    dependent: npt.NDArray
    predictors: pd.DataFrame
    inflation: pd.DataFrame
    alpha: float = DEFAULT_REGULARIZATION_STRENGTH
    _start_parameters: npt.NDArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Compute start parameters automatically after initialization."""
        self._compute_start_params()

    @property
    def start_parameters(self) -> npt.NDArray:
        """Read-only access to the precomputed start vector."""
        return self._start_parameters

    def _compute_start_params(self) -> None:
        """Compute start parameters for ZINB: [count_coefs..., infl_coefs..., alpha_dispersion]."""
        y = self.dependent.ravel()
        X_count = self.predictors.copy()
        X_infl = self.inflation.copy()

        # --- Inflation start: regularized logit ---
        inflation_target = (y == 0).astype(int)
        logit = sm.Logit(inflation_target, X_infl)
        logit_res = logit.fit_regularized(alpha=self.alpha, disp=False)
        start_infl = np.asarray(logit_res.params, dtype=float)

        # --- Count start: NB GLM ---
        nb_model = sm.GLM(y, X_count, family=sm.families.NegativeBinomial(alpha=1.0))
        nb_res = nb_model.fit()
        start_count = np.asarray(nb_res.params, dtype=float)

        # --- Dispersion: use GLM scale as starting guess ---
        dispersion = np.clip(float(nb_res.scale), 1e-6, 100.0)
        start_alpha = np.array([dispersion], dtype=float)

        # --- Concatenate full start vector ---
        self._start_parameters = np.concatenate([start_count, start_infl, start_alpha])


def prepare_znib_input(
    df: pd.DataFrame,
    dependent_variable: str,
    predictor_variables: List[str],
    zero_inflation_vars: Union[str, List[str]],
) -> ZNIBInput:
    """
    Prepare ZINB input matrices and wrap them in a ZNIBVars dataclass.

    :param df:
         Source dataframe containing all relevant columns.
    :param dependent_variable:
        Name of the dependent variable.
    :param predictor_variables:
        Columns used as predictors in the NB count part.
    :param zero_inflation_vars:
        Column(s) used as predictors for the inflation (logit) part.
    :return:
        Dataclass containing dependent array, predictor matrix, inflation matrix,
        and precomputed start parameters.
    """

    if isinstance(zero_inflation_vars, str):
        zero_inflation_vars = [zero_inflation_vars]

    dependent = df[dependent_variable].to_numpy()
    predictors = sm.add_constant(df[predictor_variables].copy(), has_constant="add")
    inflation = sm.add_constant(df[zero_inflation_vars].copy(), has_constant="add")

    return ZNIBInput(dependent=dependent, predictors=predictors, inflation=inflation)
