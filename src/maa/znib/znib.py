from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.count_model import (
    ZeroInflatedNegativeBinomialP,
    ZeroInflatedNegativeBinomialResultsWrapper,
)
from statsmodels.iolib.summary import Summary

from maa.network.network import AffiliationNetworkProcessor
from maa.utils.wrappers import get_execution_time
from maa.znib.configuration import ZINBConfig
from maa.znib.utils import ZNIBInput, enrich_edges_with_org_info, model_results_to_df


class ZINBModel:
    """
    Zero-Inflated Negative Binomial (ZINB) modeling pipeline.

    This class provides a high-level interface for fitting a Zero-Inflated Negative Binomial
    regression model using statsmodels. It wraps data preparation, model initialization, and
    fitting into a reusable pipeline, based on configuration provided via a `ZINBConfig` object.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing the dependent variable, predictor variables, and inflation
        variables as defined in the configuration.
    config : ZINBConfig
        Configuration object specifying model parameters such as dependent variable name,
        predictors, inflation variables, zero-inflation model type, dispersion power, fitting
        method, and regularization strength.
    """

    def __init__(self, df: pd.DataFrame, config: ZINBConfig):
        self.df: pd.DataFrame = df
        self.config: ZINBConfig = config
        self.input_data: ZNIBInput = self._prepare_input()
        self._model: Optional[ZeroInflatedNegativeBinomialP] = None
        self._result: Optional[ZeroInflatedNegativeBinomialResultsWrapper] = None

    def _prepare_input(self) -> ZNIBInput:
        if len(self.config.scale_vars) > 0:
            self.df[self.config.scale_vars] = StandardScaler().fit_transform(
                self.df[self.config.scale_vars]
            )
        dep = self.df[self.config.dependent_var].to_numpy(dtype=float)
        predictors = sm.add_constant(self.df[self.config.predictor_vars].copy(), has_constant="add")

        infl_vars = self.config.inflation_var
        if isinstance(infl_vars, str):
            infl_vars = [infl_vars]
        inflation = sm.add_constant(self.df[infl_vars].copy(), has_constant="add")

        return ZNIBInput(
            dependent=dep, predictors=predictors, inflation=inflation, alpha=self.config.alpha_reg
        )

    def fit(self) -> ZeroInflatedNegativeBinomialResultsWrapper:
        self._model = ZeroInflatedNegativeBinomialP(
            endog=self.input_data.dependent,
            exog=self.input_data.predictors,
            exog_infl=self.input_data.inflation,
            inflation=self.config.zero_inflation_model,
            p=self.config.dispersion_power,
        )
        self._result = self._model.fit(
            start_params=self.input_data.start_parameters,
            method=self.config.fit_method,
            maxiter=self.config.max_iterations,
            disp=True,
            cov_type="HC0",
        )

        return self._result

    @property
    def summary(self) -> Optional[Summary]:
        if self._result is not None:
            return self._result.summary()
        return None

    @property
    def result(self) -> Optional[ZeroInflatedNegativeBinomialResultsWrapper]:
        return self._result

    @property
    def result_df(self) -> Optional[pd.DataFrame]:
        if self._result is not None:
            return model_results_to_df(znib_result=self._result)
        return None


@dataclass
class ZNIB(AffiliationNetworkProcessor):
    """
    A processor for preparing input data for, and performing modelling with, the
    **Zero-Inflated Negative Binomial (ZNIB)** model to estimate the degree of proximity and
    distance in inter-organisational collaboration.

    This class inherits from :class:`AffiliationNetworkProcessor` and extends its
    functionality to both construct and analyse edge-level features for modelling
    collaboration intensity between affiliations. It encapsulates the workflow
    required to prepare, enrich, and optionally fit ZNIB models that capture
    the probabilistic structure of multi-affiliation collaboration networks.

    Core responsibilities include:
      - Generating and enriching affiliation-to-affiliation edge data.
      - Integrating travel-time and proximity indicators between organisations.
      - Applying filtering by edge weight and organisation types.
      - Producing a feature-rich dataset suitable for ZNIB model fitting.
      - Supporting subsequent model estimation, diagnostics, and interpretation.

    Inherits from:
        AffiliationNetworkProcessor: Provides the base methods and attributes
        for managing the affiliation graph and edge data structures.
    """

    _enriched_edges: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    @property
    def enriched_edges(self) -> Optional[pd.DataFrame]:
        return self._enriched_edges

    @enriched_edges.setter
    def enriched_edges(self, value: Optional[pd.DataFrame]) -> None:
        self._enriched_edges = value

    @get_execution_time
    def enrich_edges_with_org_info(
        self,
        route_df: pd.DataFrame,
        min_edge_weight: Optional[int] = None,
        org_type_list: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Creates the base edge-level DataFrame if not cached and the input DataFrame for the
        ZNIB (Zero-Inflated Negative Binomial) model.

        This function prepares and enriches edge-level data to serve as input for
        a ZNIB gravity model of multi-affiliation collaboration intensity.
        It performs the following steps:

          1. Generates all affiliation-to-affiliation edge combinations.
          2. Merges travel-time information for each edge of the two affiliations (from → to).
          3. Applies a binary proximity indicator variable to represent organisational
             proximity or distance between affiliations.
          4. Adds gravity model–related variables such as log-transformed
             article counts and travel durations.

        :param route_df:
            DataFrame containing travel time information between affiliation pairs.
        :param min_edge_weight:
            The minimum link strength to retain. Edges with lower weights are removed from
            the graph. If ``None``, all edges are included.
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

        self._enrich_edges_with_org_info(
            route_df=route_df, org_type_list=org_type_list, min_edge_weight=min_edge_weight
        )
        return self.enriched_edges

    def _enrich_edges_with_org_info(
        self,
        route_df: pd.DataFrame,
        org_type_list: Optional[List[str]],
        min_edge_weight: Optional[int] = None,
    ) -> None:

        self._create_affiliation_graph(min_edge_weight=min_edge_weight)

        if self.enriched_edges is None:
            self.enriched_edges = enrich_edges_with_org_info(
                edge_gdf=self.edge.edge_gdf, route_df=route_df, org_type_list=org_type_list
            )

    @get_execution_time
    def fit_znib(
        self, config: ZINBConfig, enriched_edge_df: Optional[pd.DataFrame] = None
    ) -> ZINBModel:
        """
        Fits a Zero-Inflated Negative Binomial (ZINB) model on the enriched edge data.

        This method creates and fits a `ZINBModel` instance using either the internally stored
        enriched edges or an optionally provided edge DataFrame. It returns the fitted model
        object for further inspection or analysis.

        :param config:
            ZINB configuration object specifying the dependent variable, predictor variables,
            zero-inflation specification, and other model parameters to be used in fitting
            the model.
        :param enriched_edge_df:
            The enriched edge dataframe.
        :return:
            The fitted ZNIB model.
        """

        if enriched_edge_df is not None:
            self.enriched_edges = enriched_edge_df

        znib_model = ZINBModel(df=self.enriched_edges, config=config)
        znib_model.fit()
        return znib_model
