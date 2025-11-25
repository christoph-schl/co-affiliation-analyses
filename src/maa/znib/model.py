from typing import Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels import api as sm
from statsmodels.discrete.count_model import (
    ZeroInflatedNegativeBinomialP,
    ZeroInflatedNegativeBinomialResultsWrapper,
)
from statsmodels.iolib.summary import Summary

from maa.znib.configuration import ZINBConfig
from maa.znib.utils import ZNIBInput, model_results_to_df


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
