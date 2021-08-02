import numpy as np
import pandas as pd

# from sklearn.preprocessing import (
#     OrdinalEncoder,
#     OneHotEncoder
# )

from feature_engine.encoding import OrdinalEncoder
from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union


class TransformerWrapper(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None, transformer=None):

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.transformer = transformer

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.transformer.fit(X[self.variables])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variables] = self.transformer.transform(X[self.variables])
        return X


class PipelineModel(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None, model=None):

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.model = model

    def fit(self, X: Union[pd.DataFrame, pd.Series, np.ndarray],
                  y: Union[pd.Series, np.ndarray, list] = None):

        if type(X) == pd.DataFrame:
            self.model.fit(X[self.variables], y)
        return self

    def transform(self, X, y=None):
        return X

    def predict(self, X):

        return self.model.predict(X[self.variables])


def ml_pipeline(*, model, features, category_variable):
    price_pipe = Pipeline(

        [
            # (
            #     "categorical_encoder",
            #     TransformerWrapper(
            #         variables=category_variable,
            #         transformer=OrdinalEncoder(),
            #     ),
            (
                "categorical_encoder",
                OrdinalEncoder(
                    encoding_method='ordered',
                    variables=category_variable,
                ),
            ),
            (
                "model",
                PipelineModel(
                    variables=features,
                    model=model
                ),
            ),



        ]
    )

    return price_pipe

