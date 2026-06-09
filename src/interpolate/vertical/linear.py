from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression

from .base import BaseFittedVerticalModel, BaseVerticalModel

@dataclass(frozen=True)
class LinearVerticalFit(BaseFittedVerticalModel):
    coef: tuple[float, ...]
    intercept: float

    def _predict_numpy_2d(self, X):
        coef = np.asarray(self.coef, dtype=float)
        return X @ coef + self.intercept

class LinearVerticalModel(BaseVerticalModel):
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)

    @classmethod
    def key(cls):
        return "linear"

    def fit(self, X, y):
        estimator = LinearRegression(**self.kwargs)
        estimator.fit(X, y)
        return LinearVerticalFit(
            coef=tuple(float(value) for value in np.ravel(estimator.coef_)),
            intercept=float(estimator.intercept_),
        )
