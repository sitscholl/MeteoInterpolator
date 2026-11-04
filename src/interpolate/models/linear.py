from dataclasses import dataclass

import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression

from .base import FittedInterpolationModel, InterpolationModel


@dataclass(frozen=True)
class FittedLinearLapseRateModel(FittedInterpolationModel):
    required_predictors = ('elevation',)

    coef: tuple[float, ...]
    intercept: float

    def _predict(self, targets: xr.Dataset, distances=None):
        X = np.asarray(targets['elevation'].values)
        original_shape = X.shape
        X_2d = X.reshape(-1, 1)

        coef = np.asarray(self.coef, dtype=float)
        preds = (X_2d @ coef + self.intercept).reshape(original_shape)

        return xr.DataArray(
            data=preds,
            coords=targets['elevation'].coords,
            dims=targets['elevation'].dims,
            name="LinearLapseRate predictions",
            attrs={
                "description": "Linear lapse rate predictions.",
                "coef": self.coef,
                "intercept": self.intercept,
            },
        )

class LinearLapseRateModel(InterpolationModel):
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)

    @classmethod
    def key(cls):
        return "linear_lapse_rate"

    def fit(self, X: xr.Dataset, y: xr.DataArray):
        X, y = self._validate_training_data(X, y, required_predictors=('elevation',))

        estimator = LinearRegression(**self.kwargs)
        estimator.fit(X["elevation"].to_numpy().reshape(-1, 1), y)
        return FittedLinearLapseRateModel(
            coef=tuple(float(value) for value in np.ravel(estimator.coef_)),
            intercept=float(estimator.intercept_),
        )
