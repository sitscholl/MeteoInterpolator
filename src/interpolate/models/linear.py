from dataclasses import dataclass

import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression

from .base import FittedInterpolationModel, InterpolationModel


@dataclass(frozen=True)
class FittedLinearLapseRateModel(FittedInterpolationModel):
    coef: tuple[float, ...]
    intercept: float

    def _predict(self, targets: xr.Dataset, distances=None):
        if not isinstance(targets, xr.Dataset):
            raise TypeError(f"X must be an xarray Dataset. Got {type(targets)}")
        if not 'elevation' in targets.data_vars:
            raise ValueError(f"LinearLapseRateModel Targets require a variable called 'elevation'. Got {targets.data_vars}")
        
        X = np.asarray(targets['elevation'].values)
        original_shape = X.shape
        X_2d = X.reshape(-1, 1)

        coef = np.asarray(self.coef, dtype=float)
        preds = (X_2d @ coef + self.intercept).reshape(original_shape)

        return xr.DataArray(
            data=preds,
            coords=targets.coords,
            dims=targets.dims,
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
        return "linear lapse-rate model"

    def fit(self, X: xr.Dataset, y: xr.DataArray):
        X, y = self._validate_training_data(X, y, required_predictors=('elevation',))

        estimator = LinearRegression(**self.kwargs)
        estimator.fit(X['elevation'], y)
        return FittedLinearLapseRateModel(
            coef=tuple(float(value) for value in np.ravel(estimator.coef_)),
            intercept=float(estimator.intercept_),
        )
