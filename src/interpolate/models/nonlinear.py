import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from .base import FittedInterpolationModel, InterpolationModel
from .linear import LinearLapseRateModel

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class NonLinearProfileParams:
    t0: float
    gamma: float
    a: float
    h0: float
    h1: float

    def as_array(self) -> np.ndarray:
        return np.array([self.t0, self.gamma, self.a, self.h0, self.h1], dtype=float)

    @classmethod
    def from_array(cls, values):
        t0, gamma, a, h0, h1 = values
        return cls(float(t0), float(gamma), float(a), float(h0), float(h1))

    @classmethod
    def from_config(cls, values):
        if values is None or isinstance(values, cls):
            return values
        if isinstance(values, Mapping):
            return cls(**values)
        if isinstance(values, Sequence) and not isinstance(values, str):
            return cls.from_array(values)
        raise TypeError(f"Invalid nonlinear profile parameters: {type(values)}")

def _calculate_nonlinear_profile(elevation, t0, gamma, a, h0, h1):
    elevation = np.asarray(elevation, dtype=float)
    denominator = h1 - h0
    if np.isclose(denominator, 0.0):
        denominator = np.finfo(float).eps

    above = elevation >= h1
    transition = (elevation < h1) & (elevation > h0)
    below = elevation <= h0

    return (
        (t0 - gamma * elevation) * above
        + (
            t0
            - gamma * elevation
            - (a / 2) * (1 + np.cos(np.pi * (elevation - h0) / denominator))
        )
        * transition
        + (t0 - gamma * elevation - a) * below
    )

@dataclass(frozen=True)
class FittedNonLinearLapseRateModel(FittedInterpolationModel):
    required_predictors = ('elevation',)

    parameters: NonLinearProfileParams

    def _predict(self, targets: xr.Dataset, distances=None):
        X = np.asarray(targets['elevation'].values)
        original_shape = X.shape
        X_2d = X.reshape(-1, 1)

        preds = _calculate_nonlinear_profile(
            X_2d,
            self.parameters.t0,
            self.parameters.gamma,
            self.parameters.a,
            self.parameters.h0,
            self.parameters.h1,
        )
        preds = preds.reshape(original_shape)

        return xr.DataArray(
            data=preds,
            coords=targets['elevation'].coords,
            dims=targets['elevation'].dims,
            name="NonLinearLapseRate predictions",
            attrs={
                "description": "Non linear lapse rate predictions.",
                "t0": self.parameters.t0,
                "gamma": self.parameters.gamma,
                "a": self.parameters.a,
                "h0": self.parameters.h0,
                "h1": self.parameters.h1,
            },
        )

class NonLinearLapseRateModel(InterpolationModel):
    def __init__(
        self,
        fit_method: str = "trf",
        fit_initial_params: bool = True,
        initial_params_elev_threshold: float = 1500,
        initial_parameters: NonLinearProfileParams | None = None,
        optimized_parameters: NonLinearProfileParams | None = None
        ):
        self.fit_method = fit_method
        self.fit_initial_params = fit_initial_params
        self.initial_params_elev_threshold = initial_params_elev_threshold
        self.initial_parameters = NonLinearProfileParams.from_config(initial_parameters)
        self.optimized_parameters = NonLinearProfileParams.from_config(optimized_parameters)

    @classmethod
    def key(cls):
        return "nonlinear_lapse_rate"

    def _get_initial_params(self, X, y):
        if self.initial_parameters is not None:
            return self.initial_parameters

        if not self.fit_initial_params:
            return NonLinearProfileParams(
                t0 = np.nanmean(y),
                gamma = 0,
                a = 0,
                h0 = 500,
                h1 = 1500
            )

        elevation = X["elevation"]
        high_ids = elevation.id.where(
            elevation > self.initial_params_elev_threshold,
            drop=True,
        )
        if high_ids.size < 2:
            logger.warning(
                "Fewer than two stations above %.1f m. Using all stations for nonlinear initial parameters.",
                self.initial_params_elev_threshold,
            )
        fit_ids = high_ids if high_ids.size >= 2 else elevation.id

        linear_fit = LinearLapseRateModel().fit(
            X.sel(id=fit_ids),
            y.sel(id=fit_ids),
        )

        t0 = float(linear_fit.intercept)
        gamma = float(-linear_fit.coef[0])
        return NonLinearProfileParams(t0, gamma, 0.0, 500.0, 1500.0)

    def fit(self, X: xr.Dataset, y: xr.DataArray):
        X, y = self._validate_training_data(X, y, required_predictors=('elevation',))

        if X['elevation'].size < 5:
            raise ValueError(f"Nonlinear lapse-rate model requires at least five valid samples. Got {x_arr.size}.")
        
        params = self._get_initial_params(X, y)
        
        y = np.asarray(y, dtype=float).reshape(-1)
        sigma = np.ones_like(y)

        ##todo: Handle failed fit. Raise error? or return None?
        x_arr = np.asarray(X['elevation'], dtype=float).reshape(-1)
        popt, pcov = curve_fit(
            _calculate_nonlinear_profile, 
            x_arr, y, 
            params.as_array(), 
            method = self.fit_method, 
            sigma = sigma, 
            maxfev=5000
            )

        return FittedNonLinearLapseRateModel(parameters=NonLinearProfileParams.from_array(popt))

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)

    # Synthetic stations spanning a valley floor, inversion layer, and free atmosphere.
    X = np.array(
        [240, 360, 520, 690, 830, 980, 1160, 1340, 1510, 1730, 1980, 2260, 2580, 2950, 3380],
        dtype=float,
    )
    true_params = NonLinearProfileParams(
        t0=5.0,
        gamma=0.0055,
        a=4.5,
        h0=650.0,
        h1=1750.0,
    )
    y_true = _calculate_nonlinear_profile(
        elevation=X,
        t0=true_params.t0,
        gamma=true_params.gamma,
        a=true_params.a,
        h0=true_params.h0,
        h1=true_params.h1,
    )
    y = y_true + rng.normal(loc=0.0, scale=0.35, size=X.shape)

    model = NonLinearLapseRateModel()

    fit = model.fit(X, y)

    x_pred = np.arange(np.min(X), np.max(X), step = 1)
    preds = fit.predict(x_pred)

    fig, ax = plt.subplots()
    ax.scatter(y, X, label="Synthetic stations", color="tab:blue")
    ax.plot(
        _calculate_nonlinear_profile(x_pred, *true_params.as_array()),
        x_pred,
        label="True profile",
        color="tab:green",
    )
    ax.plot(preds, x_pred, label="Fitted profile", color="tab:orange")
    ax.set_ylabel("Elevation [m]")
    ax.set_xlabel("Temperature [deg C]")
    ax.legend()
