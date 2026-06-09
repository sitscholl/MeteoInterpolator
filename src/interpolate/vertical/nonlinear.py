import numpy as np
from scipy.optimize import curve_fit

import logging
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

from .base import BaseVerticalModel
from .linear import LinearVerticalModel

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

class NonLinearVerticalModel(BaseVerticalModel):
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
        return "nonlinear"

    @staticmethod
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

        high_elevation = X > self.initial_params_elev_threshold
        fit_X = X[high_elevation]
        fit_y = y[high_elevation]
        if fit_X.size < 2:
            logger.warning(
                "Fewer than two stations above %.1f m. Using all stations for nonlinear initial parameters.",
                self.initial_params_elev_threshold,
            )
            fit_X = X
            fit_y = y

        linear_model = LinearVerticalModel().fit(fit_X.reshape(-1, 1), fit_y)
        t0 = float(linear_model.estimator.intercept_)
        gamma = float(-linear_model.estimator.coef_[0])
        return NonLinearProfileParams(t0, gamma, 0.0, 500.0, 1500.0)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must contain the same number of samples. Got {X.shape[0]} and {y.shape[0]}.")

        valid = np.isfinite(X) & np.isfinite(y)
        if not valid.all():
            logger.warning("Input contains non-finite values. Dropping invalid samples before fitting.")
            X = X[valid]
            y = y[valid]

        if X.size < 5:
            raise ValueError(f"Nonlinear vertical model requires at least five valid samples. Got {X.size}.")

        params = self._get_initial_params(X, y)
        sigma = np.ones_like(y)

        ##todo: Handle failed fit. Raise error? or return None?
        popt, pcov = curve_fit(
            self._calculate_nonlinear_profile, 
            X, y, 
            params.as_array(), 
            method = self.fit_method, 
            sigma = sigma, 
            maxfev=5000
            )

        self.optimized_parameters = NonLinearProfileParams.from_array(popt)
        return self

    def _predict_numpy_2d(self, X):
        if self.optimized_parameters is None:
            raise ValueError("Fit model first before calling predict")
        params = self.optimized_parameters
        return self._calculate_nonlinear_profile(
            X,
            params.t0,
            params.gamma,
            params.a,
            params.h0,
            params.h1,
        )

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
    y_true = NonLinearVerticalModel._calculate_nonlinear_profile(
        elevation=X,
        t0=true_params.t0,
        gamma=true_params.gamma,
        a=true_params.a,
        h0=true_params.h0,
        h1=true_params.h1,
    )
    y = y_true + rng.normal(loc=0.0, scale=0.35, size=X.shape)

    model = NonLinearVerticalModel()

    model.fit(X, y)

    x_pred = np.arange(np.min(X), np.max(X), step = 1)
    preds = model.predict(x_pred)

    fig, ax = plt.subplots()
    ax.scatter(y, X, label="Synthetic stations", color="tab:blue")
    ax.plot(
        NonLinearVerticalModel._calculate_nonlinear_profile(x_pred, *true_params.as_array()),
        x_pred,
        label="True profile",
        color="tab:green",
    )
    ax.plot(preds, x_pred, label="Fitted profile", color="tab:orange")
    ax.set_ylabel("Elevation [m]")
    ax.set_xlabel("Temperature [deg C]")
    ax.legend()
