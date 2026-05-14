import numpy as np
from scipy.optimize import curve_fit

import logging

from .base import BaseVerticalModel
from .linear import LinearVerticalModel

logger = logging.getLogger(__name__)

class NonLinearVerticalModel(BaseVerticalModel):
    def __init__(
        self,
        fit_method: str = 'trf',
        fit_initial_params: bool = True,
        initial_params_elev_threshold: float = 1500,
        optimized_parameters: list[float] | None = None
        ):
        self.fit_method = fit_method
        self.fit_initial_params = fit_initial_params
        self.initial_params_elev_threshold = initial_params_elev_threshold
        self.optimized_parameters = optimized_parameters

    @classmethod
    def key(cls):
        return "non-linear"

    @staticmethod
    def _calculate_nonlinear_profile(elevation, t0, gamma, a, h0, h1):
        
        res = (t0 - gamma * elevation) * (elevation >= h1) +\
            (t0 - gamma * elevation - (a/2) * (1 + np.cos(np.pi * (elevation - h0) / (h1 - h0)))) * ((elevation < h1) & (elevation > h0)) +\
            (t0 - gamma * elevation - a) * (elevation <= h0)
        
        return(res)

    def fit(self, X, y):

        if (np.isnan(y).any()) or (np.isnan(X).any()):
            logger.warning('Input contains nan values!')

        if self.fit_initial_params:
            ##Get initial parameters from linear regression based on high-elevation stations
            X_idx = X > self.initial_params_elev_threshold
            lreg = LinearVerticalModel().fit(X[X_idx].reshape(-1, 1), y[X_idx])
            params = np.array([lreg.intercept_, lreg.coef_[0]*-1, 0, 500, 1500]) #t0, gamma, a, h0, h1
        else:
            params = [np.nan] * 5

        sigma = np.ones_like(X)
        popt, pcov = curve_fit(
            self._calculate_nonlinear_profile, 
            X, y, 
            params, 
            method = self.fit_method, 
            sigma = sigma, 
            maxfev=5000
            )

        self.optimized_parameters = popt
        return self

    def _predict_numpy_2d(self, X):
        if self.optimized_parameters is None:
            raise ValueError("Fit model first before calling predict")
        return self._calculate_nonlinear_profile(X, self.optimized_parameters)