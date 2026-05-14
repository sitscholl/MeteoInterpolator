import numpy as np
import xarray as xr
from abc import ABC, abstractmethod

class BaseVerticalModel(ABC):

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def _predict_numpy_2d(self, X: np.ndarray) -> np.ndarray:
        pass

    def predict(self, X: np.ndarray | xr.DataArray):
        if isinstance(X, xr.DataArray):
            data = self._predict_numpy(X.values)
            return xr.DataArray(
                data=data,
                coords=X.coords,
                dims=X.dims,
                name=X.name,
                attrs=X.attrs,
            )

        if isinstance(X, np.ndarray):
            return self._predict_numpy(X)

        raise TypeError(f"X must be a numpy array or xarray DataArray. Got {type(X)}")

    def _predict_numpy(self, X):
        X = np.asarray(X)
        original_shape = X.shape
        X_2d = X.reshape(-1, 1)
        return self._predict_numpy_2d(X_2d).reshape(original_shape)
