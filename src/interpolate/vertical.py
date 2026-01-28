from sklearn.linear_model import LinearRegression
import numpy as np
import xarray as xr

class VerticalModel:
    def __init__(self, **kwargs):
        self._params = dict(kwargs)
        self._model = LinearRegression(**kwargs)

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray | xr.DataArray):
        
        if isinstance(X, np.ndarray):
            return self._predict_numpy(X)
        elif isinstance(X, xr.DataArray):
            return self._predict_dataarray(X)
        else:
            raise ValueError(f"X must be either a numpy array or a DataArray. Got {type(X)}")

    def _predict_numpy(self, X):
        X = np.asarray(X)
        if X.ndim == 0:
            X_2d = X.reshape(1, 1)
            return self._model.predict(X_2d).reshape(())
        if X.ndim == 1:
            X_2d = X.reshape(-1, 1)
            return self._model.predict(X_2d).reshape(X.shape)
        if X.ndim == 2 and X.shape[1] == 1:
            return self._model.predict(X).reshape(X.shape)
        if X.ndim >= 2 and X.shape[-1] == 1:
            X_2d = X.reshape(-1, 1)
            return self._model.predict(X_2d).reshape(X.shape)
        raise ValueError(
            "VerticalModel expects a single feature per sample. "
            f"Got array with shape {X.shape}."
        )

    def _predict_dataarray(self, X):
        data = self._predict_numpy(X.values)
        return xr.DataArray(
            data = data,
            coords = X.coords,
            dims = X.dims,
            name = X.name,
            attrs = X.attrs,
        )

    def get_params(self):
        return dict(self._params)

    def set_params(self, **kwargs):
        self._params.update(kwargs)
        self._model.set_params(**kwargs)
        return self
