import numpy as np
import xarray as xr
from abc import ABC, abstractmethod

class BaseVerticalModel(ABC):
    registry: dict[str, type["BaseVerticalModel"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls is not BaseVerticalModel:
            BaseVerticalModel.registry[cls.key()] = cls

    @classmethod
    @abstractmethod
    def key(cls) -> str:
        pass

    @classmethod
    def create(cls, key: str, **kwargs):
        model_cls = cls.registry.get(key)
        if model_cls is None:
            available = ", ".join(sorted(cls.registry)) or "none"
            raise ValueError(f"Unknown vertical model '{key}'. Available: {available}")
        return model_cls(**kwargs)

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
