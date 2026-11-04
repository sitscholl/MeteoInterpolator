from abc import ABC, abstractmethod

import xarray as xr


class FittedInterpolationModel(ABC):

    @staticmethod
    def _validate_targets(targets: xr.Dataset, required_predictors = ()):
        if not isinstance(targets, xr.Dataset):
            raise TypeError(f"targets input should be an xarray Dataset. Got {type(targets)}")
        
        invalid_dims = [i for i in targets.dims if i not in ['x', 'y']]
        if invalid_dims:
            raise ValueError(f"Targets array should have only x and y dimensions. Got {targets.dims}")

        missing_dims = [i for i in ['x', 'y'] if i not in targets.dims]
        if missing_dims:
            raise ValueError(f"Targets array require both 'x' and 'y' dimensions. Got {targets.dims}")

        if len(required_predictors) > 0:
            missing_predictors = [i for i in required_predictors if i not in targets.data_vars]
            if missing_predictors:
                raise ValueError(f"X input is missing the following data variables: {missing_predictors}")

    @staticmethod
    def _validate_distances(distances: xr.DataArray, source_ids, targets):
        if not isinstance(distances, xr.DataArray):
            raise TypeError(f"distances must be an xarray DataArray. Got {type(distances)}")
        if "id" not in distances.dims or distances.sizes["id"] == 0:
            raise ValueError(f"distance_field must have an id dimension with length > 0. Got {distances.dims}")
        if distances.ndim < 2:
            raise ValueError(f"distance_field must include at least one target dimension besides id. Got {distances.dims}")
        if "lam_value" in distances.dims:
            raise ValueError("Expected a distance field for a single lambda value.")

    @staticmethod
    def _validate_predictions(predictions, targets):
        pass

    @abstractmethod
    def _predict(
        self, targets: xr.Dataset, distances: xr.DataArray | None = None,
    ) -> xr.DataArray:
        pass

    def predict(
        self, targets: xr.Dataset, distances: xr.DataArray | None = None,
    ) -> xr.DataArray:
        preds = self._predict(targets, distances)
        preds = self._validate_predictions(preds, targets)
        return preds


class InterpolationModel(ABC):
    registry: dict[str, type["InterpolationModel"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls is not InterpolationModel:
            InterpolationModel.registry[cls.key()] = cls

    @classmethod
    @abstractmethod
    def key(cls) -> str:
        pass

    @property
    def requires_distances(self):
        return False

    @classmethod
    def create(cls, key: str, **kwargs):
        model_cls = cls.registry.get(key)
        if model_cls is None:
            available = ", ".join(sorted(cls.registry)) or "none"
            raise ValueError(f"Unknown vertical model '{key}'. Available: {available}")
        return model_cls(**kwargs)

    @staticmethod
    def _validate_training_data(X, y, required_predictors = ()):
        if not isinstance(X, xr.Dataset):
            raise TypeError(f'X input should be an xarray Dataset. Got {type(X)}')
        if 'id' not in X.dims or X.sizes['id'] == 0:
            raise ValueError(f"X input must have an id dimension with length > 0. Got {X.dims}")

        if len(required_predictors) > 0:
            missing_predictors = [i for i in required_predictors if i not in X.data_vars]
            if missing_predictors:
                raise ValueError(f"X input is missing the following data variables: {missing_predictors}")

        if not isinstance(y, xr.DataArray):
            raise TypeError(f"y input should be an xarray DataArray. Got {type(y)}")
        if y.ndim != 1:
            raise ValueError(f"y input residuals must be one-dimensional over id. Got {y.dims}")

        y = y.sortby('id')
        return X.reindex(y.id), y

    @abstractmethod
    def fit(self, X: xr.Dataset, y: xr.DataArray) -> FittedInterpolationModel:
        pass
