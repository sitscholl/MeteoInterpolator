from abc import ABC, abstractmethod

import numpy as np
import xarray as xr


def _validate_dimension(data, dim):
    """Require an explicit, nonempty dimension index with unambiguous labels."""
    if dim not in data.dims or data.sizes[dim] == 0:
        raise ValueError(f"Expected a nonempty '{dim}' dimension.")
    if dim not in data.coords or data.coords[dim].dims != (dim,):
        raise ValueError(f"Expected an explicit '{dim}' coordinate over '{dim}'.")
    index = data.get_index(dim)
    if not index.is_unique or index.hasnans:
        raise ValueError(f"Coordinate '{dim}' must contain unique, nonmissing labels.")


def _validate_numeric(array, *, allow_nan=False, allow_inf=False):
    if array.dtype.kind not in "iuf":
        raise TypeError(f"'{array.name}' must contain real numeric values.")
    values = array.values
    if (not allow_nan and np.isnan(values).any()) or (not allow_inf and np.isinf(values).any()):
        raise ValueError(f"'{array.name}' contains unsupported non-finite values.")


def _validate_predictors(data, required_predictors, dims, *, allow_nan=False):
    for name in required_predictors:
        if name not in data.data_vars:
            raise ValueError(f"Missing required predictor '{name}'.")
        if set(data[name].dims) != set(dims):
            raise ValueError(f"Predictor '{name}' must have dimensions {dims}.")
        _validate_numeric(data[name], allow_nan=allow_nan)


def _target_dims(targets):
    if set(targets.dims) == {"target_id"}:
        return ("target_id",)
    if set(targets.dims) == {"y", "x"}:
        return ("y", "x")
    raise ValueError("Targets must have dimensions ('target_id',) or ('y', 'x').")


def _validate_target_coordinates(data, targets):
    """Compare labels and point locations without implicit xarray alignment."""
    for name in dict.fromkeys((*_target_dims(targets), "x", "y")):
        if name not in data.coords:
            raise ValueError(f"Missing target coordinate '{name}'.")
        actual, expected = data.coords[name], targets.coords[name]
        if not actual.variable.equals(expected.variable):
            raise ValueError(f"Target coordinate '{name}' does not match targets.")


class FittedInterpolationModel(ABC):
    required_predictors = ()

    @staticmethod
    def _validate_targets(targets: xr.Dataset, required_predictors = ()):
        if not isinstance(targets, xr.Dataset):
            raise TypeError(f"targets input should be an xarray Dataset. Got {type(targets)}")
        
        dims = _target_dims(targets)
        for dim in dims:
            _validate_dimension(targets, dim)
        for name in ("x", "y"):
            expected_dims = ("target_id",) if dims == ("target_id",) else (name,)
            if name not in targets.coords or targets.coords[name].dims != expected_dims:
                raise ValueError(f"Target coordinate '{name}' must have dimensions {expected_dims}.")
            _validate_numeric(targets.coords[name])
        _validate_predictors(targets, required_predictors, dims, allow_nan=True)
        return dims

    @staticmethod
    def _validate_distances(distances: xr.DataArray, source_ids, targets):
        """Return distances in fitted-source order and canonical target order.

        Extra cached sources are allowed; missing fitted sources are errors.
        NaN and positive infinity represent unavailable distances.
        """
        if not isinstance(distances, xr.DataArray):
            raise TypeError(f"distances must be an xarray DataArray. Got {type(distances)}")

        dims = FittedInterpolationModel._validate_targets(targets)
        if set(distances.dims) != {"id", *dims}:
            raise ValueError(f"Distances must have dimensions {('id', *dims)}; select lambda/time first.")

        _validate_dimension(distances, "id")
        _validate_target_coordinates(distances, targets)

        source_ids = list(source_ids)
        if not source_ids:
            raise ValueError("At least one fitted source ID is required.")

        missing = [source_id for source_id in source_ids if source_id not in distances.get_index("id")]
        if missing:
            raise ValueError(f"Distances are missing fitted source IDs: {missing}")

        distances = distances.sel(id=source_ids).transpose("id", *dims)
        _validate_numeric(distances, allow_nan=True, allow_inf=True)
        if (distances.values < 0).any():
            raise ValueError("Distances must be nonnegative.")

        return distances

    @staticmethod
    def _validate_predictions(predictions, targets):
        if not isinstance(predictions, xr.DataArray):
            raise TypeError("Predictions must be an xarray DataArray.")
        dims = _target_dims(targets)
        if set(predictions.dims) != set(dims):
            raise ValueError(f"Predictions must have target dimensions {dims}.")
        _validate_target_coordinates(predictions, targets)
        _validate_numeric(predictions, allow_nan=True)
        return predictions.transpose(*dims)

    @abstractmethod
    def _predict(
        self, targets: xr.Dataset, distances: xr.DataArray | None = None,
    ) -> xr.DataArray:
        pass

    def predict(
        self, targets: xr.Dataset, distances: xr.DataArray | None = None,
    ) -> xr.DataArray:
        self._validate_targets(targets, self.required_predictors)
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
        """Validate finite training samples and align X to y's existing ID order."""
        if not isinstance(X, xr.Dataset):
            raise TypeError(f'X input should be an xarray Dataset. Got {type(X)}')
        if not isinstance(y, xr.DataArray):
            raise TypeError(f"y input should be an xarray DataArray. Got {type(y)}")
        if set(X.dims) != {"id"} or y.dims != ("id",):
            raise ValueError("Training X and y must have only the 'id' dimension.")
        _validate_dimension(X, "id")
        _validate_dimension(y, "id")
        if len(X.id) != len(y.id) or not y.get_index("id").isin(X.get_index("id")).all():
            raise ValueError("X and y must contain exactly the same source IDs.")
        _validate_numeric(y)
        _validate_predictors(X, required_predictors, ("id",))
        return X.sel(id=y.id), y

    @abstractmethod
    def fit(self, X: xr.Dataset, y: xr.DataArray) -> FittedInterpolationModel:
        pass
