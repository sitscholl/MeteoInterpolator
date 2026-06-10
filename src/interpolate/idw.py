import logging

import numpy as np
import xarray as xr

from .distance import DistanceField

logger = logging.getLogger(__name__)

class InverseDistanceWeighting:
    def __init__(
        self,
        neighbours: int = 4,
        weight_exponent: float = 2.0,
        ):
        if neighbours < 1:
            raise ValueError(f"neighbours must be >= 1. Got {neighbours}")
        if weight_exponent <= 0:
            raise ValueError(f"weight_exponent must be > 0. Got {weight_exponent}")
        self.neighbours = neighbours
        self.weight_exponent = float(weight_exponent)

    def _validate_input_array(self, array: xr.DataArray):
        if not isinstance(array, xr.DataArray):
            raise ValueError(f'Input should be an xarray DataArray. Got {type(array)}')
        if 'id' not in array.dims or array.sizes['id'] == 0:
            raise ValueError(f"Input must have an id dimension with length > 0. Got {array.dims}")
        if array.ndim != 1:
            raise ValueError(f"Input residuals must be one-dimensional over id. Got {array.dims}")

    def interpolate(self, y: xr.DataArray, distance_field: xr.DataArray):
        
        self._validate_input_array(y)
        self._validate_input_weights(distance_field)

        distance_ids = set(distance_field.id.values)
        common_ids = [pid for pid in y.id.values if pid in distance_ids]
        if len(common_ids) == 0:
            logger.warning("No common ids between y input and distance_fields array. y-values cannot be interpolated")
            return None

        missing_ids = set(y.id.values) - set(distance_field.id.values)
        if len(missing_ids) > 0:
            logger.warning(f"No distance fields for the following ids were provided. They will not be considered in the interpolation: {missing_ids}")
        
        y = y.sel(id = common_ids)
        distance_field = distance_field.sel(id = common_ids)

        output_dims = tuple(dim for dim in distance_field.dims if dim != "id")
        distances = distance_field.transpose("id", *output_dims)
        distance_values = np.asarray(distances.values, dtype=float)
        residual_values = np.asarray(y.values, dtype=float)

        n_stations = len(common_ids)
        n_neighbours = min(self.neighbours, n_stations)
        broadcast_shape = (n_stations,) + (1,) * (distance_values.ndim - 1)
        residual_values = residual_values.reshape(broadcast_shape)

        zero_distance = np.isfinite(distance_values) & (distance_values == 0)
        zero_count = zero_distance.sum(axis=0)
        exact_values = np.divide(
            (zero_distance * residual_values).sum(axis=0),
            zero_count,
            out=np.full(zero_count.shape, np.nan, dtype=float),
            where=zero_count > 0,
        )

        valid_distances = np.where(
            np.isfinite(distance_values) & (distance_values > 0),
            distance_values,
            np.inf,
        )
        nearest_indices = np.argpartition(valid_distances, kth=n_neighbours - 1, axis=0)[:n_neighbours]
        nearest_distances = np.take_along_axis(valid_distances, nearest_indices, axis=0)
        nearest_residuals = np.take_along_axis(
            np.broadcast_to(residual_values, distance_values.shape),
            nearest_indices,
            axis=0,
        )

        valid_neighbours = np.isfinite(nearest_distances)
        weights = np.zeros_like(nearest_distances, dtype=float)
        weights[valid_neighbours] = nearest_distances[valid_neighbours] ** (-self.weight_exponent)

        numerator = (nearest_residuals * weights).sum(axis=0)
        denominator = weights.sum(axis=0)
        interpolated = np.divide(
            numerator,
            denominator,
            out=np.full(denominator.shape, np.nan, dtype=float),
            where=denominator > 0,
        )
        interpolated = np.where(zero_count > 0, exact_values, interpolated)

        coords = {dim: distances.coords[dim] for dim in output_dims if dim in distances.coords}
        return xr.DataArray(
            interpolated,
            dims=output_dims,
            coords=coords,
            name="idw_residual",
            attrs={
                "description": "Inverse distance weighted residual interpolation.",
                "neighbours": n_neighbours,
                "weight_exponent": self.weight_exponent,
                "distance_type": distance_field.attrs.get("distance_type"),
            },
        )
