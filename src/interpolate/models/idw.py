import logging
from dataclasses import dataclass

import numpy as np
import xarray as xr

from .base import FittedInterpolationModel, InterpolationModel

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class FittedIDWModel(FittedInterpolationModel):
    y: xr.DataArray
    neighbours: int
    weight_exponent: float
    
    def _predict(self, targets: xr.Dataset, distances: xr.DataArray | None = None):

        ##Filters for source ids and raises on missing ids
        distances = self._validate_distances(distances, self.y.id.values, targets)

        distance_values = np.asarray(distances.values, dtype=float)
        residual_values = np.asarray(self.y.values, dtype=float)

        n_stations = len(distances.id.values)
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

        return xr.DataArray(
            interpolated,
            dims=targets.dims,
            coords=targets.coors,
            name="IDWModel predictions",
            attrs={
                "description": "Inverse distance weighted predictions.",
                "neighbours": n_neighbours,
                "weight_exponent": self.weight_exponent,
                "distance_type": distances.attrs.get("distance_type"),
            },
        )

class IDWModel(InterpolationModel):

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

    @classmethod
    def key(cls) -> str:
        return "inverse distance weighting"

    @property
    def requires_distances(self):
        return True

    def fit(self, X: xr.Dataset, y: xr.DataArray) -> FittedIDWModel:
        
        X, y = self._validate_training_data(X, y)

        return FittedIDWModel(
            y = y.copy(),
            neighbours = self.neighbours,
            weight_exponent = self.weight_exponent,
        )


