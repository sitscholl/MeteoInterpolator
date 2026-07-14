import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401

from dataclasses import dataclass
import logging

from .vertical import BaseFittedVerticalModel, BaseVerticalModel
from .distance import DistanceField
from .idw import InverseDistanceWeighting
from .regions import InterpolationRegions

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = ['elevation', 'x', 'y', 'station_id']

@dataclass(frozen=True)
class InterpolationJob:
    timestamp: pd.Timestamp
    parameter: str
    observations: pd.DataFrame
    training_points: gpd.GeoDataFrame

    @property
    def required_columns(self):
        return [self.parameter, *_REQUIRED_COLUMNS]

    def __post_init__(self):
        for req_col in self.required_columns:
            if req_col not in self.observations.columns:
                raise ValueError(f"InterpolationJob observations is missing required column {req_col}. Got {self.observations.columns}")
            if self.observations[req_col].isna().any():
                raise ValueError(f"Found NaN values in InterpolationJob observations for column {req_col}")

        if self.observations["station_id"].duplicated().any():
            duplicated = self.observations.loc[self.observations["station_id"].duplicated(), "station_id"].tolist()
            raise ValueError(f"InterpolationJob observations contain duplicated station_id values: {duplicated}")

        missing_coords = set(self.observations['station_id'].unique()).difference(self.training_points.index.values)
        if missing_coords:
            raise ValueError(f"The following stations have no coordinate values in training_points: {missing_coords}")

        if self.training_points.crs is None:
            raise ValueError("InterpolationJob training_points crs cannot be None.")

    def to_arrays(self):
        x_coords = self.training_points.geometry.x.to_numpy(dtype=float)
        y_coords = self.training_points.geometry.y.to_numpy(dtype=float)

        y = self.observations[self.parameter].to_numpy(dtype=float)
        X = self.observations["elevation"].to_numpy(dtype=float).reshape(-1, 1)
        ids = self.observations["station_id"].to_numpy(dtype=str)

        return y, X, x_coords, y_coords, ids

    def __repr__(self):
        return f"InterpolationJob (date: {self.timestamp}, parameter: {self.parameter}, samples: {len(self.observations)})"

@dataclass(frozen=True)
class FittedInterpolator:
    vertical_fit: BaseFittedVerticalModel
    residuals: xr.DataArray
    job: InterpolationJob

@dataclass
class Interpolator:
    vertical_model: BaseVerticalModel
    residual_model: InverseDistanceWeighting | None = None
    regions: InterpolationRegions | None = None
    min_sample_size: int = 3

    @classmethod
    def from_config(
        cls, config: dict,
    ):
        vertical_config = dict(config["vertical_model"])
        vertical_handler = vertical_config.pop("type")
        vertical_model = BaseVerticalModel.create(vertical_handler, **vertical_config)

        idw_config = config.get('inverse_distance_weighting')
        if idw_config is None:
            residual_model = None
        else:
            idw_config = dict(idw_config)
            residual_model = InverseDistanceWeighting(**idw_config)

        ## Interpolation Regions
        region_config = config.get('interpolation_regions')
        interpolation_regions = InterpolationRegions(**region_config) if region_config is not None else None
        if region_config is None:
            logger.info("No interpolation regions specified.")

        min_sample_size = config.get('min_sample_size', 3)

        return cls(
            vertical_model = vertical_model, 
            residual_model = residual_model, 
            regions = interpolation_regions, 
            min_sample_size = min_sample_size
            )

    @staticmethod
    def _select_distance_field(
        distance_fields: DistanceField | xr.DataArray,
        lam_value: float | int | None = None,
    ) -> xr.DataArray:
        if isinstance(distance_fields, DistanceField):
            if distance_fields.data is None:
                raise ValueError("distance_fields.data must not be None for residual interpolation.")
            distance_data = distance_fields.data
        elif isinstance(distance_fields, xr.DataArray):
            distance_data = distance_fields
        else:
            raise ValueError(f"distance_fields must be a DistanceField or xarray DataArray. Got {type(distance_fields)}")

        if "lam_value" in distance_data.dims:
            if lam_value is not None:
                return distance_data.sel(lam_value=lam_value)
            return distance_data.isel(lam_value=0)
        return distance_data

    def _check_grid_alignment(self, grid: xr.DataArray, distance_field: xr.DataArray):
        if not isinstance(distance_field, xr.DataArray):
            raise ValueError(f"distance_field must be an xarray DataArray. Got {type(distance_field)}")

        for coord_name in ("x", "y"):
            if coord_name not in grid.coords:
                raise ValueError(f"Prediction grid is missing coordinate '{coord_name}'.")
            if coord_name not in distance_field.coords:
                raise ValueError(f"Distance fields are missing coordinate '{coord_name}'.")

            grid_values = np.asarray(grid.coords[coord_name].values)
            distance_values = np.asarray(distance_field.coords[coord_name].values)
            if grid_values.shape != distance_values.shape or not np.allclose(
                grid_values,
                distance_values,
                rtol=0,
                atol=1e-9,
            ):
                raise ValueError(
                    f"Distance field {coord_name} coordinates do not align with the prediction grid."
                )

    @staticmethod
    def _residual_array(residuals, ids, x_coords, y_coords) -> xr.DataArray:
        return xr.DataArray(
            np.asarray(residuals, dtype=float),
            dims=("id",),
            coords={
                "id": ids,
                "x": ("id", np.asarray(x_coords, dtype=float)),
                "y": ("id", np.asarray(y_coords, dtype=float)),
            },
            name="residual",
        )

    @staticmethod
    def _point_frame_to_arrays(points: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        missing_cols = [col for col in _REQUIRED_COLUMNS if col not in points.columns]
        if missing_cols:
            raise ValueError(f"Point predictions require columns {_REQUIRED_COLUMNS}. Missing: {missing_cols}")

        for req_col in _REQUIRED_COLUMNS:
            if points[req_col].isna().any():
                raise ValueError(f"Found NaN values in target point column {req_col}")

        ids = points["station_id"].to_numpy(dtype=str)
        X = points["elevation"].to_numpy(dtype=float).reshape(-1, 1)
        x_coords = points["x"].to_numpy(dtype=float)
        y_coords = points["y"].to_numpy(dtype=float)
        return X, x_coords, y_coords, ids

    def fit(self, job: InterpolationJob) -> FittedInterpolator:
        if not isinstance(job, InterpolationJob):
            raise TypeError(f"Interpolator.fit requires an InterpolationJob. Got {type(job)}")

        y, X, x_coords, y_coords, ids = job.to_arrays()

        if len(y) < self.min_sample_size:
            raise ValueError(
                f"Cannot fit interpolation job {job} because only {len(y)} station sample(s) "
                f"are available and {self.min_sample_size} are required."
            )

        vertical_fit = self.vertical_model.fit(X, y)
        station_predictions = np.asarray(vertical_fit.predict(X), dtype=float).reshape(-1)
        residuals = (
            self._residual_array(y - station_predictions, ids, x_coords, y_coords)
        )

        return FittedInterpolator(
            vertical_fit = vertical_fit,
            residuals = residuals,
            job = job
        )

    def predict(
        self,
        model: FittedInterpolator,
        target_points: xr.DataArray | pd.DataFrame,
        distance_fields: DistanceField | xr.DataArray | None = None,
        lam_value: float | int | None = None,
    ) -> xr.DataArray | pd.Series:
        ## TODO: Make sure target_points has the same crs that was used when calling .fit()
        if isinstance(target_points, xr.DataArray):
            return self._predict_grid(
                model, target_points, distance_fields=distance_fields, lam_value=lam_value
                )
        if isinstance(target_points):
            return self._predict_points(
                model, target_points, distance_fields=distance_fields, lam_value=lam_value
                )
        raise TypeError(f"target_points must be  an xarray DataArray or a pandas DataFrame. Got {type(target_points)}")

    def _predict_grid(
        self,
        model: FittedInterpolator,
        grid: xr.DataArray,
        distance_fields: DistanceField | xr.DataArray | None = None,
        lam_value: float | int | None = None,
    ) -> xr.DataArray:
        vertical_prediction = model.vertical_fit.predict(grid)
        prediction = vertical_prediction

        if self.residual_model is not None and distance_fields is None:
            logger.warning("Residual model available but no distance fields provided. Residuals will not be interpolated.")

        if self.residual_model is not None and distance_fields is not None:
            distance_field = self._select_distance_field(distance_fields, lam_value=lam_value)
            residual_prediction = self.residual_model.interpolate(y=model.residuals, distance_field=distance_field)

            if residual_prediction is not None:
                self._check_grid_alignment(vertical_prediction, distance_field)
                residual_prediction = residual_prediction.assign_coords(
                    {
                        "x": vertical_prediction.x.values,
                        "y": vertical_prediction.y.values,
                    }
                )
                prediction = vertical_prediction + residual_prediction

        return prediction.rename(model.parameter)

    def _predict_points(
        self,
        model: FittedInterpolator,
        points: pd.DataFrame,
        distance_fields: DistanceField | xr.DataArray | None = None,
        lam_value: float | int | None = None,
    ) -> pd.Series:

        point_X, x_coords, y_coords, ids = self._point_frame_to_arrays(points)
        vertical_prediction = np.asarray(model.vertical_fit.predict(point_X), dtype=float).reshape(-1)
        prediction = vertical_prediction

        if self.residual_model is not None and distance_fields is None:
            logger.warning("Residual model available but no point distance fields provided. Residuals will not be interpolated.")

        if self.residual_model is not None and distance_fields is not None:
            if isinstance(distance_fields, DistanceField):
                point_distances = distance_fields.to_points(ids, x_coords, y_coords)
            else:
                point_distances = distance_fields
            point_distances = self._select_distance_field(point_distances, lam_value=lam_value)
            if "target_id" not in point_distances.dims:
                raise ValueError(
                    "Point prediction distance fields must include a target_id dimension. "
                    "Use DistanceField.to_points(...) for station-target predictions."
                )
            residual_prediction = self.residual_model.interpolate(y=model.residuals, distance_field=point_distances)
            if residual_prediction is not None:
                residual_prediction = residual_prediction.sel(target_id=ids)
                prediction = vertical_prediction + np.asarray(residual_prediction.values, dtype=float).reshape(-1)

        return pd.Series(
            prediction,
            index=pd.Index(ids, name="station_id"),
            name=model.parameter,
        )
