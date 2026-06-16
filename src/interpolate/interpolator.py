import xarray as xr
import pandas as pd
import numpy as np
import rioxarray  # noqa: F401
from pyproj import CRS

from dataclasses import dataclass
import logging

from ..array.base_grid import BaseGrid
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
    crs: CRS | str | int

    @property
    def required_columns(self):
        return [self.parameter, *_REQUIRED_COLUMNS]

    def __post_init__(self):
        crs = CRS.from_user_input(self.crs)
        object.__setattr__(self, "crs", crs)

        for req_col in self.required_columns:
            if req_col not in self.observations.columns:
                raise ValueError(f"InterpolationJob observations is missing required column {req_col}. Got {self.observations.columns}")
            if self.observations[req_col].isna().any():
                raise ValueError(f"Found NaN values in InterpolationJob observations for column {req_col}")

        if self.observations["station_id"].duplicated().any():
            duplicated = self.observations.loc[self.observations["station_id"].duplicated(), "station_id"].tolist()
            raise ValueError(f"InterpolationJob observations contain duplicated station_id values: {duplicated}")

    def to_arrays(self):
        y = self.observations[self.parameter].to_numpy(dtype=float)
        X = self.observations["elevation"].to_numpy(dtype=float).reshape(-1, 1)
        x_coords = self.observations["x"].to_numpy(dtype=float)
        y_coords = self.observations["y"].to_numpy(dtype=float)
        ids = self.observations["station_id"].to_numpy(dtype=str)
        return y, X, x_coords, y_coords, ids

    def __repr__(self):
        return f"InterpolationJob (date: {self.timestamp}, parameter: {self.parameter}, samples: {len(self.observations)})"

@dataclass
class Interpolator:
    base_grid: BaseGrid
    vertical_model: BaseVerticalModel
    residual_model: InverseDistanceWeighting | None = None
    regions: InterpolationRegions | None = None
    min_sample_size: int = 3

    @property
    def target_grid(self) -> xr.DataArray:
        return self.base_grid.data

    def __post_init__(self):
        if not isinstance(self.base_grid, BaseGrid):
            raise TypeError(f"Interpolator base_grid must be a BaseGrid. Got {type(self.base_grid)}")
        if not isinstance(self.target_grid, xr.DataArray):
            raise TypeError(f"Interpolator target_grid must be an xarray DataArray. Got {type(self.target_grid)}")
        if "x" not in self.target_grid.dims or "y" not in self.target_grid.dims:
            raise ValueError(f"Interpolator target_grid must contain x and y dimensions. Got {self.target_grid.dims}")
        if self.target_grid.rio.crs is None:
            raise ValueError("Interpolator target_grid must have an explicit CRS.")
        self.vertical_fit_: BaseFittedVerticalModel | None = None
        self.residuals_: xr.DataArray | None = None
        self.timestamp_: pd.Timestamp | None = None
        self.parameter_: str | None = None
        self.station_ids_: np.ndarray | None = None

    @classmethod
    def from_config(
        cls,
        base_grid: BaseGrid,
        config: dict,
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
            base_grid = base_grid,
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

    def _check_job_crs(self, job: InterpolationJob):
        target_crs = CRS.from_user_input(self.target_grid.rio.crs)
        if job.crs != target_crs:
            raise ValueError(
                f"InterpolationJob CRS {job.crs.to_string()} does not match "
                f"target grid CRS {target_crs.to_string()}."
            )

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
                raise ValueError(f"Found NaN values in point prediction column {req_col}")

        ids = points["station_id"].to_numpy(dtype=str)
        X = points["elevation"].to_numpy(dtype=float).reshape(-1, 1)
        x_coords = points["x"].to_numpy(dtype=float)
        y_coords = points["y"].to_numpy(dtype=float)
        return X, x_coords, y_coords, ids

    def _check_fitted(self) -> BaseFittedVerticalModel:
        if self.vertical_fit_ is None:
            raise ValueError("Interpolator is not fitted yet. Call fit(job) before predict(...).")
        return self.vertical_fit_

    def fit(self, job: InterpolationJob) -> "Interpolator":
        if not isinstance(job, InterpolationJob):
            raise TypeError(f"Interpolator.fit requires an InterpolationJob. Got {type(job)}")
        self._check_job_crs(job)

        y, X, x_coords, y_coords, ids = job.to_arrays()

        if len(y) < self.min_sample_size:
            raise ValueError(
                f"Cannot fit interpolation job {job} because only {len(y)} station sample(s) "
                f"are available and {self.min_sample_size} are required."
            )

        vertical_fit = self.vertical_model.fit(X, y)
        station_predictions = np.asarray(vertical_fit.predict(X), dtype=float).reshape(-1)

        self.vertical_fit_ = vertical_fit
        self.residuals_ = (
            self._residual_array(y - station_predictions, ids, x_coords, y_coords)
            if self.residual_model is not None
            else None
        )
        self.timestamp_ = job.timestamp
        self.parameter_ = job.parameter
        self.station_ids_ = ids
        return self

    def predict(
        self,
        X: xr.DataArray | pd.DataFrame | None = None,
        distance_fields: DistanceField | xr.DataArray | None = None,
        lam_value: float | int | None = None,
    ) -> xr.DataArray | pd.Series:
        if X is None:
            return self._predict_grid(self.target_grid, distance_fields=distance_fields, lam_value=lam_value)
        if isinstance(X, xr.DataArray):
            return self._predict_grid(X, distance_fields=distance_fields, lam_value=lam_value)
        if isinstance(X, pd.DataFrame):
            return self._predict_points(X, distance_fields=distance_fields, lam_value=lam_value)
        raise TypeError(f"predict X must be None, an xarray DataArray, or a pandas DataFrame. Got {type(X)}")

    def _predict_grid(
        self,
        grid: xr.DataArray,
        distance_fields: DistanceField | xr.DataArray | None = None,
        lam_value: float | int | None = None,
    ) -> xr.DataArray:
        vertical_fit = self._check_fitted()
        vertical_prediction = vertical_fit.predict(grid)
        prediction = vertical_prediction

        if self.residual_model is not None and distance_fields is None:
            logger.warning("Residual model available but no distance fields provided. Residuals will not be interpolated.")

        if self.residual_model is not None and distance_fields is not None:
            distance_field = self._select_distance_field(distance_fields, lam_value=lam_value)
            residual_prediction = self.residual_model.interpolate(y=self.residuals_, distance_field=distance_field)

            if residual_prediction is not None:
                self._check_grid_alignment(vertical_prediction, distance_field)
                residual_prediction = residual_prediction.assign_coords(
                    {
                        "x": vertical_prediction.x.values,
                        "y": vertical_prediction.y.values,
                    }
                )
                prediction = vertical_prediction + residual_prediction

        return prediction.rename(self.parameter_)

    def _predict_points(
        self,
        points: pd.DataFrame,
        distance_fields: DistanceField | xr.DataArray | None = None,
        lam_value: float | int | None = None,
    ) -> pd.Series:
        vertical_fit = self._check_fitted()
        point_X, x_coords, y_coords, ids = self._point_frame_to_arrays(points)
        vertical_prediction = np.asarray(vertical_fit.predict(point_X), dtype=float).reshape(-1)
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
            residual_prediction = self.residual_model.interpolate(y=self.residuals_, distance_field=point_distances)
            if residual_prediction is not None:
                residual_prediction = residual_prediction.sel(target_id=ids)
                prediction = vertical_prediction + np.asarray(residual_prediction.values, dtype=float).reshape(-1)

        return pd.Series(
            prediction,
            index=pd.Index(ids, name="station_id"),
            name=self.parameter_,
        )
