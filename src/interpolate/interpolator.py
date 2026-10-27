import logging
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr
from pyproj import CRS

from .distance import DistanceField
from .idw import InverseDistanceWeighting
from .regions import InterpolationRegions
from .vertical import BaseFittedVerticalModel, BaseVerticalModel

logger = logging.getLogger(__name__)

##TODO: Move naming of prediction, vertical_prediction and residual prediction dataarrays into InterpolationJob results and out of Interpolator methods
##TODO: Remove pd.Series support from InterpolationJob and store only xarray arrays

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
        object.__setattr__(self, "observations", self.observations.copy())
        self.observations["station_id"] = self.observations["station_id"].astype(str)
        object.__setattr__(self, "training_points", self.training_points.copy())
        self.training_points.index = self.training_points.index.astype(str)

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
        ids = self.observations["station_id"].to_numpy(dtype=str)
        training_points = self.training_points.loc[ids]
        x_coords = training_points.geometry.x.to_numpy(dtype=float)
        y_coords = training_points.geometry.y.to_numpy(dtype=float)

        y = self.observations[self.parameter].to_numpy(dtype=float)
        X = self.observations["elevation"].to_numpy(dtype=float).reshape(-1, 1)

        return y, X, x_coords, y_coords, ids

    def __repr__(self):
        return f"InterpolationJob (date: {self.timestamp:%Y-%m-%d %H:%M}, parameter: {self.parameter}, samples: {len(self.observations)})"

@dataclass(frozen=True)
class FittedInterpolator:
    vertical_fit: BaseFittedVerticalModel
    residuals: xr.DataArray
    job: InterpolationJob

    @property
    def parameter(self) -> str:
        return self.job.parameter

    @property
    def timestamp(self) -> pd.Timestamp:
        return self.job.timestamp

    @property
    def residual_summary(self) -> dict[str, float | int]:
        values = np.asarray(self.residuals.values, dtype=float)
        return {
            "n": int(values.size),
            "mean": float(np.mean(values)),
            "mae": float(np.mean(np.abs(values))),
            "rmse": float(np.sqrt(np.mean(values**2))),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

@dataclass(frozen=True)
class InterpolationPrediction:
    prediction: xr.DataArray
    vertical_prediction: xr.DataArray
    residual_prediction: xr.DataArray | None
    lam_value: float | int | None
    station_ids: tuple[str, ...]

    @property
    def used_residuals(self) -> bool:
        return self.residual_prediction is not None

    def to_dataset(self, parameter_name: str):
        ds = xr.Dataset({
            parameter_name: self.prediction,
            f"{parameter_name}_vertical": self.vertical_prediction
        })
        if self.residual_prediction is not None:
            ds[f"{parameter_name}_residuals"] = self.residual_prediction
        
        return ds

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
    def _crs_from_target(target) -> CRS | None:
        rio = getattr(target, "rio", None)
        if rio is not None and getattr(rio, "crs", None) is not None:
            return CRS.from_user_input(rio.crs)

        attrs = getattr(target, "attrs", {})
        if isinstance(attrs, dict) and attrs.get("crs") is not None:
            return CRS.from_user_input(attrs["crs"])

        return None

    @classmethod
    def _check_prediction_crs(cls, model: FittedInterpolator, target) -> None:
        training_crs = CRS.from_user_input(model.job.training_points.crs)
        target_crs = cls._crs_from_target(target)
        if target_crs is None:
            raise ValueError("Prediction target CRS cannot be None.")
        if target_crs != training_crs:
            raise ValueError(
                "Prediction target CRS does not match interpolation training point CRS. "
                f"Got {target_crs.to_string()} vs {training_crs.to_string()}."
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
        target_points: xr.DataArray,
        distance_fields: DistanceField | xr.DataArray | None = None,
        lam_value: float | None = None,
    ) -> InterpolationPrediction:
        
        if not isinstance(target_points, xr.DataArray):
            raise TypeError(f"target_points must be  an xarray DataArray. Got {type(target_points)}")
        
        self._check_prediction_crs(model, target_points)

        vertical_prediction = model.vertical_fit.predict(target_points)
        prediction = vertical_prediction
        residual_prediction = None

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

        return InterpolationPrediction(
            prediction=prediction.rename(model.parameter),
            vertical_prediction=vertical_prediction.rename(f"{model.parameter}_vertical"),
            residual_prediction=(
                residual_prediction.rename(f"{model.parameter}_residual")
                if residual_prediction is not None
                else None
            ),
            lam_value=lam_value,
            station_ids=tuple(model.job.observations["station_id"].astype(str)),
        )
