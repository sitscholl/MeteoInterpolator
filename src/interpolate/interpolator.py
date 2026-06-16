import xarray as xr
import pandas as pd
import numpy as np
import rioxarray  # noqa: F401
from pyproj import CRS

from dataclasses import dataclass
import logging

from ..array.base_grid import BaseGrid
from .vertical import BaseVerticalModel
from .distance import DistanceField
from .idw import InverseDistanceWeighting
from .regions import InterpolationRegions
from .cv import CrossValidator

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

@dataclass(frozen=True)
class InterpolationResult:
    timestamp: pd.Timestamp
    parameter: str
    prediction: xr.DataArray
    vertical_prediction: xr.DataArray | None = None
    residual_prediction: xr.DataArray | None = None
    cv_results: pd.DataFrame | None = None

@dataclass
class Interpolator:
    base_grid: BaseGrid
    vertical_model: BaseVerticalModel
    residual_model: InverseDistanceWeighting | None = None
    regions: InterpolationRegions | None = None
    cross_validator: CrossValidator | None = None
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

        ## Cross validation
        cv_config = config.get('cross_validation')
        cross_validator = CrossValidator(**cv_config) if cv_config is not None else None
        if cv_config is None:
            logger.info('No cross validation configuration provided. Cross validation will be skipped')

        min_sample_size = config.get('min_sample_size', 3)

        return cls(
            base_grid = base_grid,
            vertical_model = vertical_model, 
            residual_model = residual_model, 
            regions = interpolation_regions, 
            cross_validator = cross_validator,
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

    def interpolate(
        self,
        job: InterpolationJob,
        distance_fields: DistanceField | xr.DataArray | None = None,
    ) -> InterpolationResult | None:
        if not isinstance(job, InterpolationJob):
            raise TypeError(f"Interpolator.interpolate requires an InterpolationJob. Got {type(job)}")
        self._check_job_crs(job)

        if self.cross_validator is not None:
            raise NotImplementedError("Cross Validation has not been implemented yet")
        else:
            cv_results = None

        y, X, x_coords, y_coords, ids = job.to_arrays()
        
        if len(y) < self.min_sample_size:
            logger.warning(
                "Skipping interpolation job %s because only %s station sample(s) are available and %s are required.",
                job,
                len(y),
                self.min_sample_size,
            )
            return None

        ##todo: handle failed fits or very poor fits. Either log warnign or return early
        vertical_fit = self.vertical_model.fit(X, y)
        vertical_prediction = vertical_fit.predict(self.target_grid)
        prediction = vertical_prediction

        if self.residual_model is not None and distance_fields is None:
            logger.warning("Residual model available but no distance fields provided. Residuals will not be interpolated.")

        if self.residual_model is not None and distance_fields is not None:
            distance_field = self._select_distance_field(distance_fields)

            station_predictions = np.asarray(vertical_fit.predict(X), dtype=float).reshape(-1)
            residuals = y - station_predictions
            residuals = self._residual_array(residuals, ids, x_coords, y_coords)

            residual_prediction = self.residual_model.interpolate(y=residuals, distance_field=distance_field)

            if residual_prediction is not None:
                self._check_grid_alignment(vertical_prediction, distance_field)
                residual_prediction = residual_prediction.assign_coords(
                    {
                        "x": vertical_prediction.x.values,
                        "y": vertical_prediction.y.values,
                    }
                )
                prediction = vertical_prediction + residual_prediction
        else:
            residual_prediction = None

        return InterpolationResult(
            timestamp=job.timestamp,
            parameter=job.parameter,
            prediction=prediction,
            vertical_prediction=vertical_prediction,
            residual_prediction=residual_prediction,
            cv_results=cv_results,
        )
