import xarray as xr
import pandas as pd
import numpy as np
import rioxarray  # noqa: F401
from pyproj import CRS

from dataclasses import dataclass
import logging

from .vertical import BaseVerticalModel
from .distance import BaseDistanceCalculator, DistanceField
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
    target_grid: xr.DataArray
    crs: CRS | str | int
    distance_fields: DistanceField | None = None

    @property
    def required_columns(self):
        return [self.parameter, *_REQUIRED_COLUMNS]

    def __post_init__(self):
        if not isinstance(self.target_grid, xr.DataArray):
            raise TypeError(f"InterpolationJob target_grid must be an xarray DataArray. Got {type(self.target_grid)}")
        if "x" not in self.target_grid.dims or "y" not in self.target_grid.dims:
            raise ValueError(f"InterpolationJob target_grid must contain x and y dimensions. Got {self.target_grid.dims}")
        if self.target_grid.rio.crs is None:
            raise ValueError("InterpolationJob target_grid must have an explicit CRS.")

        crs = CRS.from_user_input(self.crs)
        target_crs = CRS.from_user_input(self.target_grid.rio.crs)
        if crs != target_crs:
            raise ValueError(f"InterpolationJob CRS {crs.to_string()} does not match target grid CRS {target_crs.to_string()}.")
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
    vertical_model: BaseVerticalModel
    distance_calculator: BaseDistanceCalculator | None
    residual_model: InverseDistanceWeighting | None = None
    regions: InterpolationRegions | None = None
    cross_validator: CrossValidator | None = None
    min_sample_size: int = 3

    @classmethod
    def from_config(cls, config: dict):
        vertical_config = dict(config["vertical_model"])
        vertical_handler = vertical_config.pop("type")
        vertical_model = BaseVerticalModel.create(vertical_handler, **vertical_config)

        ##Residual interpolation
        distance_config = config.get('distance')
        if distance_config is None:
            logger.info("No distance calculator configuration provided. Residuals will not be interpolated")
            distance_calculator = None
        else:
            distance_config = dict(distance_config)
            distance_handler = distance_config.pop("type")
            distance_calculator = BaseDistanceCalculator.create(distance_handler, **distance_config)

        idw_config = config.get('inverse_distance_weighting')
        if idw_config is None:
            logger.warning("No inverse distance weigthing configuration provided. Residuals will not be interpolated")
            residual_model = None
            distance_calculator = None
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
            vertical_model = vertical_model, 
            distance_calculator = distance_calculator, 
            residual_model = residual_model, 
            regions = interpolation_regions, 
            cross_validator = cross_validator,
            min_sample_size = min_sample_size
            )

    def prepare_distance_fields(self, job: InterpolationJob) -> DistanceField:
        if self.distance_calculator is None:
            raise ValueError(
                "Residual interpolation requires distance_fields on the InterpolationJob "
                "or a configured distance calculator."
            )

        _, _, x_coords, y_coords, ids = job.to_arrays()
        return self.distance_calculator.calculate_fields(
            dem=job.target_grid,
            x_coords=x_coords,
            y_coords=y_coords,
            point_ids=ids,
        )

    def _check_grid_alignment(self, grid: xr.DataArray, distance_fields: DistanceField):
        if not isinstance(distance_fields, DistanceField):
            raise ValueError(f"distance_fields must be a DistanceField. Got {type(distance_fields)}")
        if distance_fields.data is None:
            raise ValueError("distance_fields.data must not be None for residual interpolation.")

        distance_data = distance_fields.data
        for coord_name in ("x", "y"):
            if coord_name not in grid.coords:
                raise ValueError(f"Prediction grid is missing coordinate '{coord_name}'.")
            if coord_name not in distance_data.coords:
                raise ValueError(f"Distance fields are missing coordinate '{coord_name}'.")

            grid_values = np.asarray(grid.coords[coord_name].values)
            distance_values = np.asarray(distance_data.coords[coord_name].values)
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

    def interpolate(self, job: InterpolationJob) -> InterpolationResult | None:
        if not isinstance(job, InterpolationJob):
            raise TypeError(f"Interpolator.interpolate requires an InterpolationJob. Got {type(job)}")

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

        vertical_fit = self.vertical_model.fit(X, y)
        vertical_prediction = vertical_fit.predict(job.target_grid)
        residual_prediction = None
        prediction = vertical_prediction

        if self.residual_model is not None:
            distance_fields = job.distance_fields
            if distance_fields is None:
                distance_fields = self.prepare_distance_fields(job)

            station_predictions = np.asarray(vertical_fit.predict(X), dtype=float).reshape(-1)
            residuals = y - station_predictions
            residuals = self._residual_array(residuals, ids, x_coords, y_coords)
            residual_prediction = self.residual_model.interpolate(y=residuals, distance_fields=distance_fields)

            if residual_prediction is not None:
                self._check_grid_alignment(vertical_prediction, distance_fields)
                residual_prediction = residual_prediction.assign_coords(
                    {
                        "x": vertical_prediction.x.values,
                        "y": vertical_prediction.y.values,
                    }
                )
                prediction = vertical_prediction + residual_prediction

        return InterpolationResult(
            timestamp=job.timestamp,
            parameter=job.parameter,
            prediction=prediction,
            vertical_prediction=vertical_prediction,
            residual_prediction=residual_prediction,
            cv_results=cv_results,
        )
