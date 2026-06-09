import xarray as xr
import pandas as pd

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
    distance_fields: DistanceField | None = None

    @property
    def required_columns(self):
        return [self.parameter, *_REQUIRED_COLUMNS]

    def __post_init__(self):
        
        for req_col in self.required_columns:

            if req_col not in self.observations.columns:
                raise ValueError(f"InterpolationJob observations is missing required column {req_col}. Got {self.observations.columns}")

            if self.observations[req_col].isna().any():
                raise ValueError(f"Found NaN values in InterpolationJob observations for column {req_col}")

    def to_arrays(self):
        y = self.observations[self.parameter].to_numpy(dtype=float)
        X = self.observations["elevation"].to_numpy(dtype=float).reshape(-1, 1)
        coords = [(x, y) for x,y in zip(self.observations['x'], self.observations['y'])]
        ids = self.observations['station_id'].to_numpy(dtype = str)
        return (y, X, coords, ids)

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

    def prepare_distance_fields(self, source_points, target_grid):
        pass

    def _check_grid_alignment(self, grid1, grid2):
        pass

    def interpolate(self, job: InterpolationJob) -> InterpolationResult:
        if self.cross_validator is not None:
            raise NotImplementedError("Cross Validation has not been implemented yet")
        else:
            cv_results = None     
        y, X, coords, ids = job.to_arrays()
        
        if len(y) < self.min_sample_size:
            logger.warning(
                "Skipping interpolation job %s because only %s station sample(s) are available and %s are required.",
                job,
                len(y),
                self.min_sample_size,
            )
            return None

        vertical_fit = self.vertical_model.fit(X, y)
        predictions = vertical_fit.predict(job.target_grid)

        if self.residual_model is not None:
            self._check_grid_alignment(predictions, job.distance_fields) #raise if spatial coords do not align

            residuals = y - vertical_fit.predict(X)
            residuals = xr.DataArray(residuals, dims = {'id': ids, 'coords': coords})
            residual_field = self.residual_model.interpolate(y = residuals, distance_fields = job.distance_fields)

            predictions += residual_field.assign_coords({'x': predictions.x.values, 'y': predictions.y.values}) #avoid floating point mismatches in coords

        return predictions, cv_results
