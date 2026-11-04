import logging
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr
from pyproj import CRS

from .distance import DistanceField
from .regions import InterpolationRegions
from .models import InterpolationModel, FittedInterpolationModel

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

        X = xr.Dataset(
            data_vars = {
                'elevation': (['id'], self.observations["elevation"].to_numpy(dtype=float))
            },
            coords = {
                'id': ids,
                'x': ('id', x_coords),
                'y': ('id', y_coords),
            },
            attrs = {'crs': self.training_points.crs.to_wkt()}
        )

        y = xr.DataArray(
            data = self.observations[self.parameter].to_numpy(dtype=float),
            dims=("id",),
            coords = {'id': ids},
            name = 'observations',
        )

        return X, y

    def __repr__(self):
        return f"InterpolationJob (date: {self.timestamp:%Y-%m-%d %H:%M}, parameter: {self.parameter}, samples: {len(self.observations)})"

@dataclass(frozen=True)
class InterpolationPrediction:
    prediction: xr.DataArray
    primary_prediction: xr.DataArray
    primary_residuals: xr.DataArray
    residual_prediction: xr.DataArray | None
    lam_value: float | int | None
    station_ids: tuple[str, ...]

    @property
    def used_residuals(self) -> bool:
        return self.residual_prediction is not None

    def to_dataset(self, parameter_name: str):
        ds = xr.Dataset({
            parameter_name: self.prediction,
            f"{parameter_name}_primary": self.primary_prediction
        })
        if self.residual_prediction is not None:
            ds[f"{parameter_name}_residuals"] = self.residual_prediction
        
        return ds
    
    def residual_summary(self) -> dict[str, float | int]:
        values = np.asarray(self.primary_residuals.values, dtype=float)
        return {
            "n": int(values.size),
            "mean": float(np.mean(values)),
            "mae": float(np.mean(np.abs(values))),
            "rmse": float(np.sqrt(np.mean(values**2))),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

def _point_distances(
    selected_distances: xr.DataArray,
    targets: xr.Dataset,
) -> xr.DataArray:
    """Sample a raster at point targets, retaining requested point locations."""
    field = DistanceField(selected_distances.attrs.get("distance_type", "distance"), selected_distances)
    point_distances = field.to_points(
        targets.target_id.values, targets.x.values, targets.y.values,
    )
    # Nearest-cell sampling leaves raster cell coordinates in x/y. Associate
    # those sampled distances with the requested station coordinates for predict.
    return point_distances.assign_coords(
        x=point_distances.coords["target_x"],
        y=point_distances.coords["target_y"],
    )

@dataclass
class Interpolator:
    primary_model: InterpolationModel
    residual_model: InterpolationModel | None = None
    regions: InterpolationRegions | None = None
    min_sample_size: int = 3

    @classmethod
    def from_config(
        cls, config: dict,
    ):
        primary_config = dict(config["primary_model"])
        primary_handler = primary_config.pop("type")
        primary_model = InterpolationModel.create(primary_handler, **primary_config)

        residual_config = config.get('residual_model')
        if residual_config is None:
            residual_model = None
        else:
            residual_config = dict(residual_config)
            residual_handler = residual_config.pop('type')
            residual_model = InterpolationModel.create(residual_handler, **residual_config)

        ## Interpolation Regions
        region_config = config.get('interpolation_regions')
        interpolation_regions = InterpolationRegions(**region_config) if region_config is not None else None
        if region_config is None:
            logger.info("No interpolation regions specified.")

        min_sample_size = config.get('min_sample_size', 3)

        return cls(
            primary_model = primary_model, 
            residual_model = residual_model, 
            regions = interpolation_regions, 
            min_sample_size = min_sample_size
            )

    @property
    def requires_distances(self):
        return any(i.requires_distances for i in [self.primary_model, self.residual_model] if i is not None)

    @staticmethod
    def _select_distance_field(
        distance_fields: DistanceField | xr.DataArray,
        lam_value: float | None = None,
    ) -> xr.DataArray:
        if isinstance(distance_fields, DistanceField):
            if distance_fields.data is None:
                raise ValueError("distance_fields.data must not be None for residual interpolation.")
            distance_data = distance_fields.data
        elif isinstance(distance_fields, xr.DataArray):
            distance_data = distance_fields
        else:
            raise TypeError(f"distance_fields must be a DistanceField or xarray DataArray. Got {type(distance_fields)}")

        if "lam_value" in distance_data.dims:
            if lam_value is not None:
                return distance_data.sel(lam_value=lam_value)
            
            if distance_data.sizes['lam_value'] > 1:
                raise ValueError("Distance fields contain multiple lambda values but no value for selection provided.")
            return distance_data.isel(lam_value=0)
        return distance_data

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
    def _check_prediction_crs(cls, job: InterpolationJob, target) -> None:
        training_crs = CRS.from_user_input(job.training_points.crs)
        target_crs = cls._crs_from_target(target)
        if target_crs is None:
            raise ValueError("Prediction target CRS cannot be None.")
        if target_crs != training_crs:
            raise ValueError(
                "Prediction target CRS does not match interpolation training point CRS. "
                f"Got {target_crs.to_string()} vs {training_crs.to_string()}."
            )

    def run(
        self, 
        job: InterpolationJob, 
        targets: xr.Dataset, 
        distance_fields: DistanceField | xr.DataArray | None = None, 
        lam_value: float | None = None,
    ) -> InterpolationPrediction:

        if self.requires_distances and distance_fields is None:
            raise ValueError('Interpolation requires distance fields but none are provided')
        if not isinstance(job, InterpolationJob):
            raise TypeError(f"Interpolator.run requires an InterpolationJob. Got {type(job)}")
        if not isinstance(targets, xr.Dataset):
            raise TypeError(f"target_points must be  an xarray Dataset. Got {type(targets)}")
        self._check_prediction_crs(job, targets)
        FittedInterpolationModel._validate_targets(targets)

        X, y = job.to_arrays()

        if len(y.id) < self.min_sample_size:
            raise ValueError(
                f"Cannot fit interpolation job {job} because only {len(y)} station sample(s) "
                f"are available and {self.min_sample_size} are required."
            )

        # Select one raster field, then prepare each set of destinations.
        selected_distances = target_distances = station_distances = None
        training_targets = X.rename(id="target_id")
        if self.requires_distances:
            selected_distances = self._select_distance_field(distance_fields, lam_value=lam_value)
            target_distances = (
                _point_distances(selected_distances, targets)
                if "target_id" in targets.dims else selected_distances
            )
            if self.primary_model.requires_distances:
                station_distances = _point_distances(selected_distances, training_targets)

        if selected_distances is not None and "lam_value" in selected_distances.coords:
            lam_value = selected_distances.lam_value.values.item()

        #Fit primary model
        primary_fit = self.primary_model.fit(X, y)
        primary_prediction = primary_fit.predict(targets, distances=target_distances)

        ## Calculate residuals
        station_prediction = primary_fit.predict(
            training_targets,
            distances=station_distances,
        )
        station_prediction = station_prediction.rename(target_id="id")
        station_prediction = station_prediction.sel(id=y.id)
        residuals = (y - station_prediction).rename("residual")

        ## Fit and predict residual model
        prediction = primary_prediction
        residual_prediction = None
        if self.residual_model is not None:

            # Fit residual model and predict
            residual_fit = self.residual_model.fit(X = X, y=residuals)
            residual_prediction = residual_fit.predict(targets = targets, distances=target_distances)

            if residual_prediction is not None:
                prediction = primary_prediction + residual_prediction

        return InterpolationPrediction(
            prediction=prediction.rename(job.parameter),
            primary_prediction=primary_prediction.rename(f"{job.parameter}_primary"),
            primary_residuals=residuals,
            residual_prediction=(
                residual_prediction.rename(f"{job.parameter}_residual")
                if residual_prediction is not None
                else None
            ),
            lam_value=lam_value,
            station_ids=tuple(job.observations["station_id"].astype(str)),
        )
