import xarray as xr

import logging
from dataclasses import asdict, dataclass, field
from typing import Literal, Any

from .runtime import CrossValidationConfig
from .interpolate import Interpolator, InterpolationJob, DistanceField, InterpolationPrediction, cross_validate, CrossValidationResult
from .utils import to_utc_timestamp
from .utils import to_jsonable

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class InterpolationJobResult:
    job: InterpolationJob
    prediction: InterpolationPrediction | None
    cv_result: CrossValidationResult | None
    status: Literal["completed", "skipped", "failed"]
    error: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    model_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataset(self):
        if self.status in ['skipped', 'failed']:
            raise ValueError(f"to_dataset() works only with completed Interpolation Jobs. Status={self.status}")
        if self.prediction is None:
            raise ValueError("to_dataset() requires an InterpolationPrediction object. Got None")
        if self.prediction.prediction is None:
            raise ValueError("to_dataset() requires a predicted array. Got None.")
        
        job_metadata = {
            'parameter': self.job.parameter,
            'n_observations': len(self.job.observations),
            'training_stations': self.prediction.station_ids,
            'n_training_points': len(self.job.training_points),
        }

        if self.cv_result is not None:
            best_parameters, best_score = self.cv_result.get_best_parameters()
            cv_metadata = {
                'enabled': True,
                'param_grid': self.cv_result.param_grid,
                'n_splits': self.cv_result.n_splits,
                'selection_metric': self.cv_result.selection_metric,
                'cv_search_stopped': self.cv_result.search_stopped,
                'selected_lam_value': self.prediction.lam_value,
                'best_parameters': best_parameters,
                'best_score': best_score
            }
        else:
            cv_metadata = {
                'enabled': False
            }

        ds = self.prediction.to_dataset(parameter_name=self.job.parameter)

        timestamp = to_utc_timestamp(
            self.job.timestamp, raise_missing_timezone=True
        )
        time_value = timestamp.tz_localize(None).to_datetime64()
        ds = ds.expand_dims(time=[time_value]).transpose("time", "y", "x")

        ds.attrs['lam_value'] = self.prediction.lam_value
        ds.attrs['residuals_interpolated'] = self.prediction.used_residuals
        ds.attrs['job_metadata'] = to_jsonable(job_metadata)
        ds.attrs['cv_metadata'] = to_jsonable(cv_metadata)
        ds.attrs['model_metadata'] = to_jsonable(self.model_metadata)
        ds.attrs['residual_summary'] = to_jsonable(self.stats)
        return ds

def process_interpolation_job(
    job: InterpolationJob,
    interpolator: Interpolator,
    cross_validation_config: CrossValidationConfig,
    prediction_target: xr.DataArray,
    distance_fields: DistanceField | None,
) -> InterpolationJobResult:
    logger.debug('Starting %s', job)

    if len(job.observations) < interpolator.min_sample_size:
        logger.warning(
            "Skipping interpolation job %s because only %s station sample(s) are available and %s are required.",
            job,
            len(job.observations),
            interpolator.min_sample_size,
        )
        
        return InterpolationJobResult(
            job = job, prediction = None, cv_result = None, status = 'skipped'
        )

    try:
        cv_result = None
        best_params = {}
        if cross_validation_config.enabled:
            if distance_fields is None:
                raise ValueError("Cross validation requires distance fields. Provide the distance fields or disable cross validation")
            
            cv_result = cross_validate(
                estimator=interpolator,
                job=job,
                distance_fields=distance_fields,
                param_grid=cross_validation_config.param_grid,
                n_splits=cross_validation_config.n_splits,
                min_relative_score_improvement=cross_validation_config.min_relative_score_improvement,
                improvement_latency_steps=cross_validation_config.improvement_latency_steps,
                metric=cross_validation_config.select_by
            )

            best_params, best_score = cv_result.get_best_parameters()
            logger.info(
                "Cross-validation selected parameters for %s: %s (%s=%s)",
                job,
                best_params,
                cross_validation_config.select_by,
                best_score,
            )

        fitted = interpolator.fit(job)
        prediction_result = interpolator.predict(
            fitted,
            prediction_target,
            distance_fields=distance_fields,
            **best_params,
        )

        logger.debug('Finished %s', job)
        return InterpolationJobResult(
            job = job,
            prediction = prediction_result,
            cv_result = cv_result,
            status = 'completed',
            stats = fitted.residual_summary,
            model_metadata = to_jsonable({
                "vertical_model": {
                    "type": interpolator.vertical_model.key(),
                    "settings": vars(interpolator.vertical_model),
                    "fitted_parameters": asdict(fitted.vertical_fit),
                },
                "residual_model": {
                    "configured": interpolator.residual_model is not None,
                    "used": prediction_result.used_residuals,
                    "type": (
                        type(interpolator.residual_model).__name__
                        if interpolator.residual_model is not None else None
                    ),
                    "settings": (
                        vars(interpolator.residual_model)
                        if interpolator.residual_model is not None else {}
                    ),
                    "distance_type": (
                        distance_fields.distance_type if distance_fields is not None else None
                    ),
                },
                "min_sample_size": interpolator.min_sample_size,
            }),
        )

    except Exception as e:
        logger.exception(f"{job} failed with error: {e}")
        return InterpolationJobResult(
            job = job, prediction = None, cv_result = None, status = 'failed', error = str(e)
        )
