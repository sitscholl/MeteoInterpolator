import xarray as xr
import pandas as pd
import geopandas as gpd

import logging
from dataclasses import dataclass, field
from typing import Literal, Any

from .runtime import CrossValidationConfig
from .interpolate import Interpolator, InterpolationJob, DistanceField, InterpolationPrediction, cross_validate, CrossValidationResult

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class InterpolationJobResult:
    job: InterpolationJob
    prediction: InterpolationPrediction | None
    cv_result: CrossValidationResult | None
    status: Literal["completed", "skipped", "failed"]
    error: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)

def process_interpolation_job(
    job: InterpolationJob,
    interpolator: Interpolator,
    cross_validation_config: CrossValidationConfig,
    prediction_target: xr.DataArray | pd.DataFrame | gpd.GeoDataFrame,
    distance_fields: DistanceField | None,
) -> InterpolationJobResult:
    logger.info('Starting %s', job)

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

        return InterpolationJobResult(
            job = job,
            prediction = prediction_result,
            cv_result = cv_result,
            status = 'completed',
        )

    except Exception as e:
        logger.exception(f"{job} failed with error: {e}")
        return InterpolationJobResult(
            job = job, prediction = None, cv_result = None, status = 'failed', error = str(e)
        )
