import xarray as xr
import pandas as pd
import geopandas as gpd

import logging
from dataclasses import dataclass, field
from typing import Literal, Any

from .interpolate import Interpolator, CrossValidator, InterpolationJob, DistanceField, InterpolationPrediction, CrossValidationResult

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
    job,
    interpolator: Interpolator,
    cross_validator: CrossValidator,
    prediction_target: xr.DataArray | pd.DataFrame | gpd.GeoDataFrame,
    distance_fields: DistanceField,
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
        cv_result = cross_validator.cross_validate(
            interpolator,
            job,
            distance_fields=distance_fields,
        )

        lam_value = None
        if cv_result is not None:
            lam_value = cv_result.best_params.get("lambda")
            logger.info(
                "Cross-validation selected parameters for %s: %s (%s=%s)",
                job,
                cv_result.best_params,
                cv_result.select_by,
                cv_result.best_score,
            )

        fitted = interpolator.fit(job)
        prediction_result = interpolator.predict(
            fitted,
            prediction_target,
            distance_fields=distance_fields,
            lam_value=lam_value,
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
            job = job, prediction = None, cv_result = None, status = 'failed', error = e
        )
