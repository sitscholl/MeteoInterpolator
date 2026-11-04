import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold, ParameterGrid

from ..config import CrossValidationConfig
from ..utils import to_utc_timestamp
from .distance import DistanceField
from .interpolator import InterpolationJob, Interpolator

logger = logging.getLogger(__name__)

_SCORING_FUNCTIONS = {
    'rmse': root_mean_squared_error,
    'mae': mean_absolute_error
}

def _is_marginal_improvement(candidate_score, current_best, min_improvement):
    if candidate_score is None:
        raise ValueError("Candidate score cannot be None")
    
    if current_best is None:
        return True

    if not np.isfinite(current_best) or not np.isfinite(candidate_score):
        return False
    
    improvement = current_best - candidate_score
    if improvement <= 0:
        return False

    denominator = max(abs(current_best), np.finfo(float).eps)
    return improvement / denominator >= min_improvement

def _build_target_grid(observations: pd.DataFrame, crs: str | int):
    required_columns = ['elevation', 'station_id', 'x', 'y']
    missing_columns = [i for i in required_columns if i not in observations.columns]
    if missing_columns:
        raise ValueError(f"Cannot build target grid due to missing columns: {missing_columns}")

    return xr.DataArray(
        observations["elevation"].to_numpy(dtype=float),
        dims=("target_id",),
        coords={
            "target_id": observations["station_id"].to_numpy(dtype=str),
            "x": ("target_id", observations["x"].to_numpy(dtype=float)),
            "y": ("target_id", observations["y"].to_numpy(dtype=float)),
        },
        name="elevation",
    ).to_dataset().rio.write_crs(crs)

@dataclass(frozen=True)
class CrossValidationResult:
    fold_results: pd.DataFrame
    param_grid: dict[str, list[Any]]
    n_splits: int
    search_stopped: bool
    selection_metric: str

    def __post_init__(self):
        required_cols = [*list(self.param_grid.keys()), 'observed', 'predicted']
        for c in required_cols:
            if c not in self.fold_results.columns:
                raise ValueError(f"Missing column {c} in fold results. Got {self.fold_results.columns}")
        
        for param_values in self.param_grid.values():
            if any(i is None for i in param_values):
                raise ValueError("The parameter grid cannot contain None values for any parameter")

    def calculate_fold_scores(self, metric: str | None = None):

        if metric is None:
            metric = self.selection_metric

        score_fn = _SCORING_FUNCTIONS.get(metric)
        if score_fn is None:
            raise ValueError(f"Unknown metric {metric}. Choose one of {_SCORING_FUNCTIONS.keys()}")

        return self.fold_results.groupby(list(self.param_grid.keys())).apply(lambda x: score_fn(x['observed'], x['predicted']))

    def get_best_parameters(self, metric: str | None = None):

        fold_scores = self.calculate_fold_scores(metric)
        if fold_scores.empty:
            return {}, None

        best_params = fold_scores.idxmin()
        
        if fold_scores.index.nlevels == 1:
            best_params_dict = {fold_scores.index.name: best_params}
        else:
            best_params_dict = dict(zip(fold_scores.index.names, best_params))

        return best_params_dict, fold_scores.loc[best_params]

def cross_validate(
    estimator: Interpolator, 
    job: InterpolationJob, 
    param_grid: dict[str, list[Any]], 
    distance_fields: DistanceField,
    n_splits: int | None = None, 
    min_relative_score_improvement: float | None = None,
    improvement_latency_steps: int = 3,
    metric: str = 'rmse',
):

    ##Validate cv options
    config = CrossValidationConfig(
        enabled = True,
        param_grid=param_grid,
        n_splits=n_splits,
        min_relative_score_improvement=min_relative_score_improvement,
        improvement_latency_steps=improvement_latency_steps,
        select_by=metric,
    )

    if not isinstance(distance_fields, DistanceField):
        raise TypeError(f"Distance fields must be of type DistanceField. Got {type(distance_fields)}")
    if not estimator.requires_distances:
        raise ValueError("Cross validation enabled but estimator requires no distance tuning.")

    scoring_function = _SCORING_FUNCTIONS[config.select_by]

    observations = job.observations.reset_index(drop=True)
    # Use the same authoritative geometry as InterpolationJob.to_arrays().
    points = job.training_points.loc[observations["station_id"]]
    observations["x"] = points.geometry.x.to_numpy()
    observations["y"] = points.geometry.y.to_numpy()

    if config.n_splits is None:
        n_splits = len(observations)
    else:
        n_splits = config.n_splits

    ## Build cv-folds
    cv_folds = list(KFold(n_splits).split(observations))

    ## Build cv-parameter-grid
    sorted_grid = {"lam_value": sorted(config.param_grid["lam_value"])}
    parameter_grid = ParameterGrid(sorted_grid)

    results = []
    best_param_score = None
    stalled_steps = 0
    search_stopped = False
    for j, param_combination in enumerate(parameter_grid):
        logger.debug("Testing parameter combination %s", param_combination)

        rows = []
        for i, (train_index, test_index) in enumerate(cv_folds):
            logger.debug("Testing fold %s", i)

            try:
                train_obs = observations.iloc[train_index].copy()
                test_obs = observations.iloc[test_index].copy()
                test_targets = _build_target_grid(test_obs, crs=job.training_points.crs)
                observed_values = test_obs[job.parameter].values
                test_ids = test_obs["station_id"].unique()

                if len(train_obs) < estimator.min_sample_size:
                    raise ValueError(f"{estimator.min_sample_size} samples are required for model fitting. Got {len(train_obs)} for fold {i}.")
                
                fold_job = InterpolationJob(
                            timestamp=to_utc_timestamp(job.timestamp, raise_missing_timezone = True),
                            parameter=job.parameter,
                            observations=train_obs,
                            training_points=job.training_points.loc[train_obs['station_id'].astype(str).tolist()],
                        )
                fold_prediction = estimator.run(
                    job=fold_job,
                    targets=test_targets,
                    distance_fields=distance_fields,
                    **param_combination,
                )
                predicted_values = fold_prediction.prediction.sel(target_id=test_ids).values
                
                fold_result = pd.DataFrame({
                    **param_combination,
                    'fold': i,
                    'test_id': test_ids,
                    'observed': observed_values,
                    'predicted': predicted_values
                })
                rows.append(fold_result)
            except Exception as e:
                logger.exception(f"Cross validation fold {i} for parameters {param_combination} failed.")
                raise
                        
        param_combination_results = pd.concat(rows)
        results.append(param_combination_results)

        if config.min_relative_score_improvement is None:
            continue
        else:
            current_score = scoring_function(param_combination_results['observed'], param_combination_results['predicted'])

            if _is_marginal_improvement(current_score, best_param_score, min_improvement = config.min_relative_score_improvement):
                logger.debug("Setting best score from %s to %s", best_param_score, current_score)
                best_param_score = current_score
                stalled_steps = 0
            else:
                stalled_steps += 1
                logger.debug(
                    "No score improvement for parameter combination %s: %s (candidate) vs %s (current). Stalled steps: %s", 
                    param_combination, current_score, best_param_score, stalled_steps)
        
        if stalled_steps >= config.improvement_latency_steps:
            logger.debug("Stopping parameter search because improvement latency has been reached at score %s", best_param_score)
            search_stopped = True
            break

    return CrossValidationResult(
        fold_results = pd.concat(results),
        param_grid = config.param_grid,
        n_splits = n_splits,
        search_stopped = search_stopped,
        selection_metric = config.select_by,
    )
