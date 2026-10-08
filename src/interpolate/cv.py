import logging
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold, ParameterGrid

from .distance import DistanceField
from .interpolator import InterpolationJob, Interpolator

logger = logging.getLogger(__name__)

def _point_distances(
    distance_fields: DistanceField | xr.DataArray | None,
    observations: pd.DataFrame,
) -> xr.DataArray | None:
    if distance_fields is None:
        return None

    ids = observations["station_id"].to_numpy(dtype=str)
    x_coords = observations["x"].to_numpy(dtype=float)
    y_coords = observations["y"].to_numpy(dtype=float)

    if isinstance(distance_fields, DistanceField):
        return distance_fields.to_points(ids, x_coords, y_coords)

    if not isinstance(distance_fields, xr.DataArray):
        raise TypeError(f"distance_fields must be a DistanceField or xarray DataArray. Got {type(distance_fields)}")

    if "target_id" in distance_fields.dims:
        return distance_fields

    return DistanceField(
        distance_type=distance_fields.attrs.get("distance_type", distance_fields.name or "distance"),
        data=distance_fields,
    ).to_points(ids, x_coords, y_coords)

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

@dataclass(frozen=True)
class CrossValidationResult:
    fold_results: pd.DataFrame
    param_grid: dict[str, list[Any]]
    n_splits: int
    search_stopped: bool

    _score_fns: ClassVar = {
            'rmse': root_mean_squared_error,
            'mae': mean_absolute_error
        }

    def __post_init__(self):
        #validate fold_results dataframe
        pass

    def calculate_fold_scores(self, metric: str = 'rmse'):

        score_fn = self._score_fns.get(metric)
        if score_fn is None:
            raise ValueError(f"Unknown metric {metric}. Choose one of {self._score_fns.keys()}")

        return self.fold_results.groupby(list(self.param_grid.keys())).apply(lambda x: score_fn(x['observed'], x['predicted']))

    def get_best_parameters(self, metric: str = 'rmse'):
        fold_scores = self.calculate_fold_scores(metric)

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
    min_score_improvement: float | None = None,
    improvement_latency_steps: int = 3
):
    observations = job.observations.reset_index(drop=True)
    point_distances = _point_distances(distance_fields, observations)

    if n_splits is None:
        n_splits = len(observations)

    ## Build cv-folds
    cv_folds = list(KFold(n_splits).split(observations))

    ## Build cv-parameter-grid
    parameter_grid = ParameterGrid(param_grid)

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
                test_obs.attrs["crs"] = job.training_points.crs #required for estimator.predict
                observed_values = test_obs[job.parameter].values
                test_ids = test_obs["station_id"].unique()

                fold_distances = (
                    point_distances.sel(target_id=point_distances.target_id.isin(test_ids))
                    if point_distances is not None
                    else None
                )
                
                fold_job = InterpolationJob(
                            timestamp=job.timestamp,
                            parameter=job.parameter,
                            observations=observations,
                            training_points=job.training_points.loc[train_obs['station_id'].astype(str).tolist()],
                        )
                fitted = estimator.fit(fold_job)
                #Predict with distance correction
                fold_prediction = estimator.predict(
                        fitted,
                        test_obs,
                        fold_distances,
                        lam_value=param_combination['lambda'],
                    )
                predicted_values = fold_prediction.prediction.values
                
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
                        
        param_combination_results = pd.concat(rows)
        results.append(param_combination_results)

        if min_score_improvement is None:
            continue
        else:
            current_score = root_mean_squared_error(param_combination_results['observed'], param_combination_results['predicted'])
            current_score = round(current_score, 2) if current_score is not None else None

            if _is_marginal_improvement(current_score, best_param_score, min_improvement = min_score_improvement):
                logger.debug("Setting best score from %s to %s", best_param_score, current_score)
                best_param_score = current_score
                stalled_steps = 0
            else:
                stalled_steps += 1
                logger.debug(
                    "No score improvement for parameter combination %s: %s (candidate) vs %s (current). Stalled steps: %s", 
                    param_combination, current_score, best_param_score, stalled_steps)
        
        if stalled_steps >= improvement_latency_steps:
            logger.debug("Stopping parameter search because improvement latency has been reached at score %s", best_param_score)
            search_stopped = True
            break

    return CrossValidationResult(
        fold_results = pd.concat(results),
        param_grid = param_grid,
        n_splits = n_splits,
        search_stopped = search_stopped,
    )