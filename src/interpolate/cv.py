from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .distance import DistanceField
from .interpolator import FittedInterpolator, InterpolationJob, Interpolator

logger = logging.getLogger(__name__)

_SUPPORTED_SCOPES = {"vertical", "residual", "overall"}
_SUPPORTED_SCORING = {"mae", "rmse", "bias", "mbe"}


@dataclass(frozen=True)
class CrossValidationResult:
    fold_results: pd.DataFrame
    summary: pd.DataFrame
    best_params: dict[str, Any]
    best_score: float | None
    select_by: str
    select_scope: str
    early_stopped: bool = False
    stop_reason: str | None = None
    evaluated_params: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class _FoldContext:
    fold_idx: int
    test_id: str
    observed: float
    vertical_pred: float
    fitted: FittedInterpolator | None
    test_obs: pd.DataFrame
    fold_distances: xr.DataArray | None


def _as_list(value, *, default: list | None = None) -> list:
    if value is None:
        return [] if default is None else list(default)
    if isinstance(value, str):
        return [value]
    return list(value)


def _parameter_candidates(param_grid: dict[str, Sequence[Any]] | None) -> list[dict[str, Any]]:
    if not param_grid:
        return [{}]

    if set(param_grid) != {"lambda"}:
        raise ValueError(
            "Only lambda tuning is implemented for cross-validation right now. "
            f"Got parameter grid keys: {sorted(param_grid)}"
        )

    lam_values = list(param_grid["lambda"])
    if not lam_values:
        raise ValueError("param_grid['lambda'] must contain at least one value.")
    return [{"lambda": value} for value in lam_values]


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


def _fold_indices(n_samples: int, cv: str | Iterable[tuple[Sequence[int], Sequence[int]]]):
    if isinstance(cv, str):
        if cv not in {"loo", "leave_one_out"}:
            raise ValueError(f"Unsupported cv strategy '{cv}'. Only leave-one-out is implemented.")
        indices = np.arange(n_samples)
        for idx in indices:
            train = indices[indices != idx]
            test = np.array([idx])
            yield train, test
        return

    for train, test in cv:
        yield np.asarray(train, dtype=int), np.asarray(test, dtype=int)


def _score_summary(fold_results: pd.DataFrame, scoring: list[str]) -> pd.DataFrame:
    if fold_results.empty:
        return pd.DataFrame()

    group_cols = ["scope", "lambda"]
    grouped = fold_results.groupby(group_cols, dropna=False)
    rows = []
    for key, group in grouped:
        scope, lam_value = key
        row = {
            "scope": scope,
            "lambda": lam_value,
            "n": int(group["error"].notna().sum()),
        }
        errors = group["error"].dropna().to_numpy(dtype=float)
        for scorer in scoring:
            if errors.size == 0:
                row[scorer] = np.nan
            elif scorer == "mae":
                row[scorer] = float(np.mean(np.abs(errors)))
            elif scorer == "rmse":
                row[scorer] = float(np.sqrt(np.mean(errors**2)))
            elif scorer in {"bias", "mbe"}:
                row[scorer] = float(np.mean(errors))
        rows.append(row)

    return pd.DataFrame(rows)


def _score_errors(errors: Sequence[float], scorer: str) -> float:
    values = pd.Series(errors).dropna().to_numpy(dtype=float)
    if values.size == 0:
        return np.nan
    if scorer == "mae":
        return float(np.mean(np.abs(values)))
    if scorer == "rmse":
        return float(np.sqrt(np.mean(values**2)))
    if scorer in {"bias", "mbe"}:
        return float(np.mean(values))
    raise ValueError(f"Cannot score errors with unknown score '{scorer}'.")


def _is_material_improvement(
    previous_best: float,
    candidate_score: float,
    *,
    min_improvement: float,
    mode: str,
) -> bool:
    if not np.isfinite(previous_best) or not np.isfinite(candidate_score):
        return False
    improvement = previous_best - candidate_score
    if improvement <= 0:
        return False
    if mode == "absolute":
        return improvement >= min_improvement
    denominator = max(abs(previous_best), np.finfo(float).eps)
    return improvement / denominator >= min_improvement


def _select_best(
    summary: pd.DataFrame,
    *,
    select_by: str,
    select_scope: str,
) -> tuple[dict[str, Any], float | None]:
    if summary.empty:
        return {}, None
    if select_by not in summary.columns:
        raise ValueError(f"Cannot select best parameters by unknown score '{select_by}'.")

    candidates = summary[summary["scope"] == select_scope].copy()
    candidates = candidates.dropna(subset=[select_by])
    if candidates.empty:
        return {}, None

    best_row = candidates.sort_values([select_by, "lambda"], kind="mergesort").iloc[0]
    best_params: dict[str, Any] = {}
    if pd.notna(best_row["lambda"]):
        best_params["lambda"] = best_row["lambda"].item() if hasattr(best_row["lambda"], "item") else best_row["lambda"]
    return best_params, float(best_row[select_by])


def _make_job_like(job: InterpolationJob, observations: pd.DataFrame) -> InterpolationJob:
    station_ids = observations["station_id"].astype(str).tolist()
    return InterpolationJob(
        timestamp=job.timestamp,
        parameter=job.parameter,
        observations=observations,
        training_points=job.training_points.loc[station_ids],
    )

class CrossValidator:
    def __init__(
        self,
        enabled: bool = False,
        cv: str | Iterable[tuple[Sequence[int], Sequence[int]]] = "loo",
        scoring: str | Sequence[str] = ("mae",),
        scopes: Sequence[str] = ("overall",),
        param_grid: dict[str, Sequence[Any]] | None = None,
        refit_vertical_per_fold: bool = False,
        select_by: str = "mae",
        select_scope: str = "overall",
        error_score: float | str = np.nan,
        min_lambda_improvement: float | None = None,
        lambda_improvement_patience: int = 2,
        lambda_improvement_mode: str = "relative",
    ):
        self.enabled = enabled
        self.cv = cv

        self.scoring = _as_list(scoring, default=["mae"])
        unknown_scores = set(self.scoring) - _SUPPORTED_SCORING
        if unknown_scores:
            raise ValueError(f"Unsupported scoring values {sorted(unknown_scores)}. Available: {sorted(_SUPPORTED_SCORING)}")
        if select_by not in self.scoring:
            self.scoring.append(select_by)

        self.scopes = _as_list(scopes, default=["overall"])
        unknown_scopes = set(self.scopes) - _SUPPORTED_SCOPES
        if unknown_scopes:
            raise ValueError(f"Unsupported CV scopes {sorted(unknown_scopes)}. Available: {sorted(_SUPPORTED_SCOPES)}")
        if select_scope not in self.scopes:
            raise ValueError(f"select_scope '{select_scope}' must be included in scopes {self.scopes}.")

        self.candidates = _parameter_candidates(param_grid)
        self.refit_vertical_per_fold = refit_vertical_per_fold
        self.select_by = select_by
        self.select_scope = select_scope
        self.error_score = error_score
        if min_lambda_improvement is not None and min_lambda_improvement < 0:
            raise ValueError("min_lambda_improvement must be >= 0.")
        if lambda_improvement_patience < 1:
            raise ValueError("lambda_improvement_patience must be >= 1.")
        if lambda_improvement_mode not in {"relative", "absolute"}:
            raise ValueError("lambda_improvement_mode must be either 'relative' or 'absolute'.")
        self.min_lambda_improvement = min_lambda_improvement
        self.lambda_improvement_patience = int(lambda_improvement_patience)
        self.lambda_improvement_mode = lambda_improvement_mode

    def cross_validate(
        self,
        estimator: Interpolator,
        job: InterpolationJob,
        distance_fields: DistanceField | xr.DataArray | None = None,

    ) -> CrossValidationResult | None:
        if not self.enabled:
            return None

        if not isinstance(estimator, Interpolator):
            raise TypeError(f"cross_validate estimator must be an Interpolator. Got {type(estimator)}")
        if not isinstance(job, InterpolationJob):
            raise TypeError(f"cross_validate job must be an InterpolationJob. Got {type(job)}")


        needs_distances = bool({"overall", "residual"} & set(self.scopes))
        if needs_distances and estimator.residual_model is not None and distance_fields is None:
            raise ValueError("Overall or residual cross-validation requires distance_fields when a residual model is configured.")

        observations = job.observations.reset_index(drop=True)
        point_distances = _point_distances(distance_fields, observations) if needs_distances else None

        rows = []
        fold_contexts: list[_FoldContext] = []
        full_fit = estimator.fit(job) if not self.refit_vertical_per_fold else None

        for fold_idx, (train_idx, test_idx) in enumerate(_fold_indices(len(observations), self.cv)):
            if len(test_idx) != 1:
                raise ValueError("Only one held-out station per fold is currently supported.")

            train_obs = observations.iloc[train_idx].copy()
            test_obs = observations.iloc[test_idx].copy()
            test_obs.attrs["crs"] = job.training_points.crs
            test_id = str(test_obs["station_id"].iloc[0])
            observed = float(test_obs[job.parameter].iloc[0])
            train_ids = train_obs["station_id"].to_numpy(dtype=str)

            try:
                if self.refit_vertical_per_fold:
                    fitted = estimator.fit(_make_job_like(job, train_obs))
                else:
                    fitted = FittedInterpolator(
                        vertical_fit=full_fit.vertical_fit,
                        residuals=full_fit.residuals.sel(id=train_ids),
                        job=_make_job_like(job, train_obs),
                    )

                test_X = test_obs["elevation"].to_numpy(dtype=float).reshape(-1, 1)
                vertical_pred = float(np.asarray(fitted.vertical_fit.predict(test_X), dtype=float).reshape(-1)[0])

                if "vertical" in self.scopes:
                    rows.append(
                        {
                            "fold": fold_idx,
                            "station_id": test_id,
                            "scope": "vertical",
                            "lambda": np.nan,
                            "observed": observed,
                            "predicted": vertical_pred,
                            "error": vertical_pred - observed,
                        }
                    )

                fold_distances = (
                    point_distances.sel(target_id=[test_id])
                    if point_distances is not None
                    else None
                )
                fold_contexts.append(
                    _FoldContext(
                        fold_idx=fold_idx,
                        test_id=test_id,
                        observed=observed,
                        vertical_pred=vertical_pred,
                        fitted=fitted,
                        test_obs=test_obs,
                        fold_distances=fold_distances,
                    )
                )

            except Exception as e:
                if self.error_score == "raise":
                    raise
                logger.warning("Cross-validation fold %s for station %s failed: %s", fold_idx, test_id, e)
                error_value = float(self.error_score)
                if "vertical" in self.scopes:
                    rows.append(
                        {
                            "fold": fold_idx,
                            "station_id": test_id,
                            "scope": "vertical",
                            "lambda": np.nan,
                            "observed": observed,
                            "predicted": error_value,
                            "error": error_value,
                        }
                    )
                fold_contexts.append(
                    _FoldContext(
                        fold_idx=fold_idx,
                        test_id=test_id,
                        observed=observed,
                        vertical_pred=np.nan,
                        fitted=None,
                        test_obs=test_obs,
                        fold_distances=None,
                    )
                )

        evaluated_candidates: list[dict[str, Any]] = []
        lambda_search_stopped = False
        lambda_stop_reason = None
        stop_best_score: float | None = None
        stop_best_params: dict[str, Any] | None = None
        stale_steps = 0
        evaluate_lambda_candidates = bool({"overall", "residual"} & set(self.scopes))
        lambda_candidates = list(self.candidates)
        if self.min_lambda_improvement is not None and lambda_candidates and "lambda" in lambda_candidates[0]:
            lambda_candidates = sorted(lambda_candidates, key=lambda params: params["lambda"])

        if evaluate_lambda_candidates:
            for params in lambda_candidates:
                lam_value = params.get("lambda")
                evaluated_candidates.append(dict(params))
                candidate_errors = []

                for context in fold_contexts:
                    try:
                        if context.fitted is None:
                            raise RuntimeError("fold fitting failed")
                        overall_pred = float(
                            estimator.predict(
                                context.fitted,
                                context.test_obs,
                                distance_fields=context.fold_distances,
                                lam_value=lam_value,
                            ).prediction.iloc[0]
                        )
                    except Exception as e:
                        if self.error_score == "raise":
                            raise
                        logger.warning(
                            "Cross-validation fold %s for station %s and lambda %s failed: %s",
                            context.fold_idx,
                            context.test_id,
                            lam_value,
                            e,
                        )
                        overall_pred = float(self.error_score)
                        error = float(self.error_score)
                    else:
                        error = overall_pred - context.observed

                    if "overall" in self.scopes:
                        rows.append(
                            {
                                "fold": context.fold_idx,
                                "station_id": context.test_id,
                                "scope": "overall",
                                "lambda": lam_value,
                                "observed": context.observed,
                                "predicted": overall_pred,
                                "error": error,
                            }
                        )
                    if "residual" in self.scopes:
                        rows.append(
                            {
                                "fold": context.fold_idx,
                                "station_id": context.test_id,
                                "scope": "residual",
                                "lambda": lam_value,
                                "observed": context.observed - context.vertical_pred,
                                "predicted": overall_pred - context.vertical_pred,
                                "error": error,
                            }
                        )
                    if self.select_scope in {"overall", "residual"}:
                        candidate_errors.append(error)

                if self.min_lambda_improvement is None or self.select_scope not in {"overall", "residual"}:
                    continue

                candidate_score = _score_errors(candidate_errors, self.select_by)
                if stop_best_score is None or not np.isfinite(stop_best_score):
                    stop_best_score = candidate_score
                    stop_best_params = dict(params)
                    stale_steps = 0
                    continue
                if _is_material_improvement(
                    stop_best_score,
                    candidate_score,
                    min_improvement=self.min_lambda_improvement,
                    mode=self.lambda_improvement_mode,
                ):
                    stop_best_score = candidate_score
                    stop_best_params = dict(params)
                    stale_steps = 0
                else:
                    stale_steps += 1

                if stale_steps >= self.lambda_improvement_patience:
                    lambda_search_stopped = True
                    lambda_stop_reason = (
                        "lambda search stopped after "
                        f"{stale_steps} consecutive candidate(s) without "
                        f"{self.lambda_improvement_mode} improvement >= {self.min_lambda_improvement}"
                    )
                    break

        fold_results = pd.DataFrame(rows)
        if not fold_results.empty:
            fold_results["absolute_error"] = fold_results["error"].abs()
            fold_results["squared_error"] = fold_results["error"] ** 2

        summary = _score_summary(fold_results, self.scoring)
        if (
            self.min_lambda_improvement is not None
            and self.select_scope in {"overall", "residual"}
            and stop_best_params is not None
        ):
            best_params = stop_best_params
            best_score = float(stop_best_score) if stop_best_score is not None else None
        else:
            best_params, best_score = _select_best(summary, select_by=self.select_by, select_scope=self.select_scope)
        return CrossValidationResult(
            fold_results=fold_results,
            summary=summary,
            best_params=best_params,
            best_score=best_score,
            select_by=self.select_by,
            select_scope=self.select_scope,
            early_stopped=lambda_search_stopped,
            stop_reason=lambda_stop_reason,
            evaluated_params=tuple(evaluated_candidates),
        )


def cross_validate(
    estimator: Interpolator,
    job: InterpolationJob,
    *,
    distance_fields: DistanceField | xr.DataArray | None = None,
    cv: str | Iterable[tuple[Sequence[int], Sequence[int]]] = "loo",
    scoring: str | Sequence[str] = ("mae",),
    scopes: Sequence[str] = ("overall",),
    param_grid: dict[str, Sequence[Any]] | None = None,
    refit_vertical_per_fold: bool = False,
    select_by: str = "mae",
    select_scope: str = "overall",
    error_score: float | str = np.nan,
    min_lambda_improvement: float | None = None,
    lambda_improvement_patience: int = 2,
    lambda_improvement_mode: str = "relative",
) -> CrossValidationResult:
    validator = CrossValidator(
        enabled=True,
        cv=cv,
        scoring=scoring,
        scopes=scopes,
        param_grid=param_grid,
        refit_vertical_per_fold=refit_vertical_per_fold,
        select_by=select_by,
        select_scope=select_scope,
        error_score=error_score,
        min_lambda_improvement=min_lambda_improvement,
        lambda_improvement_patience=lambda_improvement_patience,
        lambda_improvement_mode=lambda_improvement_mode,
    )
    return validator.cross_validate(estimator, job, distance_fields=distance_fields)
