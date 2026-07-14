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
) -> CrossValidationResult:
    if not isinstance(estimator, Interpolator):
        raise TypeError(f"cross_validate estimator must be an Interpolator. Got {type(estimator)}")
    if not isinstance(job, InterpolationJob):
        raise TypeError(f"cross_validate job must be an InterpolationJob. Got {type(job)}")

    scoring = _as_list(scoring, default=["mae"])
    unknown_scores = set(scoring) - _SUPPORTED_SCORING
    if unknown_scores:
        raise ValueError(f"Unsupported scoring values {sorted(unknown_scores)}. Available: {sorted(_SUPPORTED_SCORING)}")
    if select_by not in scoring:
        scoring.append(select_by)

    scopes = _as_list(scopes, default=["overall"])
    unknown_scopes = set(scopes) - _SUPPORTED_SCOPES
    if unknown_scopes:
        raise ValueError(f"Unsupported CV scopes {sorted(unknown_scopes)}. Available: {sorted(_SUPPORTED_SCOPES)}")
    if select_scope not in scopes:
        raise ValueError(f"select_scope '{select_scope}' must be included in scopes {scopes}.")

    candidates = _parameter_candidates(param_grid)
    needs_distances = bool({"overall", "residual"} & set(scopes))
    if needs_distances and estimator.residual_model is not None and distance_fields is None:
        raise ValueError("Overall or residual cross-validation requires distance_fields when a residual model is configured.")

    observations = job.observations.reset_index(drop=True)
    point_distances = _point_distances(distance_fields, observations) if needs_distances else None

    rows = []
    full_fit = estimator.fit(job) if not refit_vertical_per_fold else None

    for fold_idx, (train_idx, test_idx) in enumerate(_fold_indices(len(observations), cv)):
        if len(test_idx) != 1:
            raise ValueError("Only one held-out station per fold is currently supported.")

        train_obs = observations.iloc[train_idx].copy()
        test_obs = observations.iloc[test_idx].copy()
        test_obs.attrs["crs"] = job.training_points.crs
        test_id = str(test_obs["station_id"].iloc[0])
        observed = float(test_obs[job.parameter].iloc[0])
        train_ids = train_obs["station_id"].to_numpy(dtype=str)

        try:
            if refit_vertical_per_fold:
                fitted = estimator.fit(_make_job_like(job, train_obs))
            else:
                fitted = FittedInterpolator(
                    vertical_fit=full_fit.vertical_fit,
                    residuals=full_fit.residuals.sel(id=train_ids),
                    job=_make_job_like(job, train_obs),
                )

            test_X = test_obs["elevation"].to_numpy(dtype=float).reshape(-1, 1)
            vertical_pred = float(np.asarray(fitted.vertical_fit.predict(test_X), dtype=float).reshape(-1)[0])

            if "vertical" in scopes:
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

            if {"overall", "residual"} & set(scopes):
                fold_distances = (
                    point_distances.sel(target_id=[test_id])
                    if point_distances is not None
                    else None
                )
                for params in candidates:
                    lam_value = params.get("lambda")
                    overall_pred = float(
                        estimator.predict(
                            fitted,
                            test_obs,
                            distance_fields=fold_distances,
                            lam_value=lam_value,
                        ).iloc[0]
                    )
                    if "overall" in scopes:
                        rows.append(
                            {
                                "fold": fold_idx,
                                "station_id": test_id,
                                "scope": "overall",
                                "lambda": lam_value,
                                "observed": observed,
                                "predicted": overall_pred,
                                "error": overall_pred - observed,
                            }
                        )
                    if "residual" in scopes:
                        rows.append(
                            {
                                "fold": fold_idx,
                                "station_id": test_id,
                                "scope": "residual",
                                "lambda": lam_value,
                                "observed": observed - vertical_pred,
                                "predicted": overall_pred - vertical_pred,
                                "error": overall_pred - observed,
                            }
                        )

        except Exception as e:
            if error_score == "raise":
                raise
            logger.warning("Cross-validation fold %s for station %s failed: %s", fold_idx, test_id, e)
            for scope in scopes:
                scope_candidates = candidates if scope in {"overall", "residual"} else [{}]
                for params in scope_candidates:
                    rows.append(
                        {
                            "fold": fold_idx,
                            "station_id": test_id,
                            "scope": scope,
                            "lambda": params.get("lambda", np.nan),
                            "observed": observed,
                            "predicted": float(error_score),
                            "error": float(error_score),
                        }
                    )

    fold_results = pd.DataFrame(rows)
    if not fold_results.empty:
        fold_results["absolute_error"] = fold_results["error"].abs()
        fold_results["squared_error"] = fold_results["error"] ** 2

    summary = _score_summary(fold_results, scoring)
    best_params, best_score = _select_best(summary, select_by=select_by, select_scope=select_scope)
    return CrossValidationResult(
        fold_results=fold_results,
        summary=summary,
        best_params=best_params,
        best_score=best_score,
        select_by=select_by,
        select_scope=select_scope,
    )
