from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import geopandas as gpd
import rioxarray  # noqa: F401
import xarray as xr
from shapely.geometry import Point

from src.array.cache import CacheManager
from src.domain.dem import DEM
from src.interpolate import cross_validate
from src.interpolate.distance import DistanceField
from src.interpolate.idw import InverseDistanceWeighting
from src.interpolate.interpolator import InterpolationJob, Interpolator
from src.interpolate.vertical import LinearVerticalModel


def _lambda_search_fixture():
    observations = pd.DataFrame(
        {
            "station_id": ["a", "b", "c", "d"],
            "elevation": [0.0, 0.0, 0.0, 0.0],
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "tair_2m": [0.0, 0.0, 10.0, 10.0],
        }
    )
    good_distances = np.array(
        [
            [[0.0, 0.1, 10.0, 10.0]],
            [[0.1, 0.0, 10.0, 10.0]],
            [[10.0, 10.0, 0.0, 0.1]],
            [[10.0, 10.0, 0.1, 0.0]],
        ],
        dtype=float,
    )
    mixed_distances = np.array(
        [
            [[0.0, 10.0, 10.0, 10.0]],
            [[10.0, 0.0, 10.0, 10.0]],
            [[0.1, 0.1, 0.0, 0.1]],
            [[0.1, 0.1, 0.1, 0.0]],
        ],
        dtype=float,
    )
    bad_distances = np.array(
        [
            [[0.0, 10.0, 0.1, 0.1]],
            [[10.0, 0.0, 0.1, 0.1]],
            [[0.1, 0.1, 0.0, 10.0]],
            [[0.1, 0.1, 10.0, 0.0]],
        ],
        dtype=float,
    )
    distances = DistanceField(
        "test_distance",
        xr.DataArray(
            np.stack([bad_distances, mixed_distances, good_distances, good_distances]),
            dims=("lam_value", "id", "y", "x"),
            coords={
                "lam_value": [0, 1, 2, 3],
                "id": ["a", "b", "c", "d"],
                "y": [0.0],
                "x": [0.0, 1.0, 2.0, 3.0],
            },
        ),
    )
    interpolator = Interpolator(
        vertical_model=LinearVerticalModel(),
        residual_model=InverseDistanceWeighting(neighbours=1),
        min_sample_size=3,
    )
    return interpolator, _job(observations), distances


def _dem(data: xr.DataArray) -> DEM:
    return DEM(
        path=Path("memory"),
        data=data,
        fingerprint=CacheManager.array_fingerprint(data),
    )


def _job(observations: pd.DataFrame) -> InterpolationJob:
    training_points = gpd.GeoDataFrame(
        observations[["station_id", "elevation"]].copy(),
        geometry=[Point(x, y) for x, y in zip(observations["x"], observations["y"])],
        crs=4326,
    ).set_index("station_id")
    return InterpolationJob(
        timestamp=pd.Timestamp("2026-05-13"),
        parameter="tair_2m",
        observations=observations,
        training_points=training_points,
    )


def test_cross_validate_selects_best_lambda_from_leave_one_out():
    grid = xr.DataArray(
        [[0.0, 0.0, 0.0, 0.0]],
        dims=("y", "x"),
        coords={"y": [0.0], "x": [0.0, 1.0, 2.0, 3.0]},
    ).rio.write_crs(4326)
    observations = pd.DataFrame(
        {
            "station_id": ["a", "b", "c", "d"],
            "elevation": [0.0, 0.0, 0.0, 0.0],
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "tair_2m": [0.0, 0.0, 10.0, 10.0],
        }
    )
    good_distances = np.array(
        [
            [[0.0, 0.1, 10.0, 10.0]],
            [[0.1, 0.0, 10.0, 10.0]],
            [[10.0, 10.0, 0.0, 0.1]],
            [[10.0, 10.0, 0.1, 0.0]],
        ],
        dtype=float,
    )
    bad_distances = np.array(
        [
            [[0.0, 10.0, 0.1, 0.1]],
            [[10.0, 0.0, 0.1, 0.1]],
            [[0.1, 0.1, 0.0, 10.0]],
            [[0.1, 0.1, 10.0, 0.0]],
        ],
        dtype=float,
    )
    distances = DistanceField(
        "test_distance",
        xr.DataArray(
            np.stack([bad_distances, good_distances]),
            dims=("lam_value", "id", "y", "x"),
            coords={
                "lam_value": [0, 1],
                "id": ["a", "b", "c", "d"],
                "y": [0.0],
                "x": [0.0, 1.0, 2.0, 3.0],
            },
        ),
    )
    interpolator = Interpolator(
        vertical_model=LinearVerticalModel(),
        residual_model=InverseDistanceWeighting(neighbours=1),
        min_sample_size=3,
    )

    result = cross_validate(
        interpolator,
        _job(observations),
        distance_fields=distances,
        scoring=["mae", "rmse"],
        scopes=["overall"],
        param_grid={"lambda": [0, 1]},
        refit_vertical_per_fold=False,
    )

    assert result.best_params == {"lambda": 1}
    assert result.best_score == pytest.approx(0.0)
    assert len(result.fold_results) == 8
    assert set(result.summary["lambda"].tolist()) == {0, 1}


def test_cross_validate_stops_lambda_search_when_improvement_is_below_threshold():
    interpolator, job, distances = _lambda_search_fixture()

    result = cross_validate(
        interpolator,
        job,
        distance_fields=distances,
        scopes=["overall"],
        param_grid={"lambda": [0, 1, 2, 3]},
        min_lambda_improvement=0.6,
        lambda_improvement_patience=1,
    )

    assert result.early_stopped is True
    assert result.best_params == {"lambda": 0}
    assert result.best_score == pytest.approx(10.0)
    assert [params["lambda"] for params in result.evaluated_params] == [0, 1]
    assert set(result.summary["lambda"].tolist()) == {0, 1}


def test_cross_validate_lambda_search_patience_allows_later_improvement():
    interpolator, job, distances = _lambda_search_fixture()

    result = cross_validate(
        interpolator,
        job,
        distance_fields=distances,
        scopes=["overall"],
        param_grid={"lambda": [0, 1, 2, 3]},
        min_lambda_improvement=0.6,
        lambda_improvement_patience=2,
    )

    assert result.early_stopped is False
    assert result.best_params == {"lambda": 2}
    assert result.best_score == pytest.approx(0.0)
    assert [params["lambda"] for params in result.evaluated_params] == [0, 1, 2, 3]


def test_cross_validate_vertical_scope_without_distance_fields():
    grid = xr.DataArray(
        [[0.0, 1.0, 2.0]],
        dims=("y", "x"),
        coords={"y": [0.0], "x": [0.0, 1.0, 2.0]},
    ).rio.write_crs(4326)
    observations = pd.DataFrame(
        {
            "station_id": ["a", "b", "c"],
            "elevation": [0.0, 1.0, 2.0],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "tair_2m": [0.0, 1.0, 2.0],
        }
    )
    interpolator = Interpolator(
        vertical_model=LinearVerticalModel(),
        min_sample_size=2,
    )

    result = cross_validate(
        interpolator,
        _job(observations),
        scoring="mae",
        scopes=["vertical"],
        select_scope="vertical",
        refit_vertical_per_fold=True,
    )

    assert result.best_params == {}
    assert result.best_score == pytest.approx(0.0)
    assert result.fold_results["scope"].unique().tolist() == ["vertical"]
