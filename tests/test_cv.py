import numpy as np
import pandas as pd
import pytest
import geopandas as gpd
import xarray as xr
from shapely.geometry import Point

from src.interpolate import cross_validate
from src.interpolate.distance import DistanceField
from src.interpolate.idw import InverseDistanceWeighting
from src.interpolate.interpolator import InterpolationJob, Interpolator
from src.interpolate.vertical import LinearVerticalModel
from src.runtime import CrossValidationConfig


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
        param_grid={"lam_value": [0, 1]},
        metric="rmse",
    )

    best_params, best_score = result.get_best_parameters()

    assert best_params == {"lam_value": 1}
    assert best_score == pytest.approx(0.0)
    assert result.selection_metric == "rmse"
    assert len(result.fold_results) == 8
    assert set(result.fold_results["lam_value"].tolist()) == {0, 1}


def test_cross_validate_stops_lambda_search_when_improvement_is_below_threshold():
    interpolator, job, distances = _lambda_search_fixture()

    result = cross_validate(
        interpolator,
        job,
        distance_fields=distances,
        param_grid={"lam_value": [0, 1, 2, 3]},
        min_relative_score_improvement=0.6,
        improvement_latency_steps=1,
    )

    best_params, best_score = result.get_best_parameters()

    assert result.search_stopped is True
    assert best_params == {"lam_value": 0}
    assert best_score == pytest.approx(10.0)
    assert set(result.fold_results["lam_value"].tolist()) == {0, 1}


def test_cross_validate_lambda_search_patience_allows_later_improvement():
    interpolator, job, distances = _lambda_search_fixture()

    result = cross_validate(
        interpolator,
        job,
        distance_fields=distances,
        param_grid={"lam_value": [0, 1, 2, 3]},
        min_relative_score_improvement=0.6,
        improvement_latency_steps=2,
    )

    best_params, best_score = result.get_best_parameters()

    assert result.search_stopped is False
    assert best_params == {"lam_value": 2}
    assert best_score == pytest.approx(0.0)
    assert set(result.fold_results["lam_value"].tolist()) == {0, 1, 2, 3}


def test_cross_validation_config_rejects_unknown_selection_metric():
    with pytest.raises(ValueError, match="Unknown metric"):
        CrossValidationConfig(
            enabled=True,
            param_grid={"lam_value": [0, 1]},
            select_by="bias",
        )


def test_cross_validate_validates_its_public_options():
    interpolator, job, distances = _lambda_search_fixture()

    with pytest.raises(ValueError, match="exactly one parameter named 'lam_value'"):
        cross_validate(
            interpolator,
            job,
            distance_fields=distances,
            param_grid={"lambda": [0, 1]},
        )
