import numpy as np
import pandas as pd
import pytest
import geopandas as gpd
import xarray as xr
from shapely.geometry import Point

from src.interpolate import cross_validate
from src.interpolate.cv import _build_target_grid
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
        timestamp=pd.Timestamp("2026-05-13", tz="UTC"),
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
    # The threshold controls when to stop, while selection retains the best
    # score among all evaluated candidates, including sub-threshold improvements.
    assert best_params == {"lam_value": 1}
    assert best_score == pytest.approx(np.sqrt(50.0))
    scores = result.calculate_fold_scores()
    assert scores.loc[0] == pytest.approx(10.0)
    assert 0 < (scores.loc[0] - scores.loc[1]) / scores.loc[0] < 0.6
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


@pytest.mark.parametrize("off_grid", [False, True])
def test_cross_validate_multiple_targets_preserves_order_and_excludes_held_out_stations(
    monkeypatch, off_grid
):
    interpolator, original_job, distances = _lambda_search_fixture()
    interpolator.min_sample_size = 2
    observations = original_job.observations.iloc[[2, 0, 3, 1]].copy()
    observations.index = [40, 10, 30, 20]
    distances.data = distances.data.reindex(y=[0.0, 1.0], method="nearest")
    observations["y"] = [0.0, 1.0, 0.0, 1.0]
    if off_grid:
        observations["x"] += [0.1, 0.1, -0.1, 0.1]
        observations["y"] += [0.1, -0.1, 0.1, -0.1]
    job = _job(observations)
    original_predict = interpolator.predict
    predicted_folds = []

    def check_predict(model, target_points, distance_fields, **kwargs):
        ids = target_points.target_id.values.tolist()
        held_out = observations.set_index("station_id").loc[ids]
        assert target_points.dims == ("target_id",)
        assert target_points.rio.crs == job.training_points.crs
        np.testing.assert_array_equal(target_points.values, held_out["elevation"])
        np.testing.assert_array_equal(target_points.x.values, held_out["x"])
        np.testing.assert_array_equal(target_points.y.values, held_out["y"])
        assert set(ids).isdisjoint(model.residuals.id.values)
        assert set(model.job.training_points.index) == set(model.residuals.id.values)
        np.testing.assert_array_equal(distance_fields.target_id.values, ids)
        prediction = original_predict(model, target_points, distance_fields, **kwargs)
        assert prediction.prediction.dims == ("target_id",)
        np.testing.assert_array_equal(prediction.prediction.target_id.values, ids)
        predicted_folds.append(ids)
        return prediction

    monkeypatch.setattr(interpolator, "predict", check_predict)
    result = cross_validate(
        interpolator, job, {"lam_value": [2]}, distances, n_splits=2
    )

    assert predicted_folds == [["c", "a"], ["d", "b"]]
    assert result.n_splits == 2
    assert result.fold_results["test_id"].tolist() == ["c", "a", "d", "b"]
    np.testing.assert_allclose(result.fold_results["observed"], [10, 0, 10, 0])
    np.testing.assert_allclose(result.fold_results["predicted"], [10, 0, 10, 0])
    pd.testing.assert_frame_equal(job.observations, observations)


def test_cross_validate_uses_elevation_for_vertical_prediction():
    interpolator, original_job, distances = _lambda_search_fixture()
    observations = original_job.observations.copy()
    observations["elevation"] = [100.0, 500.0, 250.0, 900.0]
    observations["tair_2m"] = 20 - 0.006 * observations["elevation"]

    result = cross_validate(interpolator, _job(observations), {"lam_value": [2]}, distances)

    np.testing.assert_allclose(result.fold_results["predicted"], observations["tair_2m"])
    assert result.get_best_parameters()[1] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("column", ["elevation", "station_id", "x", "y"])
def test_build_target_grid_rejects_missing_columns(column):
    _, job, _ = _lambda_search_fixture()
    with pytest.raises(ValueError, match=column):
        _build_target_grid(job.observations.drop(columns=column), crs=4326)


@pytest.mark.parametrize(
    "options, error, message",
    [
        ({"param_grid": {"lam_value": []}}, ValueError, "cannot be empty"),
        ({"param_grid": {"lam_value": "1"}}, TypeError, "sequence"),
        ({"param_grid": {"lam_value": [np.nan]}}, ValueError, "finite numeric"),
        ({"param_grid": {"lam_value": [np.inf]}}, ValueError, "finite numeric"),
        ({"param_grid": {"lam_value": ["1"]}}, ValueError, "finite numeric"),
        ({"n_splits": 1}, ValueError, "n_splits"),
        ({"n_splits": True}, ValueError, "n_splits"),
        ({"n_splits": 1.5}, ValueError, "n_splits"),
        ({"n_splits": 5}, ValueError, "number of splits"),
        ({"improvement_latency_steps": 0}, ValueError, "improvement_latency_steps"),
        ({"improvement_latency_steps": True}, ValueError, "improvement_latency_steps"),
        ({"min_relative_score_improvement": -0.1}, ValueError, "min_relative"),
        ({"min_relative_score_improvement": np.nan}, ValueError, "min_relative"),
        ({"metric": "bias"}, ValueError, "Unknown metric"),
        ({"metric": None}, TypeError, "metric must be a string"),
        ({"distance_fields": None}, TypeError, "Distance fields must be"),
    ],
)
def test_cross_validate_rejects_invalid_options(options, error, message):
    interpolator, job, distances = _lambda_search_fixture()
    kwargs = {"param_grid": {"lam_value": [0]}, "distance_fields": distances, **options}
    with pytest.raises(error, match=message):
        cross_validate(interpolator, job, **kwargs)


def test_cross_validate_requires_residual_model():
    interpolator, job, distances = _lambda_search_fixture()
    interpolator.residual_model = None
    with pytest.raises(ValueError, match="requires a residual model"):
        cross_validate(interpolator, job, {"lam_value": [0]}, distances)


def test_cross_validate_rejects_folds_with_too_few_training_stations():
    interpolator, job, distances = _lambda_search_fixture()
    with pytest.raises(ValueError, match="3 samples are required.*Got 2 for fold 0"):
        cross_validate(interpolator, job, {"lam_value": [0]}, distances, n_splits=2)


def test_cross_validate_normalizes_mae_and_sorts_lambda_search():
    interpolator, job, distances = _lambda_search_fixture()
    result = cross_validate(
        interpolator, job, {"lam_value": [2, 0, 1]}, distances, metric="MAE"
    )

    assert result.selection_metric == "mae"
    assert result.search_stopped is False
    assert result.fold_results["lam_value"].drop_duplicates().tolist() == [0, 1, 2]
    scores = result.calculate_fold_scores()
    np.testing.assert_allclose(scores.loc[[0, 1, 2]], [10, 5, 0])
    assert result.calculate_fold_scores("rmse").loc[1] == pytest.approx(np.sqrt(50))
    assert result.get_best_parameters() == ({"lam_value": 2}, 0)


def test_cross_validate_stops_on_zero_score_plateau():
    interpolator, job, distances = _lambda_search_fixture()
    result = cross_validate(
        interpolator,
        job,
        {"lam_value": [2, 3]},
        distances,
        min_relative_score_improvement=0.1,
        improvement_latency_steps=1,
    )

    assert result.search_stopped is True
    assert result.get_best_parameters() == ({"lam_value": 2}, 0)
    np.testing.assert_array_equal(result.calculate_fold_scores().values, [0, 0])
