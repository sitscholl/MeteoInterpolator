import numpy as np
import pandas as pd
import pytest
import geopandas as gpd
import xarray as xr
from pydantic import ValidationError
from shapely.geometry import Point

from src.interpolate import cross_validate
from src.interpolate.cv import _build_target_grid
from src.interpolate.distance import DistanceField
from src.interpolate.models import IDWModel
from src.interpolate.interpolator import InterpolationJob, Interpolator
from src.interpolate.models import LinearLapseRateModel
from src.config import CrossValidationConfig


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
        primary_model=LinearLapseRateModel(),
        residual_model=IDWModel(neighbours=1),
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
        primary_model=LinearLapseRateModel(),
        residual_model=IDWModel(neighbours=1),
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
    # Selection excludes candidates whose improvement failed the threshold,
    # while their scores remain available with include_stalled=True.
    assert best_params == {"lam_value": 0.0}
    assert best_score == pytest.approx(10.0)
    scores = result.calculate_fold_scores(include_stalled=True)
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
    with pytest.raises(ValidationError) as error:
        CrossValidationConfig(
            enabled=True,
            param_grid={"lam_value": [0, 1]},
            select_by="bias",
        )
    assert error.value.errors()[0]["loc"] == ("select_by",)
    assert error.value.errors()[0]["type"] == "literal_error"


def test_cross_validation_config_disabled_defaults_need_no_grid():
    config = CrossValidationConfig()

    assert not config.enabled
    assert config.param_grid is None
    assert config.select_by == "rmse"
    assert config.improvement_latency_steps == 3


def test_cross_validation_config_normalizes_and_preserves_settings():
    config = CrossValidationConfig.model_validate({
        "enabled": True,
        "param_grid": {"lam_value": [0, "1", 2.5]},
        "n_splits": 3,
        "min_relative_score_improvement": 0.1,
        "improvement_latency_steps": 2,
        "select_by": "MAE",
    })

    assert config.param_grid == {"lam_value": [0.0, 1.0, 2.5]}
    assert config.n_splits == 3
    assert config.min_relative_score_improvement == 0.1
    assert config.improvement_latency_steps == 2
    assert config.select_by == "mae"


@pytest.mark.parametrize("grid", [None, {}, {"lambda": [0]}, {"lam_value": []}])
def test_enabled_cross_validation_requires_nonempty_supported_grid(grid):
    with pytest.raises(ValidationError):
        CrossValidationConfig(enabled=True, param_grid=grid)


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
    original_run = interpolator.run
    predicted_folds = []

    def check_run(job, targets, distance_fields, **kwargs):
        ids = targets.target_id.values.tolist()
        held_out = observations.set_index("station_id").loc[ids]
        assert set(targets.dims) == {"target_id"}
        assert targets.rio.crs == job.training_points.crs
        np.testing.assert_array_equal(targets.elevation.values, held_out["elevation"])
        np.testing.assert_array_equal(targets.x.values, held_out["x"])
        np.testing.assert_array_equal(targets.y.values, held_out["y"])
        assert set(ids).isdisjoint(job.observations.station_id)
        assert set(job.training_points.index) == set(job.observations.station_id)
        prediction = original_run(job, targets, distance_fields, **kwargs)
        assert prediction.prediction.dims == ("target_id",)
        np.testing.assert_array_equal(prediction.prediction.target_id.values, ids)
        predicted_folds.append(ids)
        return prediction

    monkeypatch.setattr(interpolator, "run", check_run)
    result = cross_validate(
        interpolator, job, {"lam_value": [2]}, distances, n_splits=2
    )

    assert predicted_folds == [["c", "a"], ["d", "b"]]
    assert result.n_splits == 2
    assert result.fold_results["test_id"].tolist() == ["c", "a", "d", "b"]
    np.testing.assert_allclose(result.fold_results["observed"], [10, 0, 10, 0])
    np.testing.assert_allclose(result.fold_results["predicted"], [10, 0, 10, 0])
    pd.testing.assert_frame_equal(job.observations, observations)


def test_cross_validate_uses_elevation_for_primary_prediction():
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
    "options, location",
    [
        ({"param_grid": {"lam_value": "1"}}, ("param_grid", "lam_value")),
        ({"param_grid": {"lam_value": [np.nan]}}, ()),
        ({"param_grid": {"lam_value": [np.inf]}}, ()),
        ({"n_splits": 1}, ("n_splits",)),
        ({"n_splits": True}, ("n_splits",)),
        ({"n_splits": 1.5}, ("n_splits",)),
        ({"improvement_latency_steps": 0}, ("improvement_latency_steps",)),
        ({"improvement_latency_steps": True}, ("improvement_latency_steps",)),
        ({"min_relative_score_improvement": -0.1}, ("min_relative_score_improvement",)),
        ({"min_relative_score_improvement": np.nan}, ("min_relative_score_improvement",)),
        ({"min_relative_score_improvement": np.inf}, ("min_relative_score_improvement",)),
        ({"metric": "bias"}, ("select_by",)),
        ({"metric": None}, ("select_by",)),
    ],
)
def test_cross_validate_rejects_invalid_options(options, location):
    interpolator, job, distances = _lambda_search_fixture()
    kwargs = {"param_grid": {"lam_value": [0]}, "distance_fields": distances, **options}
    with pytest.raises(ValidationError) as error:
        cross_validate(interpolator, job, **kwargs)
    assert error.value.errors()[0]["loc"] == location


@pytest.mark.parametrize("options,error,message", [
    ({"distance_fields": None}, TypeError, "Distance fields must be"),
])
def test_cross_validate_checks_job_requirements_after_config(options, error, message):
    interpolator, job, distances = _lambda_search_fixture()
    kwargs = {"param_grid": {"lam_value": [0]}, "distance_fields": distances, **options}
    with pytest.raises(error, match=message):
        cross_validate(interpolator, job, **kwargs)


def test_cross_validate_requires_distance_dependent_model():
    interpolator, job, distances = _lambda_search_fixture()
    interpolator.residual_model = None
    with pytest.raises(ValueError, match="requires no distance tuning"):
        cross_validate(interpolator, job, {"lam_value": [0]}, distances)


def test_cross_validate_standalone_idw_excludes_held_out_sources():
    interpolator, job, distances = _lambda_search_fixture()
    interpolator.primary_model = IDWModel(neighbours=1)
    interpolator.residual_model = None
    result = cross_validate(interpolator, job, {"lam_value": [0, 2]}, distances)
    assert result.get_best_parameters() == ({"lam_value": 2}, 0.)
    # Self-distance is zero in every field: including held-out sources would
    # incorrectly make even the bad lambda score zero.
    assert result.calculate_fold_scores().loc[0] == pytest.approx(10.)


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
    np.testing.assert_array_equal(result.calculate_fold_scores(include_stalled=True).values, [0, 0])


@pytest.mark.parametrize("eligible_count", [0, 1])
@pytest.mark.parametrize("cold_pool_only", [False, True])
def test_cv_rejects_fewer_than_two_eligible_stations(monkeypatch, eligible_count, cold_pool_only):
    from unittest.mock import Mock
    interpolator, job, distances = _lambda_search_fixture()
    if cold_pool_only:
        job.observations["station_type"] = [
            "cold-pool" if i < eligible_count else None for i in range(4)
        ]
    else:
        job = _job(job.observations.iloc[:eligible_count].copy())
    run = Mock()
    monkeypatch.setattr(interpolator, "run", run)
    with pytest.raises(ValueError, match=f"at least 2 eligible stations.*Got {eligible_count}"):
        cross_validate(interpolator, job, {"lam_value": [0]}, distances,
                       use_only_cold_pool=cold_pool_only)
    run.assert_not_called()


def test_cold_pool_cv_scores_only_eligible_stations_and_keeps_other_training(monkeypatch):
    interpolator, job, distances = _lambda_search_fixture()
    job.observations["station_type"] = ["cold-pool", None, "cold-pool", None]
    original = interpolator.run
    calls = []
    def record(**kwargs):
        training = set(kwargs["job"].observations.station_id)
        held_out = set(kwargs["targets"].target_id.values)
        assert training.isdisjoint(held_out)
        assert {"b", "d"} <= training
        calls.append((training, held_out))
        return original(**kwargs)
    monkeypatch.setattr(interpolator, "run", record)
    result = cross_validate(interpolator, job, {"lam_value": [0, 2]}, distances,
                            use_only_cold_pool=True)
    assert len(calls) == 4
    assert set(result.fold_results.test_id) == {"a", "c"}
    assert result.get_best_parameters() == ({"lam_value": 2}, 0.)
    assert result.cv_station_type == "cold-pool"
    assert result.evaluated_stations == result.n_splits == 2


@pytest.mark.parametrize("cold_pool_only", [False, True])
def test_cv_caps_split_count_to_eligible_stations(caplog, cold_pool_only):
    interpolator, job, distances = _lambda_search_fixture()
    job.observations["station_type"] = ["cold-pool", None, "cold-pool", None]
    result = cross_validate(interpolator, job, {"lam_value": [2]}, distances,
                            n_splits=9, use_only_cold_pool=cold_pool_only)
    assert result.n_splits == (2 if cold_pool_only else 4)
    assert "n_splits is larger" in caplog.text


def test_cold_pool_cv_rejects_insufficient_training_before_fitting(monkeypatch):
    from unittest.mock import Mock
    interpolator, job, distances = _lambda_search_fixture()
    job.observations["station_type"] = ["cold-pool"] * 3 + [None]
    run = Mock()
    monkeypatch.setattr(interpolator, "run", run)
    with pytest.raises(ValueError, match="3 samples are required.*Got 2 for fold 0"):
        cross_validate(interpolator, job, {"lam_value": [0, 2]}, distances,
                       n_splits=2, use_only_cold_pool=True)
    run.assert_not_called()


def test_catalog_to_worker_preserves_classification_and_tunes_subset(monkeypatch):
    from src.database.db import InterpolationDB
    from src.domain.meteo_data import MeteoData
    from src.domain.schemas import _STATION_CATALOG_SCHEMA
    from src.meteo.retrieve import retrieve_stations
    from src.worker import process_interpolation_job

    interpolator, source_job, distances = _lambda_search_fixture()
    start = pd.Timestamp("2026-05-13", tz="UTC")
    end = start + pd.Timedelta(hours=1)
    catalog = pd.DataFrame({
        "station_id": ["a", "b", "c", "d"], "name": ["a", "b", "c", "d"],
        "longitude": [0., 1., 2., 3.], "latitude": [0.] * 4,
        "elevation": [0.] * 4, "crs": [4326] * 4, "status": ["active"] * 4,
        "station_type": ["COLD-POOL", None, "cold-pool", None],
    })
    database = InterpolationDB("sqlite:///:memory:")
    try:
        database.sync_catalog(_STATION_CATALOG_SCHEMA.validate(catalog))
        for row in source_job.observations.itertuples():
            database.add_station_data(row.station_id, pd.DataFrame({
                "datetime": [start], "tair_2m": [row.tair_2m],
            }))
        stations = retrieve_stations(database, catalog.station_id.tolist(), ["tair_2m"], start, end)
        job, = MeteoData.from_list(stations).build_jobs(start, end, "tair_2m")
        result = process_interpolation_job(
            job, interpolator,
            CrossValidationConfig(enabled=True, param_grid={"lam_value": [0, 2]},
                                  use_only_cold_pool=True),
            _build_target_grid(source_job.observations, 4326), distances,
        )
        assert result.status == "completed", result.error
        assert set(result.cv_result.fold_results.test_id) == {"a", "c"}
        assert result.cv_result.get_best_parameters() == ({"lam_value": 2}, 0.)
        assert set(result.prediction.station_ids) == {"a", "b", "c", "d"}
    finally:
        database.close()


@pytest.mark.parametrize("eligible_count,n_splits,error", [
    (0, None, "Got 0"),
    (1, None, "Got 1"),
    (3, 2, "3 samples are required"),
])
def test_worker_marks_invalid_cv_as_failed(eligible_count, n_splits, error, monkeypatch):
    from unittest.mock import Mock
    from src.worker import process_interpolation_job
    interpolator, job, distances = _lambda_search_fixture()
    job.observations["station_type"] = [
        "cold-pool" if i < eligible_count else None for i in range(4)
    ]
    run = Mock()
    monkeypatch.setattr(interpolator, "run", run)
    result = process_interpolation_job(
        job, interpolator,
        CrossValidationConfig(enabled=True, param_grid={"lam_value": [0]},
                              n_splits=n_splits, use_only_cold_pool=True),
        _build_target_grid(job.observations, 4326), distances,
    )
    assert result.status == "failed"
    assert error in result.error
    assert result.prediction is None
    run.assert_not_called()
