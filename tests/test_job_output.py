import asyncio
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from pyproj import CRS

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from shapely.geometry import Point

from src.api.schemas import InterpolationRequest
from src.coordinator import InterpolationCoordinator, PreparedInterpolationRun
from src.interpolate.cv import CrossValidationResult
from src.interpolate.distance import DistanceField
from src.interpolate.idw import InverseDistanceWeighting
from src.interpolate.interpolator import InterpolationJob, Interpolator
from src.interpolate.vertical import LinearVerticalModel
from src.meteo.resample import MeteoResampler
from src.runtime import CrossValidationConfig
from src.worker import process_interpolation_job
from src.write import write_job_result


def _result(timestamp="2026-10-25", *, residuals=False, lambda_dimension=True, crs=4326):
    observations = pd.DataFrame({
        "station_id": ["a", "b", "c"], "elevation": [0., 1., 2.],
        "x": [0., 1., 2.], "y": [0., 0., 0.], "tair_2m": [1., 3., 4.],
    })
    stations = gpd.GeoDataFrame(
        observations[["station_id", "elevation"]],
        geometry=[Point(x, 0.) for x in observations.x], crs=crs,
    ).set_index("station_id")
    job = InterpolationJob(
        pd.Timestamp(timestamp, tz="Europe/Rome"), "tair_2m", observations, stations,
    )
    grid = xr.DataArray(
        [[0., 1., 2.], [0., 1., 2.]], dims=("y", "x"),
        coords={"y": [0., 1.], "x": [0., 1., 2.]},
        name="elevation", attrs={"units": "m"},
    ).rio.write_crs(crs)
    distances = None
    if residuals:
        data = xr.DataArray(
            np.hypot(
                np.arange(3)[:, None, None] - np.arange(3)[None, None, :],
                np.arange(2)[None, :, None],
            ),
            dims=("id", "y", "x"),
            coords={"id": ["a", "b", "c"], "y": [0., 1.], "x": [0., 1., 2.]},
        )
        if lambda_dimension:
            data = data.expand_dims(lam_value=[25., 50.])
        distances = DistanceField("test_distance", data)
    result = process_interpolation_job(
        job, Interpolator(
            LinearVerticalModel(), InverseDistanceWeighting(neighbours=1) if residuals else None,
        ), CrossValidationConfig(), grid, distances,
    )
    assert result.status == "completed", result.error
    return result, grid


@pytest.mark.parametrize("day", ["2026-03-29", "2026-05-13", "2026-10-25"])
def test_daily_output_round_trip_preserves_values_time_and_selected_metadata(tmp_path, day):
    result, grid = _result(day)
    path = write_job_result(result, tmp_path, "request-1", {
        "freq": "D", "timezone": "Europe/Rome", "dem_path": Path("dem.tif"),
        "resampling": {"aggregation": "mean", "min_coverage": np.float64(.7)},
    })
    start = pd.Timestamp(day, tz="Europe/Rome").tz_convert("UTC")
    with xr.open_zarr(path, consolidated=False) as ds:
        np.testing.assert_allclose(ds.tair_2m.values[0], result.prediction.prediction.values)
        assert ds.rio.crs == grid.rio.crs
        assert ds.time.values[0] == start.tz_localize(None).to_datetime64()
        assert ds.attrs["request_metadata"]["dem_path"] == "dem.tif"
        assert ds.attrs["request_metadata"]["resampling"]["min_coverage"] == .7
        model = ds.attrs["model_metadata"]["vertical_model"]
        assert model["type"] == "linear"
        assert model["fitted_parameters"]["coef"] == pytest.approx([1.5])
        assert ds.attrs["job_metadata"]["training_stations"] == ["a", "b", "c"]
        assert ds.attrs["job_metadata"]["parameter"] == "tair_2m"
        assert ds.attrs["job_metadata"]["n_observations"] == 3
        assert ds.attrs["residual_summary"]["n"] == 3
        assert not ds.attrs["cv_metadata"]["enabled"]
    assert path.name == f"tair_2m_D_{start:%Y%m%dT%H%M%SZ}.zarr"


def test_repeated_hour_outputs_have_distinct_utc_timestamps_and_names(tmp_path):
    result, _ = _result()
    paths = []
    for timestamp in ("2026-10-25T02:00:00+02:00", "2026-10-25T02:00:00+01:00"):
        hourly = replace(result, job=replace(result.job, timestamp=pd.Timestamp(timestamp)))
        paths.append(write_job_result(hourly, tmp_path, "request", {"freq": "h"}))
    assert paths[0].name == "tair_2m_h_20261025T000000Z.zarr"
    assert paths[1].name == "tair_2m_h_20261025T010000Z.zarr"
    for path, expected in zip(paths, ["2026-10-25T00:00:00", "2026-10-25T01:00:00"]):
        with xr.open_zarr(path, consolidated=False) as ds:
            assert ds.time.values[0] == np.datetime64(expected)


@pytest.mark.parametrize("lambda_dimension", [True, False])
def test_residual_metadata_with_and_without_lambda_coordinate(tmp_path, lambda_dimension):
    result, _ = _result(residuals=True, lambda_dimension=lambda_dimension)
    path = write_job_result(result, tmp_path, "request", {"freq": "D", "timezone": "Europe/Rome"})
    with xr.open_zarr(path, consolidated=False) as ds:
        assert ds.attrs["residuals_interpolated"]
        assert ds.attrs["lam_value"] == (25. if lambda_dimension else None)
        residual_model = ds.attrs["model_metadata"]["residual_model"]
        assert residual_model["used"]
        assert residual_model["settings"]["neighbours"] == 1
        assert residual_model["distance_type"] == "test_distance"
        np.testing.assert_allclose(ds.tair_2m_residuals.values[0], result.prediction.residual_prediction.values)


@pytest.mark.parametrize("crs", [4326, 32632, CRS.from_proj4(
    "+proj=tmerc +lat_0=0 +lon_0=11.1234 +k=0.9997 +x_0=12345 "
    "+y_0=0 +ellps=WGS84 +units=m +no_defs"
)])
@pytest.mark.parametrize("residuals", [False, True])
def test_output_crs_round_trip_with_default_xarray_reader(tmp_path, crs, residuals):
    result, grid = _result(crs=crs, residuals=residuals)
    path = write_job_result(result, tmp_path, "request", {"freq": "D"})
    # No decode_coords override or custom CRS loader should be necessary.
    with xr.open_zarr(path) as ds:
        assert CRS.from_user_input(ds.rio.crs) == CRS.from_user_input(grid.rio.crs)
        assert CRS.from_user_input(ds.attrs["crs"]) == CRS.from_user_input(crs)
        if CRS.from_user_input(crs).to_epsg() is not None:
            assert ds.attrs["crs_epsg"] == f"EPSG:{CRS.from_user_input(crs).to_epsg()}"
        assert ds.rio.bounds() == grid.rio.bounds()
    with xr.open_zarr(path, decode_coords="all", consolidated=False) as ds:
        for name in ("tair_2m", "tair_2m_vertical", "tair_2m_residuals"):
            if name in ds:
                assert CRS.from_user_input(ds[name].rio.crs) == CRS.from_user_input(crs)


def test_output_requires_crs_and_does_not_publish_a_partial_store(tmp_path):
    result, _ = _result()
    prediction = replace(result.prediction,
        prediction=result.prediction.prediction.drop_vars("spatial_ref"),
        vertical_prediction=result.prediction.vertical_prediction.drop_vars("spatial_ref"),
    )
    with pytest.raises(ValueError, match="must define a CRS"):
        write_job_result(replace(result, prediction=prediction), tmp_path, "request", {"freq": "D"})
    assert list((tmp_path / "request").iterdir()) == []


def test_cv_metadata_with_numpy_scalars_round_trips(tmp_path):
    result, _ = _result()
    cv = CrossValidationResult(pd.DataFrame({
        "lam_value": [0, 0, 1, 1], "observed": [1., 2., 1., 2.],
        "predicted": [1., 2., 1.5, 2.5],
    }), {"lam_value": [np.int64(0), np.int64(1)]}, 2, False, "rmse")
    path = write_job_result(replace(result, cv_result=cv), tmp_path, "request", {
        "freq": "D", "timezone": "Europe/Rome",
    })
    with xr.open_zarr(path, consolidated=False) as ds:
        assert ds.attrs["cv_metadata"]["best_parameters"] == {"lam_value": 0}


def test_numpy_metadata_round_trips_on_dataset_and_variables(tmp_path):
    result, _ = _result()
    prediction = result.prediction.prediction.copy()
    prediction.attrs["quality"] = np.float32(.5)
    result = replace(result, prediction=replace(
        result.prediction, prediction=prediction, lam_value=np.int64(25),
    ))
    output = write_job_result(result, tmp_path, "request", {
        "freq": "D", "scalar": np.array(.5), "values": np.array([1, 2]),
    })
    with xr.open_zarr(output, consolidated=False) as ds:
        assert ds.attrs["lam_value"] == 25
        assert ds.attrs["request_metadata"]["scalar"] == .5
        assert ds.attrs["request_metadata"]["values"] == [1, 2]
        assert ds.tair_2m.attrs["quality"] == .5


@pytest.mark.parametrize("status", ["failed", "skipped"])
def test_unsuccessful_jobs_do_not_create_output(tmp_path, status):
    result, _ = _result()
    with pytest.raises(ValueError, match="completed interpolation jobs"):
        write_job_result(replace(result, status=status), tmp_path, "request", {"freq": "D"})
    assert not (tmp_path / "request").exists()


def test_duplicate_output_is_rejected_without_modifying_existing_store(tmp_path):
    result, _ = _result()
    path = write_job_result(result, tmp_path, "request", {"freq": "D"})
    with pytest.raises(ValueError, match="already exists"):
        write_job_result(result, tmp_path, "request", {"freq": "D"})
    with xr.open_zarr(path, consolidated=False) as ds:
        np.testing.assert_allclose(ds.tair_2m.values[0], result.prediction.prediction.values)
    assert list(path.parent.iterdir()) == [path]


def test_failed_publication_cleans_up_temporary_output(tmp_path, monkeypatch):
    def fail_publication(*args):
        raise PermissionError("publication failed")

    monkeypatch.setattr("src.write._publish_store", fail_publication)
    result, _ = _result()
    with pytest.raises(PermissionError, match="publication failed"):
        write_job_result(result, tmp_path, "request", {"freq": "D"})
    assert list((tmp_path / "request").iterdir()) == []


@pytest.mark.parametrize("missing_prediction", [True, False])
def test_result_export_rejects_missing_prediction_or_naive_time(missing_prediction):
    result, _ = _result()
    if missing_prediction:
        result = replace(result, prediction=None)
        message = "requires an InterpolationPrediction"
    else:
        result = replace(result, job=replace(result.job, timestamp=pd.Timestamp("2026-10-25")))
        message = "Missing timezone"
    with pytest.raises(ValueError, match=message):
        result.to_dataset()


def test_coordinator_passes_processing_metadata_to_output(tmp_path):
    result, grid = _result()
    context = SimpleNamespace(
        output_dir=tmp_path, resampler=MeteoResampler(min_coverage=.7), gapfiller=None,
        config={"interpolation": {}},
        dem=SimpleNamespace(path=Path("dem.tif"), fingerprint="dem-identity"),
    )
    coordinator = InterpolationCoordinator(context)
    coordinator._prepare_request = AsyncMock(return_value=PreparedInterpolationRun(grid, [result.job]))
    coordinator._submit_jobs = lambda _: iter([result])
    request = InterpolationRequest(param="tair_2m", start="2026-10-25", end="2026-10-26")
    run = asyncio.run(coordinator.run(request, request_id="request"))
    assert run.status == "completed", run.job_results
    with xr.open_zarr(run.job_results[0].output_path, consolidated=False) as ds:
        metadata = ds.attrs["request_metadata"]
        assert metadata["resampling"]["min_coverage"] == .7
        assert metadata["resampling"]["aggregation"] == "mean"
        assert metadata["resampling"]["source_frequency"] == "h"
        assert not metadata["gapfilling"]["enabled"]
        assert metadata["dem_fingerprint"] == "dem-identity"
    coordinator._validate_dates(request.start, request.end)


def test_submitted_run_streams_outputs_and_counts_all_job_statuses(tmp_path):
    result, grid = _result()
    skipped = replace(result, status="skipped", prediction=None)
    failed = replace(result, status="failed", prediction=None, error="insufficient data")
    context = SimpleNamespace(
        output_dir=tmp_path, resampler=MeteoResampler(), gapfiller=None,
        config={}, dem=SimpleNamespace(path="dem.tif", fingerprint="dem-id"),
    )
    coordinator = InterpolationCoordinator(context)
    coordinator._prepare_request = AsyncMock(return_value=PreparedInterpolationRun(grid, []))

    def results(_):
        yield result
        assert len(list(tmp_path.glob("*/*.zarr"))) == 1
        yield skipped
        yield failed

    coordinator._submit_jobs = results
    request = InterpolationRequest(param="tair_2m", start="2026-10-25", end="2026-10-26")

    async def submit_and_wait():
        submission = await coordinator.submit(request)
        await coordinator._tasks[submission.request_id]
        return coordinator.get_status(submission.request_id)

    state = asyncio.run(submit_and_wait())
    assert state.status == "completed"
    assert (state.n_jobs, state.n_completed, state.n_skipped, state.n_failed, state.n_outputs) == (3, 1, 1, 1, 1)
    assert all(value.utcoffset().total_seconds() == 0 for value in (
        state.submitted_at, state.started_at, state.finished_at,
    ))


def test_coordinator_reports_write_failures_in_summary(tmp_path, monkeypatch):
    result, grid = _result()
    context = SimpleNamespace(
        output_dir=tmp_path, resampler=MeteoResampler(), gapfiller=None,
        config={}, dem=SimpleNamespace(path="dem.tif", fingerprint="dem-id"),
    )
    coordinator = InterpolationCoordinator(context)
    coordinator._prepare_request = AsyncMock(return_value=PreparedInterpolationRun(grid, []))
    coordinator._submit_jobs = lambda _: iter([result])

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("src.coordinator.write_job_result", fail_write)
    request = InterpolationRequest(param="tair_2m", start="2026-10-25", end="2026-10-26")
    run = asyncio.run(coordinator.run(request))
    assert run.status == "failed"
    assert len(run.job_results) == 1
    assert run.job_results[0].error == "disk full"
    assert run.job_results[0].output_path is None


@pytest.mark.skipif(os.name != "nt", reason="Windows directory-lock recovery")
def test_publication_recovers_from_a_temporary_directory_lock(tmp_path, monkeypatch):
    original_replace = Path.replace
    locked_once = False

    def replace_after_lock(path, target):
        nonlocal locked_once
        if not locked_once and path.is_dir() and ".partial_" in path.name:
            locked_once = True
            raise PermissionError("temporary directory lock")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", replace_after_lock)
    result, _ = _result()
    output = write_job_result(result, tmp_path, "request", {"freq": "D", "timezone": "Europe/Rome"})
    assert locked_once
    assert output.exists()
    assert list(output.parent.iterdir()) == [output]
