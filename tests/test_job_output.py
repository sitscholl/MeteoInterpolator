import asyncio
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
from src.output_metadata import aggregation_interval
from src.runtime import CrossValidationConfig
from src.worker import process_interpolation_job
from src.write import write_job_result


def _result(timestamp="2026-10-25", *, residuals=False, lambda_dimension=True):
    observations = pd.DataFrame({
        "station_id": ["a", "b", "c"], "elevation": [0., 1., 2.],
        "x": [0., 1., 2.], "y": [0., 0., 0.], "tair_2m": [1., 3., 4.],
    })
    stations = gpd.GeoDataFrame(
        observations[["station_id", "elevation"]],
        geometry=[Point(x, 0.) for x in observations.x], crs=4326,
    ).set_index("station_id")
    job = InterpolationJob(
        pd.Timestamp(timestamp, tz="Europe/Rome"), "tair_2m", observations, stations,
    )
    grid = xr.DataArray(
        [[0., 1., 2.]], dims=("y", "x"), coords={"y": [0.], "x": [0., 1., 2.]},
        name="elevation", attrs={"units": "m"},
    ).rio.write_crs(4326)
    distances = None
    if residuals:
        data = xr.DataArray(
            np.abs(np.arange(3)[:, None, None] - np.arange(3)[None, None, :]),
            dims=("id", "y", "x"),
            coords={"id": ["a", "b", "c"], "y": [0.], "x": [0., 1., 2.]},
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


@pytest.mark.parametrize("day,hours", [
    ("2026-03-29", 23), ("2026-05-13", 24), ("2026-10-25", 25),
])
def test_daily_output_round_trip_has_local_calendar_bounds(tmp_path, day, hours):
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
        bounds = ds.time_bounds.values[0]
        assert (bounds[1] - bounds[0]) / np.timedelta64(1, "h") == hours
        assert ds.time.attrs["bounds"] == "time_bounds"
        assert ds.time.attrs["timezone"] == "UTC"
        assert ds.attrs["aggregation_metadata"]["duration_seconds"] == hours * 3600
        assert ds.attrs["aggregation_metadata"]["timezone"] == "Europe/Rome"
        assert ds.attrs["request_id"] == "request-1"
        assert ds.attrs["output_schema_version"] == "1.0"
        assert ds.attrs["software_version"] != "unknown"
        assert datetime.fromisoformat(ds.attrs["created_at"]).utcoffset().total_seconds() == 0
        assert ds.attrs["request_metadata"]["dem_path"] == "dem.tif"
        assert ds.tair_2m.attrs["units"] == "degC"
        assert ds.tair_2m.attrs["cell_methods"] == "time: mean"
        model = ds.attrs["model_metadata"]["vertical_model"]
        assert model["type"] == "linear"
        assert model["fitted_parameters"]["coef"] == pytest.approx([1.5])
        assert ds.attrs["job_metadata"]["training_stations"] == ["a", "b", "c"]
        assert len(ds.attrs["job_metadata"]["observation_fingerprint"]) == 64
        assert not ds.attrs["cv_metadata"]["enabled"]
    assert path.name == f"tair_2m_D_{start:%Y%m%dT%H%M%SZ}.zarr"


def test_repeated_hour_intervals_are_distinct_elapsed_hours():
    intervals = [aggregation_interval(value, "h", "Europe/Rome") for value in (
        "2026-10-25T02:00:00+02:00", "2026-10-25T02:00:00+01:00",
    )]
    assert intervals[0][1] == intervals[1][0]
    assert all(end - start == pd.Timedelta(hours=1) for start, end in intervals)


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
        assert ds.tair_2m_residuals.attrs["units"] == "degC"


def test_observation_fingerprint_detects_revised_values():
    result, _ = _result()
    original = result.to_dataset().attrs["job_metadata"]["observation_fingerprint"]
    shuffled = replace(result.job, observations=result.job.observations.iloc[::-1])
    assert replace(result, job=shuffled).to_dataset().attrs["job_metadata"]["observation_fingerprint"] == original
    revised = result.job.observations.copy()
    revised.loc[0, "tair_2m"] += 1
    changed = replace(result, job=replace(result.job, observations=revised))
    assert changed.to_dataset().attrs["job_metadata"]["observation_fingerprint"] != original


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
