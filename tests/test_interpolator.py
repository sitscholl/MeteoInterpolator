from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import rioxarray  # noqa: F401

from src.array.cache import CacheManager
from src.array.dem import DEM
from src.interpolate.distance import DistanceField
from src.interpolate.idw import InverseDistanceWeighting
from src.interpolate.interpolator import InterpolationJob, Interpolator
from src.interpolate.vertical import LinearVerticalFit, LinearVerticalModel


def _dem(data: xr.DataArray) -> DEM:
    return DEM(
        path=Path("memory"),
        data=data,
        fingerprint=CacheManager.array_fingerprint(data),
    )


def test_linear_fit_returns_immutable_fitted_model():
    model = LinearVerticalModel()

    fit = model.fit(np.array([[0.0], [1.0], [2.0]]), np.array([1.0, 3.0, 5.0]))

    assert isinstance(fit, LinearVerticalFit)
    assert fit.predict(np.array([3.0])).tolist() == pytest.approx([7.0])
    with pytest.raises(FrozenInstanceError):
        fit.intercept = 0.0


def test_residual_array_uses_id_dimension_with_station_coordinates():
    residuals = Interpolator._residual_array(
        residuals=[1.0, -1.0],
        ids=["a", "b"],
        x_coords=[10.0, 20.0],
        y_coords=[30.0, 40.0],
    )

    assert residuals.dims == ("id",)
    assert residuals.id.values.tolist() == ["a", "b"]
    assert residuals.x.dims == ("id",)
    assert residuals.y.dims == ("id",)
    assert residuals.x.values.tolist() == [10.0, 20.0]
    assert residuals.y.values.tolist() == [30.0, 40.0]


def test_interpolator_requires_dem_with_matching_crs():
    target_grid = xr.DataArray(
        [[0.0]],
        dims=("y", "x"),
        coords={"y": [0.0], "x": [0.0]},
    )
    observations = pd.DataFrame(
        {
            "station_id": ["a"],
            "elevation": [0.0],
            "x": [0.0],
            "y": [0.0],
            "tair_2m": [1.0],
        }
    )

    with pytest.raises(TypeError, match="dem must be a DEM"):
        Interpolator(
            dem=target_grid,
            vertical_model=LinearVerticalModel(),
        )

    job = InterpolationJob(
        timestamp=pd.Timestamp("2026-05-13"),
        parameter="tair_2m",
        observations=observations,
        crs=4326,
    )
    interpolator = Interpolator(
        dem=_dem(target_grid.rio.write_crs(3857)),
        vertical_model=LinearVerticalModel(),
    )
    with pytest.raises(ValueError, match="does not match"):
        interpolator.fit(job)


def test_interpolator_returns_result_and_adds_idw_residuals():
    target_grid = xr.DataArray(
        [[0.0, 1.0, 2.0]],
        dims=("y", "x"),
        coords={"y": [0.0], "x": [0.0, 1.0, 2.0]},
        name="elevation",
    ).rio.write_crs(4326)
    observations = pd.DataFrame(
        {
            "station_id": ["a", "b", "c"],
            "elevation": [0.0, 1.0, 2.0],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "tair_2m": [0.0, 3.0, 2.0],
        }
    )
    distances = xr.DataArray(
        np.array(
            [
                [[0.0, 1.0, 2.0]],
                [[1.0, 0.0, 1.0]],
                [[2.0, 1.0, 0.0]],
            ]
        ),
        dims=("id", "y", "x"),
        coords={"id": ["a", "b", "c"], "y": [0.0], "x": [0.0, 1.0, 2.0]},
    )
    job = InterpolationJob(
        timestamp=pd.Timestamp("2026-05-13"),
        parameter="tair_2m",
        observations=observations,
        crs=4326,
    )
    interpolator = Interpolator(
        dem=_dem(target_grid),
        vertical_model=LinearVerticalModel(),
        residual_model=InverseDistanceWeighting(neighbours=1),
        min_sample_size=3,
    )

    result = interpolator.fit(job).predict(distance_fields=DistanceField("test_distance", distances))

    assert interpolator.timestamp_ == pd.Timestamp("2026-05-13")
    assert interpolator.parameter_ == "tair_2m"
    assert result.dims == ("y", "x")
    xr.testing.assert_equal(result.x, target_grid.x)
    xr.testing.assert_equal(result.y, target_grid.y)
    assert float(result.sel(y=0.0, x=0.0)) == pytest.approx(0.0)
    assert float(result.sel(y=0.0, x=1.0)) == pytest.approx(3.0)
    assert float(result.sel(y=0.0, x=2.0)) == pytest.approx(2.0)


def test_interpolator_predicts_points_with_idw_residuals():
    target_grid = xr.DataArray(
        [[0.0, 1.0, 2.0]],
        dims=("y", "x"),
        coords={"y": [0.0], "x": [0.0, 1.0, 2.0]},
        name="elevation",
    ).rio.write_crs(4326)
    observations = pd.DataFrame(
        {
            "station_id": ["a", "b", "c"],
            "elevation": [0.0, 1.0, 2.0],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "tair_2m": [0.0, 3.0, 2.0],
        }
    )
    distances = DistanceField(
        "test_distance",
        xr.DataArray(
            np.array(
                [
                    [[0.0, 1.0, 2.0]],
                    [[1.0, 0.0, 1.0]],
                    [[2.0, 1.0, 0.0]],
                ]
            ),
            dims=("id", "y", "x"),
            coords={"id": ["a", "b", "c"], "y": [0.0], "x": [0.0, 1.0, 2.0]},
        ),
    )
    job = InterpolationJob(
        timestamp=pd.Timestamp("2026-05-13"),
        parameter="tair_2m",
        observations=observations,
        crs=4326,
    )
    interpolator = Interpolator(
        dem=_dem(target_grid),
        vertical_model=LinearVerticalModel(),
        residual_model=InverseDistanceWeighting(neighbours=1),
    )

    result = interpolator.fit(job).predict(observations, distance_fields=distances)

    assert isinstance(result, pd.Series)
    assert result.index.tolist() == ["a", "b", "c"]
    assert result.tolist() == pytest.approx([0.0, 3.0, 2.0])
