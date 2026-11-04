from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
import geopandas as gpd
import xarray as xr
import rioxarray  # noqa: F401
from shapely.geometry import Point

from src.interpolate.distance import DistanceField
from src.interpolate.models import IDWModel
from src.interpolate.interpolator import InterpolationJob, InterpolationPrediction, Interpolator
from src.interpolate.models import FittedLinearLapseRateModel, LinearLapseRateModel
from src.interpolate.models import NonLinearLapseRateModel


def _training_points(observations: pd.DataFrame, crs=4326) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        observations[["station_id", "elevation"]].copy(),
        geometry=[Point(x, y) for x, y in zip(observations["x"], observations["y"])],
        crs=crs,
    ).set_index("station_id")


def test_linear_fit_returns_immutable_fitted_model():
    model = LinearLapseRateModel()

    X = xr.Dataset({"elevation": ("id", [0., 1., 2.])}, coords={"id": ["a", "b", "c"]})
    y = xr.DataArray([1., 3., 5.], dims="id", coords={"id": X.id})
    fit = model.fit(X, y)
    targets = xr.Dataset(
        {"elevation": ("target_id", [3.])},
        coords={"target_id": ["p"], "x": ("target_id", [0.]), "y": ("target_id", [0.])},
    )

    assert isinstance(fit, FittedLinearLapseRateModel)
    assert fit.predict(targets).values.tolist() == pytest.approx([7.0])
    with pytest.raises(FrozenInstanceError):
        fit.intercept = 0.0


def test_interpolator_requires_target_with_matching_crs():
    target_grid = xr.DataArray(
        [[0.0]],
        dims=("y", "x"),
        coords={"y": [0.0], "x": [0.0]},
    ).rio.write_crs(3857)
    observations = pd.DataFrame(
        {
            "station_id": ["a"],
            "elevation": [0.0],
            "x": [0.0],
            "y": [0.0],
            "tair_2m": [1.0],
        }
    )

    job = InterpolationJob(
        timestamp=pd.Timestamp("2026-05-13"),
        parameter="tair_2m",
        observations=observations,
        training_points=_training_points(observations, crs=4326),
    )
    interpolator = Interpolator(primary_model=LinearLapseRateModel(), min_sample_size=1)

    with pytest.raises(ValueError, match="does not match"):
        interpolator.run(job, target_grid.to_dataset(name="elevation"))


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
        training_points=_training_points(observations),
    )
    interpolator = Interpolator(
        primary_model=LinearLapseRateModel(),
        residual_model=IDWModel(neighbours=1),
        min_sample_size=3,
    )

    result = interpolator.run(job, target_grid.to_dataset(), distance_fields=DistanceField("test_distance", distances))

    assert result.residual_summary()["n"] == 3
    assert result.primary_residuals.dims == ("id",)
    assert result.primary_residuals.x.values.tolist() == [0., 1., 2.]
    assert result.primary_residuals.y.values.tolist() == [0., 0., 0.]
    assert isinstance(result, InterpolationPrediction)
    assert result.used_residuals
    assert result.prediction.dims == ("y", "x")
    assert result.primary_prediction.dims == ("y", "x")
    assert result.residual_prediction.dims == ("y", "x")
    xr.testing.assert_equal(result.prediction.x, target_grid.x)
    xr.testing.assert_equal(result.prediction.y, target_grid.y)
    assert float(result.prediction.sel(y=0.0, x=0.0)) == pytest.approx(0.0)
    assert float(result.prediction.sel(y=0.0, x=1.0)) == pytest.approx(3.0)
    assert float(result.prediction.sel(y=0.0, x=2.0)) == pytest.approx(2.0)


def test_interpolator_rejects_dataframe_prediction_targets():
    interpolator = Interpolator(primary_model=LinearLapseRateModel())
    observations = pd.DataFrame({
        "station_id": ["a"], "elevation": [0.], "x": [0.], "y": [0.], "tair_2m": [1.],
    })
    job = InterpolationJob(pd.Timestamp("2026-05-13"), "tair_2m", observations, _training_points(observations))
    with pytest.raises(TypeError, match="xarray Dataset"):
        interpolator.run(job, pd.DataFrame({"elevation": [0.]}))


def _pipeline_fixture():
    observations = pd.DataFrame({
        "station_id": ["a", "b", "c"], "elevation": [0., 1., 2.],
        "x": [0., 1., 2.], "y": [0., 0., 0.], "tair_2m": [1., 4., 3.],
    })
    job = InterpolationJob(
        pd.Timestamp("2026-05-13", tz="UTC"), "tair_2m", observations, _training_points(observations),
    )
    grid = xr.Dataset(
        {"elevation": (("y", "x"), [[0., 1., 2.], [0., 1., 2.]])},
        coords={"y": [0., 1.], "x": [0., 1., 2.]},
    ).rio.write_crs(4326)
    distances = xr.DataArray(
        np.hypot(np.arange(3)[:, None, None] - np.arange(3)[None, None, :],
                 np.arange(2)[None, :, None]),
        dims=("id", "y", "x"), coords={"id": ["a", "b", "c"], **grid.coords},
    )
    return job, grid, distances


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("point_targets", [False, True])
@pytest.mark.parametrize("correction", [False, True])
def test_idw_primary_uses_separate_station_and_target_distances(wrapped, point_targets, correction):
    job, grid, distances = _pipeline_fixture()
    targets = grid
    if point_targets:
        targets = xr.Dataset(
            {"elevation": ("target_id", [2., 0.])},
            coords={"target_id": ["q", "p"], "x": ("target_id", [1.9, .1]),
                    "y": ("target_id", [.1, .1])},
        ).rio.write_crs(4326)
    fields = DistanceField("test_distance", distances) if wrapped else distances
    result = Interpolator(
        IDWModel(neighbours=1), LinearLapseRateModel() if correction else None,
    ).run(job, targets, fields)
    np.testing.assert_allclose(result.primary_residuals.values, 0.)
    assert result.used_residuals == correction
    if point_targets:
        np.testing.assert_allclose(result.prediction.values, [3., 1.])
        xr.testing.assert_equal(result.prediction.x, targets.x)
        assert result.prediction.target_id.values.tolist() == ["q", "p"]
    else:
        np.testing.assert_allclose(result.prediction.values, [[1., 4., 3.], [1., 4., 3.]])


@pytest.mark.parametrize("correction", [False, True])
def test_lapse_rate_pipeline_needs_no_distances(correction):
    job, grid, _ = _pipeline_fixture()
    result = Interpolator(
        LinearLapseRateModel(), LinearLapseRateModel() if correction else None,
    ).run(job, grid)
    np.testing.assert_allclose(result.prediction.values, [[5/3, 8/3, 11/3]] * 2)
    assert result.used_residuals == correction
    assert result.residual_summary()["n"] == 3


def test_singleton_lambda_is_selected_and_recorded():
    job, grid, distances = _pipeline_fixture()
    result = Interpolator(IDWModel()).run(job, grid, distances.expand_dims(lam_value=[25.]))
    assert result.lam_value == 25.
    assert result.prediction.dims == ("y", "x")


def test_multiple_lambdas_require_selection_and_use_requested_value():
    job, grid, distances = _pipeline_fixture()
    fields = distances.expand_dims(lam_value=[0., 25.])
    interpolator = Interpolator(IDWModel())
    with pytest.raises(ValueError, match="multiple lambda"):
        interpolator.run(job, grid, fields)
    result = interpolator.run(job, grid, fields, lam_value=25.)
    assert result.lam_value == 25.


def test_missing_required_distances_raise():
    job, grid, _ = _pipeline_fixture()
    with pytest.raises(ValueError, match="requires distance fields"):
        Interpolator(IDWModel()).run(job, grid)


def test_nonlinear_pipeline_uses_xarray_training_and_preserves_masked_targets():
    elevations = np.linspace(200., 3000., 15)
    observations = pd.DataFrame({
        "station_id": [str(i) for i in range(15)], "elevation": elevations,
        "x": np.arange(15, dtype=float), "y": np.zeros(15),
        "tair_2m": 20. - .006 * elevations,
    })
    job = InterpolationJob(
        pd.Timestamp("2026-05-13", tz="UTC"), "tair_2m", observations, _training_points(observations),
    )
    targets = xr.Dataset(
        {"elevation": (("y", "x"), [[500., 2000., np.nan]])},
        coords={"y": [0.], "x": [0., 1., 2.]},
    ).rio.write_crs(4326)
    result = Interpolator(NonLinearLapseRateModel()).run(job, targets)
    np.testing.assert_allclose(result.prediction.values, [[17., 8., np.nan]], atol=1e-8)
