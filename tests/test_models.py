import numpy as np
import pytest
import xarray as xr

from src.interpolate.models import (
    FittedLinearLapseRateModel,
    FittedNonLinearLapseRateModel,
    LinearLapseRateModel,
    NonLinearLapseRateModel,
)
from src.interpolate.models import nonlinear
from src.interpolate.models.nonlinear import NonLinearProfileParams


def _training_data():
    elevations = np.linspace(100., 3000., 12)
    ids = [f"station-{i}" for i in range(len(elevations))]
    X = xr.Dataset({"elevation": ("id", elevations)}, coords={"id": ids})
    y = xr.DataArray(20. - .006 * elevations, dims="id", coords={"id": ids})
    return X, y


def _targets(grid):
    if grid:
        return xr.Dataset(
            {"elevation": (("y", "x"), [[500., 2000., np.nan]])},
            coords={"y": [0.], "x": [0., 1., 2.]},
        )
    return xr.Dataset(
        {"elevation": ("target_id", [500., 2000., np.nan])},
        coords={"target_id": ["a", "b", "c"],
                "x": ("target_id", [0., 1., 2.]),
                "y": ("target_id", [0., 0., 0.])},
    )


@pytest.mark.parametrize("grid", [False, True])
def test_linear_fit_recovers_lapse_rate_and_preserves_target_geometry(grid):
    X, y = _training_data()
    # Station order must not change the association between elevation and value.
    fit = LinearLapseRateModel().fit(X.isel(id=slice(None, None, -1)), y)
    assert fit.intercept == pytest.approx(20.)
    assert fit.coef == pytest.approx((-.006,))
    targets = _targets(grid)
    prediction = fit.predict(targets)
    assert prediction.dims == targets.elevation.dims
    for name in targets.coords:
        xr.testing.assert_identical(prediction.coords[name], targets.coords[name])
    np.testing.assert_allclose(
        prediction.values, (20. - .006 * targets.elevation).values,
    )


@pytest.mark.parametrize("grid", [False, True])
def test_nonlinear_prediction_covers_profile_regions_and_boundaries(grid):
    fit = FittedNonLinearLapseRateModel(NonLinearProfileParams(20., .006, 4., 500., 1500.))
    targets = _targets(grid)
    elevations = [0., 500., 1000., 1500., 2000., np.nan]
    if grid:
        targets = xr.Dataset(
            {"elevation": (("y", "x"), [elevations])},
            coords={"y": [0.], "x": np.arange(6, dtype=float)},
        )
    else:
        targets = xr.Dataset(
            {"elevation": ("target_id", elevations)},
            coords={"target_id": list("abcdef"),
                    "x": ("target_id", np.arange(6, dtype=float)),
                    "y": ("target_id", np.zeros(6))},
        )
    prediction = fit.predict(targets)
    np.testing.assert_allclose(prediction.values.ravel(), [16., 13., 12., 11., 8., np.nan])
    assert prediction.dims == targets.elevation.dims
    for name in targets.coords:
        xr.testing.assert_identical(prediction.coords[name], targets.coords[name])


def test_nonlinear_fit_predicts_synthetic_inversion_with_real_optimizer():
    X, y = _training_data()
    # Construct the observations independently from the production profile.
    heights = X.elevation.values
    inversion = np.where(heights <= 500., 4., np.where(
        heights >= 1500., 0., 2. * (1. + np.cos(np.pi * (heights - 500.) / 1000.)),
    ))
    y.values[:] = 20. - .006 * heights - inversion
    initial = NonLinearProfileParams(19., .005, 3., 450., 1550.)
    fit = NonLinearLapseRateModel(initial_parameters=initial).fit(X, y)
    assert isinstance(fit, FittedNonLinearLapseRateModel)
    np.testing.assert_allclose(fit.predict(X.rename(id="target_id").assign_coords(
        x=("target_id", np.zeros(len(y))), y=("target_id", np.zeros(len(y))),
    )), y, atol=1e-5)


def test_nonlinear_fit_failure_propagates_original_error_by_default(monkeypatch):
    error = RuntimeError("maximum evaluations reached")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(nonlinear, "curve_fit", fail)
    with pytest.raises(RuntimeError) as caught:
        NonLinearLapseRateModel().fit(*_training_data())
    assert caught.value is error


def test_nonlinear_fallback_returns_usable_linear_fit_without_mutating_inputs(monkeypatch, caplog):
    def fail(*args, **kwargs):
        raise RuntimeError("maximum evaluations reached")

    monkeypatch.setattr(nonlinear, "curve_fit", fail)
    X, y = _training_data()
    original_X, original_y = X.copy(deep=True), y.copy(deep=True)
    fit = NonLinearLapseRateModel(allow_linear_fallback=True).fit(X, y)
    assert isinstance(fit, FittedLinearLapseRateModel)
    np.testing.assert_allclose(fit.predict(_targets(False)), [17., 8., np.nan])
    xr.testing.assert_identical(X, original_X)
    xr.testing.assert_identical(y, original_y)
    assert "maximum evaluations reached" in caplog.text


def test_nonlinear_fallback_does_not_hide_invalid_optimizer_configuration():
    with pytest.raises(ValueError, match="method"):
        NonLinearLapseRateModel(
            allow_linear_fallback=True, fit_method="invalid",
        ).fit(*_training_data())


@pytest.mark.parametrize("allow_fallback", [False, True])
def test_nonlinear_requires_five_samples_even_with_fallback_enabled(allow_fallback):
    X, y = _training_data()
    with pytest.raises(ValueError, match="at least five valid samples"):
        NonLinearLapseRateModel(allow_linear_fallback=allow_fallback).fit(
            X.isel(id=slice(4)), y.isel(id=slice(4)),
        )


def test_nonlinear_fallback_is_applied_independently_to_cv_training_folds(monkeypatch):
    import geopandas as gpd
    import pandas as pd

    from src.interpolate import cross_validate
    from src.interpolate.distance import DistanceField
    from src.interpolate.interpolator import InterpolationJob, Interpolator
    from src.interpolate.models import IDWModel

    X, y = _training_data()
    # A curved signal makes each fold's linear fit differ from the full-data fit.
    values = y.values + (X.elevation.values / 1000.) ** 2
    observations = pd.DataFrame({
        "station_id": X.id.values, "elevation": X.elevation.values,
        "x": np.arange(len(y), dtype=float), "y": np.zeros(len(y)),
        "tair_2m": values,
    })
    points = gpd.GeoDataFrame(
        observations[["station_id", "elevation"]],
        geometry=gpd.points_from_xy(observations.x, observations.y), crs=4326,
    ).set_index("station_id")
    job = InterpolationJob(pd.Timestamp("2026-01-01", tz="UTC"), "tair_2m", observations, points)
    positions = observations.x.to_numpy()
    distances = DistanceField("test", xr.DataArray(
        np.abs(positions[:, None] - positions[None, :])[:, None, :],
        dims=("id", "y", "x"),
        coords={"id": X.id.values, "y": [0.], "x": positions},
    ))
    attempted_training_values = []

    def fail(profile, heights, values, *args, **kwargs):
        attempted_training_values.append(values.copy())
        raise RuntimeError("maximum evaluations reached")

    monkeypatch.setattr(nonlinear, "curve_fit", fail)
    options = dict(job=job, param_grid={"lam_value": [0]}, distance_fields=distances, n_splits=2)
    fallback_model = NonLinearLapseRateModel(allow_linear_fallback=True)
    result = cross_validate(Interpolator(fallback_model, IDWModel(neighbours=1)), **options)
    expected = cross_validate(Interpolator(LinearLapseRateModel(), IDWModel(neighbours=1)), **options)
    pd.testing.assert_frame_equal(result.fold_results, expected.fold_results)
    assert result.get_best_parameters() == expected.get_best_parameters()
    assert len(attempted_training_values) == 2
    np.testing.assert_array_equal(attempted_training_values[0], values[6:])
    np.testing.assert_array_equal(attempted_training_values[1], values[:6])

    # Disabling fallback must still abort CV with the original optimizer error.
    with pytest.raises(RuntimeError, match="maximum evaluations reached"):
        cross_validate(Interpolator(NonLinearLapseRateModel(), IDWModel(neighbours=1)), **options)
