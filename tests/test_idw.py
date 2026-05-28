import numpy as np
import pytest
import xarray as xr

from src.interpolate.distance import DistanceField
from src.interpolate.idw import InverseDistanceWeighting


def test_idw_uses_frei_weighting_formula_for_closest_stations():
    distances = xr.DataArray(
        np.array([[[1.0]], [[2.0]], [[10.0]]]),
        dims=("id", "y", "x"),
        coords={"id": ["a", "b", "c"], "y": [0.0], "x": [0.0]},
    )
    residuals = xr.DataArray(
        [10.0, 20.0, 100.0],
        dims=("id",),
        coords={"id": ["a", "b", "c"]},
    )

    result = InverseDistanceWeighting(neighbours=2).interpolate(
        residuals,
        DistanceField("test_distance", distances),
    )

    expected = (10.0 / 1.0**2 + 20.0 / 2.0**2) / (1.0 / 1.0**2 + 1.0 / 2.0**2)
    assert float(result.sel(y=0.0, x=0.0)) == pytest.approx(expected)


def test_idw_returns_exact_residual_at_zero_distance_cell():
    distances = xr.DataArray(
        np.array([[[0.0]], [[1.0]]]),
        dims=("id", "y", "x"),
        coords={"id": ["a", "b"], "y": [0.0], "x": [0.0]},
    )
    residuals = xr.DataArray(
        [10.0, 20.0],
        dims=("id",),
        coords={"id": ["a", "b"]},
    )

    result = InverseDistanceWeighting(neighbours=2).interpolate(
        residuals,
        DistanceField("test_distance", distances),
    )

    assert float(result.sel(y=0.0, x=0.0)) == pytest.approx(10.0)


def test_idw_preserves_extra_distance_dimensions():
    distances = xr.DataArray(
        np.array(
            [
                [[[1.0]], [[2.0]]],
                [[[2.0]], [[1.0]]],
            ]
        ),
        dims=("id", "lam_value", "y", "x"),
        coords={"id": ["a", "b"], "lam_value": [0.0, 1.0], "y": [0.0], "x": [0.0]},
    )
    residuals = xr.DataArray(
        [10.0, 20.0],
        dims=("id",),
        coords={"id": ["a", "b"]},
    )

    result = InverseDistanceWeighting(neighbours=1).interpolate(
        residuals,
        DistanceField("test_distance", distances),
    )

    assert result.dims == ("lam_value", "y", "x")
    assert float(result.sel(lam_value=0.0, y=0.0, x=0.0)) == pytest.approx(10.0)
    assert float(result.sel(lam_value=1.0, y=0.0, x=0.0)) == pytest.approx(20.0)
