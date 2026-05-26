import numpy as np
import pytest
import xarray as xr

from src.interpolate.distance import calculate_non_euclidean_distance


def test_connectivity_controls_diagonal_steps():
    dem = xr.DataArray(
        np.zeros((3, 3), dtype=float),
        dims=("y", "x"),
        coords={"y": [2.0, 1.0, 0.0], "x": [0.0, 1.0, 2.0]},
    )

    distances_4 = calculate_non_euclidean_distance(
        dem,
        x_coords=[0.0],
        y_coords=[1.0],
        point_ids=["station"],
        lam_values=[0],
        connectivity=4,
    )
    distances_8 = calculate_non_euclidean_distance(
        dem,
        x_coords=[0.0],
        y_coords=[1.0],
        point_ids=["station"],
        lam_values=[0],
        connectivity=8,
    )

    target = dict(lam_value=0, id="station", y=2.0, x=1.0)
    assert float(distances_4.sel(target)) == pytest.approx(2.0)
    assert float(distances_8.sel(target)) == pytest.approx(np.sqrt(2.0))


def test_lambda_penalizes_elevation_changes_between_neighbors():
    dem = xr.DataArray(
        np.array([[0.0, 100.0, 0.0]]),
        dims=("y", "x"),
        coords={"y": [0.0], "x": [0.0, 1.0, 2.0]},
    )

    distances = calculate_non_euclidean_distance(
        dem,
        x_coords=[0.0],
        y_coords=[0.0],
        point_ids=["station"],
        lam_values=[0, 1],
        connectivity=4,
    )

    target = dict(id="station", y=0.0, x=2.0, lam_value=0)
    assert float(distances.sel(target)) == pytest.approx(2.0)
    target["lam_value"] = 1
    assert float(distances.sel(target)) == pytest.approx(2 * np.sqrt(1.0 + 100.0**2))


def test_visibility_edges_allow_paths_above_interjacent_depressions():
    dem = xr.DataArray(
        np.array([[100.0, 0.0, 100.0]]),
        dims=("y", "x"),
        coords={"y": [0.0], "x": [0.0, 1.0, 2.0]},
    )

    surface_distances = calculate_non_euclidean_distance(
        dem,
        x_coords=[0.0],
        y_coords=[0.0],
        point_ids=["station"],
        lam_values=[1],
        connectivity=4,
    )
    visibility_distances = calculate_non_euclidean_distance(
        dem,
        x_coords=[0.0],
        y_coords=[0.0],
        point_ids=["station"],
        lam_values=[1],
        connectivity=4,
        max_visibility_distance=2.0,
    )

    target = dict(lam_value=1, id="station", y=0.0, x=2.0)
    assert float(surface_distances.sel(target)) == pytest.approx(2 * np.sqrt(1.0 + 100.0**2))
    assert float(visibility_distances.sel(target)) == pytest.approx(2.0)


def test_visibility_edges_are_limited_by_max_distance():
    dem = xr.DataArray(
        np.array([[100.0, 0.0, 100.0]]),
        dims=("y", "x"),
        coords={"y": [0.0], "x": [0.0, 1.0, 2.0]},
    )

    distances = calculate_non_euclidean_distance(
        dem,
        x_coords=[0.0],
        y_coords=[0.0],
        point_ids=["station"],
        lam_values=[1],
        connectivity=4,
        max_visibility_distance=1.9,
    )

    target = dict(lam_value=1, id="station", y=0.0, x=2.0)
    assert float(distances.sel(target)) == pytest.approx(2 * np.sqrt(1.0 + 100.0**2))


def test_visibility_equality_counts_as_blocked():
    dem = xr.DataArray(
        np.zeros((3, 3), dtype=float),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0, 2.0], "x": [0.0, 1.0, 2.0]},
    )

    distances = calculate_non_euclidean_distance(
        dem,
        x_coords=[0.0],
        y_coords=[0.0],
        point_ids=["station"],
        lam_values=[0],
        connectivity=4,
        max_visibility_distance=3.0,
    )

    target = dict(lam_value=0, id="station", y=2.0, x=2.0)
    assert float(distances.sel(target)) == pytest.approx(4.0)
