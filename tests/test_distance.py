import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from pathlib import Path

from src.array.base_grid import BaseGrid
from src.array.cache import CacheManager
from src.interpolate.distance import PathDistanceCalculator


def _base_grid(data: xr.DataArray) -> BaseGrid:
    if data.rio.crs is None:
        data = data.rio.write_crs(4326)
    return BaseGrid(
        path=Path("memory"),
        data=data,
        aoi=None,
        resampling_method="nearest",
        fingerprint=CacheManager.array_fingerprint(data),
    )


def calculate_path_distance(
    dem,
    x_coords,
    y_coords,
    point_ids,
    lam_values,
    connectivity=4,
    max_visibility_distance=None,
):
    calculator = PathDistanceCalculator(
        connectivity_type=connectivity,
        lam_values=lam_values,
        max_visibility_distance=max_visibility_distance,
    )
    return calculator.calculate_fields(
        dem=_base_grid(dem),
        x_coords=x_coords,
        y_coords=y_coords,
        point_ids=point_ids,
    ).data


def test_connectivity_controls_diagonal_steps():
    dem = xr.DataArray(
        np.zeros((3, 3), dtype=float),
        dims=("y", "x"),
        coords={"y": [2.0, 1.0, 0.0], "x": [0.0, 1.0, 2.0]},
    )

    distances_4 = calculate_path_distance(
        dem,
        x_coords=[0.0],
        y_coords=[1.0],
        point_ids=["station"],
        lam_values=[0],
        connectivity=4,
    )
    distances_8 = calculate_path_distance(
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

    distances = calculate_path_distance(
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

    surface_distances = calculate_path_distance(
        dem,
        x_coords=[0.0],
        y_coords=[0.0],
        point_ids=["station"],
        lam_values=[1],
        connectivity=4,
    )
    visibility_distances = calculate_path_distance(
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

    distances = calculate_path_distance(
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

    distances = calculate_path_distance(
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


def test_calculate_fields_writes_and_reuses_cache(tmp_path):
    dem = xr.DataArray(
        np.zeros((3, 3), dtype=float),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0, 2.0], "x": [0.0, 1.0, 2.0]},
    )
    calculator = PathDistanceCalculator(
        connectivity_type=4,
        lam_values=[0],
        cache_manager=CacheManager(tmp_path),
    )

    dem = _base_grid(dem)
    first = calculator.calculate_fields(dem, [0.0], [0.0], ["station"])
    second = calculator.calculate_fields(dem, [0.0], [0.0], ["station"])

    assert len(list(tmp_path.glob("*.zarr"))) == 1
    xr.testing.assert_equal(first.data, second.data)


def test_distance_cache_key_includes_base_grid_fingerprint(tmp_path):
    cache_manager = CacheManager(tmp_path)
    calculator = PathDistanceCalculator(
        connectivity_type=4,
        lam_values=[0],
        cache_manager=cache_manager,
    )
    first_dem = _base_grid(
        xr.DataArray(
            np.zeros((3, 3), dtype=float),
            dims=("y", "x"),
            coords={"y": [0.0, 1.0, 2.0], "x": [0.0, 1.0, 2.0]},
        )
    )
    second_dem = _base_grid(
        xr.DataArray(
            np.ones((3, 3), dtype=float),
            dims=("y", "x"),
            coords={"y": [0.0, 1.0, 2.0], "x": [0.0, 1.0, 2.0]},
        )
    )

    calculator.calculate_fields(first_dem, [0.0], [0.0], ["station"])
    calculator.calculate_fields(second_dem, [0.0], [0.0], ["station"])

    assert len(list(tmp_path.glob("*.zarr"))) == 2


def test_source_points_outside_dem_raise_clear_error():
    dem = xr.DataArray(
        np.zeros((3, 3), dtype=float),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0, 2.0], "x": [0.0, 1.0, 2.0]},
    )
    calculator = PathDistanceCalculator(connectivity_type=4, lam_values=[0])

    with pytest.raises(ValueError, match="outside the supplied DEM extent"):
        calculator.calculate_fields(_base_grid(dem), [10.0], [0.0], ["station"])
