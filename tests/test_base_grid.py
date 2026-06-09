import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr

from src.array.base_grid import BaseGrid


def _source_grid(crs=None):
    data = xr.DataArray(
        np.ones((2, 2), dtype=float),
        dims=("y", "x"),
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
        name="orog",
    )
    if crs is not None:
        data = data.rio.write_crs(crs)
    return data


def test_base_grid_cache_preserves_crs_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source_path = tmp_path / "source.zarr"
    _source_grid(crs=4326).to_dataset().to_zarr(source_path, zarr_format=2)

    first = BaseGrid(source_path, var="orog", cache=True)
    second = BaseGrid(source_path, var="orog", cache=True)

    assert first.crs == 4326
    assert second.crs == 4326
    assert second.data.rio.crs.to_epsg() == 4326
    assert second.data.attrs["crs"] == "EPSG:4326"
    assert second.data.attrs["crs_epsg"] == "EPSG:4326"


def test_base_grid_cache_without_crs_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source_path = tmp_path / "source.zarr"
    _source_grid().to_dataset().to_zarr(source_path, zarr_format=2)
    cache_path = BaseGrid._cache_path(
        None,
        path=source_path,
        target_crs=None,
        target_res=None,
        target_x_dim="x",
        target_y_dim="y",
        crs=4326,
        x_dim=None,
        y_dim=None,
        aoi=None,
        aoi_buffer_m=None,
        resampling_method="bilinear",
        var="orog",
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _source_grid().to_dataset().to_zarr(cache_path, zarr_format=2)

    with pytest.raises(ValueError, match="does not define a CRS"):
        BaseGrid(source_path, var="orog", crs=4326, cache=True)
