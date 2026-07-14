import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr

from src.domain.dem import load_dem


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


def test_load_dem_preserves_crs_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source_path = tmp_path / "source.zarr"
    _source_grid(crs=4326).to_dataset().to_zarr(source_path, zarr_format=2)

    dem = load_dem(source_path, var="orog")

    assert dem.crs.to_epsg() == 4326
    assert dem.bounds == dem.data.rio.bounds()


def test_load_dem_without_crs_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source_path = tmp_path / "source.zarr"
    _source_grid().to_dataset().to_zarr(source_path, zarr_format=2)

    with pytest.raises(ValueError, match="Dataset CRS could not be loaded"):
        load_dem(source_path, var="orog")


def test_load_dem_writes_configured_crs_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source_path = tmp_path / "source.zarr"
    _source_grid().to_dataset().to_zarr(source_path, zarr_format=2)

    dem = load_dem(source_path, var="orog", crs=4326)

    assert dem.crs.to_epsg() == 4326


def test_remote_dem_url_is_not_converted_to_windows_path(monkeypatch):
    source_url = "https://example.test/data/dem.nc#mode=bytes"
    opened_calls = []

    def fake_open_dataset(path, **kwargs):
        opened_calls.append((path, kwargs))
        return _source_grid(crs=4326).to_dataset()

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    dem = load_dem(source_url, var="orog", engine="netcdf4")

    assert opened_calls == [(source_url, {"engine": "netcdf4"})]
    assert dem.path == source_url
