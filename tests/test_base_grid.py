import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr

from src.array.base_grid import _CACHE_KEY, generate_cache_payload, load_base_grid
from src.array.cache import CacheManager


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
    cache_manager = CacheManager(tmp_path / "cache")

    first = load_base_grid(source_path, var="orog", cache_manager=cache_manager)
    second = load_base_grid(source_path, var="orog", cache_manager=cache_manager)

    assert first.data.rio.crs.to_epsg() == 4326
    assert second.from_cache
    assert second.data.rio.crs.to_epsg() == 4326


def test_base_grid_cache_without_crs_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source_path = tmp_path / "source.zarr"
    _source_grid().to_dataset().to_zarr(source_path, zarr_format=2)
    cache_manager = CacheManager(tmp_path / "cache")
    cache_payload = generate_cache_payload(
        path=source_path,
        target_crs=None,
        target_res=None,
        original_crs=4326,
        x_dim=None,
        y_dim=None,
        aoi=None,
        aoi_buffer_m=None,
        resampling_method="bilinear",
        var="orog",
    )
    cache_path = cache_manager._build_cache_path(_CACHE_KEY, cache_payload)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _source_grid().to_dataset().to_zarr(cache_path, zarr_format=2)

    with pytest.raises(ValueError, match="does not define a CRS"):
        load_base_grid(source_path, var="orog", original_crs=4326, cache_manager=cache_manager)


def test_remote_base_grid_url_is_not_converted_to_windows_path(monkeypatch):
    source_url = "https://example.test/data/chelsa-w5e5_obsclim_orog_30arcsec_global.nc#mode=bytes"
    opened_calls = []

    def fake_open_dataset(path, **kwargs):
        opened_calls.append((path, kwargs))
        return _source_grid(crs=4326).to_dataset()

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    base_grid = load_base_grid(source_url, var="orog", original_crs=4326, engine="netcdf4")

    assert opened_calls == [(source_url, {"engine": "netcdf4"})]
    assert base_grid.path == source_url
