import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from pyproj import CRS

from src.array.writer import GridWriter


@pytest.fixture
def sample_coords():
    return {
        "x": np.arange(0, 5),
        "y": np.arange(0, 4),
        "time": pd.date_range("2025-01-01", "2025-01-05", freq="D"),
    }

@pytest.fixture
def sample_chunks():
    return {
        "x": 2,
        "y": 2,
        "time": 1,
    }

@pytest.fixture
def make_dataset():
    def _make(variables=("tas", "pr"), coords=None, chunks=None, seed=0):
        if coords is None:
            raise ValueError("coords must be provided.")
        if isinstance(variables, str):
            variables = [variables]
        if not all(k in coords for k in ("x", "y", "time")):
            raise ValueError(
                f"Coords must contain keys x, y, and time. Got {list(coords.keys())}"
            )

        rng = np.random.default_rng(seed)
        ds = xr.Dataset()
        shape = (len(coords["time"]), len(coords["y"]), len(coords["x"]))
        for idx, var in enumerate(variables):
            data = rng.standard_normal(np.prod(shape)).reshape(shape) + idx
            da = xr.DataArray(
                data=data,
                coords=coords,
                dims=("time", "y", "x"),
                name=var,
            )
            if chunks is not None:
                da = da.chunk(chunks)
            ds[var] = da
        return ds.rio.write_crs("EPSG:4326")

    return _make


def _make_writer(store_path, variables, coords):
    start = coords["time"][0].strftime("%Y-%m-%d")
    end = coords["time"][-1].strftime("%Y-%m-%d")
    return GridWriter(
        path=store_path,
        start=start,
        end=end,
        freq="D",
        variables=list(variables),
        dtype="float32",
    )


def test_write_creates_store_and_persists_data(tmp_path, sample_coords, sample_chunks, make_dataset):
    store_path = tmp_path / "grid.zarr"
    ds = make_dataset(coords=sample_coords, chunks=sample_chunks)
    writer = _make_writer(store_path, ds.data_vars.keys(), sample_coords)

    writer.write(ds)

    assert store_path.exists()
    stored = xr.open_zarr(store_path)
    assert stored.chunks == ds.chunks
    xr.testing.assert_allclose(stored[list(ds.data_vars)].compute(), ds.compute().drop_vars('spatial_ref'))
    assert stored.sizes == ds.sizes
    assert CRS.from_user_input(ds.rio.crs).to_epsg() == stored.attrs.get('crs')


def test_write_overwrites_existing_store(tmp_path, sample_coords, make_dataset):
    store_path = tmp_path / "grid.zarr"
    ds1 = make_dataset(coords=sample_coords, seed=1)
    ds2 = make_dataset(coords=sample_coords, seed=2)
    writer = _make_writer(store_path, ds1.data_vars.keys(), sample_coords)

    writer.write(ds1)
    writer.write(ds2)

    stored = xr.open_zarr(store_path)
    xr.testing.assert_allclose(stored[list(ds2.data_vars)], ds2.drop_vars('spatial_ref'))
    assert CRS.from_user_input(ds2.rio.crs).to_epsg() == stored.attrs.get('crs')
