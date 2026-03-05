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
def nonuniform_chunks():
    return {
        "time": (1, 1, 1, 1, 1),
        "y": (2, 2),
        "x": (2, 2, 1),
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


def _make_writer(store_path, variables, coords, **kwargs):
    start = coords["time"][0].strftime("%Y-%m-%d")
    end = coords["time"][-1].strftime("%Y-%m-%d")
    if "dtype" not in kwargs:
        kwargs["dtype"] = "float32"
    return GridWriter(
        path=store_path,
        start=start,
        end=end,
        freq="D",
        variables=list(variables),
        **kwargs,
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

def test_write_preserves_nonuniform_chunks(tmp_path, sample_coords, nonuniform_chunks, make_dataset):
    store_path = tmp_path / "grid.zarr"
    ds = make_dataset(coords=sample_coords, chunks=nonuniform_chunks)
    writer = _make_writer(store_path, ds.data_vars.keys(), sample_coords)

    writer.write(ds)

    stored = xr.open_zarr(store_path)
    assert stored.chunks == ds.chunks


def test_write_allows_middle_time_insert(tmp_path, sample_coords, make_dataset):
    store_path = tmp_path / "grid.zarr"
    ds_full = make_dataset(coords=sample_coords, seed=1)
    writer = _make_writer(store_path, ds_full.data_vars.keys(), sample_coords)

    writer.write(ds_full)

    mid_time = sample_coords["time"][2]
    mid_coords = {**sample_coords, "time": pd.DatetimeIndex([mid_time])}
    ds_mid = make_dataset(coords=mid_coords, seed=2)

    writer.write(ds_mid)

    stored = xr.open_zarr(store_path).compute()
    stored_mid = stored.sel(time=[mid_time])[list(ds_mid.data_vars)]
    ds_mid_aligned = ds_mid.drop_vars('spatial_ref')
    xr.testing.assert_allclose(
        stored_mid,
        ds_mid_aligned,
    )


def test_write_raises_on_chunk_mismatch(tmp_path, sample_coords, make_dataset):
    store_path = tmp_path / "grid.zarr"
    ds1 = make_dataset(coords=sample_coords, chunks={"time": 1, "y": 2, "x": 2}, seed=1)
    ds2 = make_dataset(coords=sample_coords, chunks={"time": 1, "y": 4, "x": 2}, seed=2)
    writer = _make_writer(store_path, ds1.data_vars.keys(), sample_coords)

    writer.write(ds1)
    with pytest.raises(ValueError, match="chunks"):
        writer.write(ds2)


def test_write_raises_on_coord_mismatch(tmp_path, sample_coords, make_dataset):
    store_path = tmp_path / "grid.zarr"
    ds1 = make_dataset(coords=sample_coords, seed=1)
    writer = _make_writer(store_path, ds1.data_vars.keys(), sample_coords, align=False)
    writer.write(ds1)

    shifted = {**sample_coords, "x": sample_coords["x"] + 0.5}
    ds2 = make_dataset(coords=shifted, seed=2)

    with pytest.raises(ValueError, match="coordinate values"):
        writer.write(ds2)


def test_write_raises_on_shape_mismatch(tmp_path, sample_coords, make_dataset):
    store_path = tmp_path / "grid.zarr"
    ds1 = make_dataset(coords=sample_coords, seed=1)
    writer = _make_writer(store_path, ds1.data_vars.keys(), sample_coords)
    writer.write(ds1)

    larger = {**sample_coords, "x": np.arange(0, 6)}
    ds2 = make_dataset(coords=larger, seed=2)

    with pytest.raises(ValueError, match="shape does not match"):
        writer.write(ds2)


def test_roundtrip_with_encoding(tmp_path, sample_coords, make_dataset):
    store_path = tmp_path / "grid.zarr"
    ds = make_dataset(coords=sample_coords, seed=3)
    writer = _make_writer(
        store_path,
        ds.data_vars.keys(),
        sample_coords,
        dtype="int16",
        scale_factor=0.1,
        fill_value=-9999,
    )

    writer.write(ds)

    stored = xr.open_zarr(store_path)
    xr.testing.assert_allclose(
        stored[list(ds.data_vars)].compute(),
        ds.compute().drop_vars('spatial_ref'),
        atol=0.05,
    )


def test_incremental_time_loop_writes(tmp_path, sample_coords, make_dataset):
    store_path = tmp_path / "grid.zarr"
    full_time = pd.date_range("2025-01-01", "2025-01-10", freq="D")
    coords_full = {**sample_coords, "time": full_time}
    writer = _make_writer(store_path, ("tas", "pr"), coords_full)

    pieces = []
    for idx, t in enumerate(full_time):
        coords_one = {**sample_coords, "time": pd.DatetimeIndex([t])}
        ds_one = make_dataset(coords=coords_one, seed=idx)
        pieces.append(ds_one)
        writer.write(ds_one)

    stored = xr.open_zarr(store_path)
    assert stored.sizes["time"] == len(full_time)
    expected = xr.concat(pieces, dim="time")
    xr.testing.assert_allclose(
        stored[["tas", "pr"]],
        expected.drop_vars('spatial_ref'),
    )


def test_initialize_spec_from_first_dataset(tmp_path, sample_coords, make_dataset):
    store_path = tmp_path / "grid.zarr"
    ds = make_dataset(coords=sample_coords, seed=7)
    start = sample_coords["time"][0].strftime("%Y-%m-%d")
    end = sample_coords["time"][-1].strftime("%Y-%m-%d")
    writer = GridWriter(
        path=store_path,
        start=start,
        end=end,
        freq="D",
        variables=list(ds.data_vars.keys()),
    )

    writer.write(ds)

    stored = xr.open_zarr(store_path)
    xr.testing.assert_allclose(
        stored[list(ds.data_vars)],
        ds.drop_vars('spatial_ref'),
    )
    assert stored.sizes == ds.sizes


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
