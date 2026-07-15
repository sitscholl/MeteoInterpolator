import pandas as pd
import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
import rioxarray  # noqa: F401
from pathlib import Path
from shapely.geometry import Point

from src.array.cache import CacheManager
from src.domain.dem import DEM
from src.domain.meteo_data import MeteoData
from src.array.writer import GridWriter
from src.array.writer.tiff_writer import TiffWriter
from src.utils import get_date_format_from_freq, localize_datetime_string


def _dem(data: xr.DataArray) -> DEM:
    return DEM(
        path=Path("memory"),
        data=data,
        fingerprint=CacheManager.array_fingerprint(data),
    )


def test_localize_datetime_string_uses_configured_timezone():
    ts = localize_datetime_string("2026-05-13", "Europe/Rome")

    assert ts.tzinfo is not None
    assert str(ts.tzinfo) == "Europe/Rome"


def test_prepare_grid_for_output_normalizes_suffix_and_renames_single_var_dataset():
    data = xr.DataArray(
        [[1.0]],
        dims=("y", "x"),
        coords={"y": [0.0], "x": [0.0]},
        name="tair_2m",
    ).to_dataset()

    result = GridWriter.prepare_grid_for_output(
        data,
        param="tair_2m",
        interp_date=pd.Timestamp("2026-05-13", tz="Europe/Rome"),
        suffix="__vertical__",
    )

    assert list(result.data_vars) == ["tair_2m_vertical"]
    assert result.sizes["time"] == 1


def test_get_date_format_from_freq_handles_multiplier_and_filename_safe():
    assert get_date_format_from_freq("1D") == "%Y-%m-%d"
    assert get_date_format_from_freq("h") == "%Y-%m-%d %H:%M"
    assert get_date_format_from_freq("15min", filename_safe=True) == "%Y-%m-%d_%H-%M"


def test_tiff_writer_filename_pattern_and_freq_date_format(tmp_path, monkeypatch):
    data = GridWriter.prepare_grid_for_output(
        xr.DataArray(
            [[1.0]],
            dims=("y", "x"),
            coords={"y": [0.0], "x": [0.0]},
            name="raw",
        ),
        param="tair_2m",
        interp_date=pd.Timestamp("2026-05-13 12:30", tz="Europe/Rome"),
    )
    writer = TiffWriter(
        root=tmp_path,
        filename_pattern="{date}_{var}",
    ).initialize(freq="D")
    written = []

    def capture_path(arr, out_path, overwrite=False):
        written.append(Path(out_path).name)

    monkeypatch.setattr(writer, "_to_tiff", capture_path)

    writer.write(data)

    assert written == ["2026-05-13_tair_2m.tif"]


def test_tiff_writer_explicit_date_format_is_preserved_on_initialize(tmp_path, monkeypatch):
    data = GridWriter.prepare_grid_for_output(
        xr.DataArray(
            [[1.0]],
            dims=("y", "x"),
            coords={"y": [0.0], "x": [0.0]},
            name="tair_2m",
        ),
        param="tair_2m",
        interp_date=pd.Timestamp("2026-05-13 12:30", tz="Europe/Rome"),
    )
    writer = TiffWriter(
        root=tmp_path,
        filename_pattern="{var}_{date}",
        date_format="%Y%m%d%H%M",
    ).initialize(freq="D")
    written = []

    def capture_path(arr, out_path, overwrite=False):
        written.append(Path(out_path).name)

    monkeypatch.setattr(writer, "_to_tiff", capture_path)

    writer.write(data)

    assert written == ["tair_2m_202605131230.tif"]


def test_tiff_writer_requires_root_keyword(tmp_path):
    with pytest.raises(TypeError):
        TiffWriter(path=tmp_path)


def test_build_jobs_is_single_parameter_and_end_exclusive():
    tz = "Europe/Rome"
    stations = gpd.GeoDataFrame(
        {
            "station_id": ["a", "b", "c"],
            "elevation": [1000.0, 1200.0, 1400.0],
        },
        geometry=[Point(11.0, 46.0), Point(11.1, 46.1), Point(11.2, 46.2)],
        crs=4326,
    ).set_index("station_id")
    datetimes = pd.to_datetime(["2026-05-13", "2026-05-14"]).tz_localize(tz)
    meteo_data = MeteoData(
        stations=stations,
        observations=pd.DataFrame(
            {
                "station_id": ["a", "a", "b", "b", "c", "c"],
                "datetime": list(datetimes) * 3,
                "tair_2m": [10.0, 11.0, 9.0, 10.0, 8.0, 9.0],
            }
        ),
    )
    target_grid = xr.DataArray(
        [[1000.0]],
        dims=("y", "x"),
        coords={"y": [46.0], "x": [11.0]},
    ).rio.write_crs(4326)

    jobs = list(
        meteo_data.build_jobs(
            start=pd.Timestamp("2026-05-13", tz=tz),
            end=pd.Timestamp("2026-05-14", tz=tz),
            param="tair_2m",
        )
    )

    assert len(jobs) == 1
    job = jobs[0]
    y, X, x_coords, y_coords, ids = job.to_arrays()
    assert job.timestamp == pd.Timestamp("2026-05-13", tz=tz)
    assert job.training_points.crs.to_epsg() == 4326
    assert X.shape == (3, 1)
    assert y.tolist() == [10.0, 9.0, 8.0]
    assert x_coords.tolist() == [11.0, 11.1, 11.2]
    assert y_coords.tolist() == [46.0, 46.1, 46.2]
    assert ids.tolist() == ["a", "b", "c"]


def test_update_elevation_fills_missing_values_from_dem_by_default():
    tz = "Europe/Rome"
    stations = gpd.GeoDataFrame(
        {
            "station_id": ["a", "b"],
            "elevation": [999.0, np.nan],
        },
        geometry=[Point(11.0, 46.0), Point(11.1, 46.1)],
        crs=4326,
    ).set_index("station_id")
    meteo_data = MeteoData(
        stations=stations,
        observations=pd.DataFrame(
            {
                "station_id": ["a", "b"],
                "datetime": pd.to_datetime(["2026-05-13", "2026-05-13"]).tz_localize(tz),
                "tair_2m": [10.0, 9.0],
            }
        ),
    )
    dem = _dem(
        xr.DataArray(
            [[100.0, 200.0], [300.0, 400.0]],
            dims=("y", "x"),
            coords={"y": [46.0, 46.1], "x": [11.0, 11.1]},
        ).rio.write_crs(4326)
    )

    updated = meteo_data.update_elevation(dem)

    assert updated.stations.loc["a", "elevation"] == 999.0
    assert updated.stations.loc["b", "elevation"] == 400.0


def test_update_elevation_overwrite_replaces_all_values_from_dem():
    tz = "Europe/Rome"
    stations = gpd.GeoDataFrame(
        {
            "station_id": ["a", "b"],
            "elevation": [999.0, np.nan],
        },
        geometry=[Point(11.0, 46.0), Point(11.1, 46.1)],
        crs=4326,
    ).set_index("station_id")
    meteo_data = MeteoData(
        stations=stations,
        observations=pd.DataFrame(
            {
                "station_id": ["a", "b"],
                "datetime": pd.to_datetime(["2026-05-13", "2026-05-13"]).tz_localize(tz),
                "tair_2m": [10.0, 9.0],
            }
        ),
    )
    dem = _dem(
        xr.DataArray(
            [[100.0, 200.0], [300.0, 400.0]],
            dims=("y", "x"),
            coords={"y": [46.0, 46.1], "x": [11.0, 11.1]},
        ).rio.write_crs(4326)
    )

    updated = meteo_data.update_elevation(dem, overwrite=True)

    assert updated.stations.loc["a", "elevation"] == 100.0
    assert updated.stations.loc["b", "elevation"] == 400.0
