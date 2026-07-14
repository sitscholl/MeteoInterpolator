import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray  # noqa: F401
from pathlib import Path
from shapely.geometry import Point

from src.array.cache import CacheManager
from src.domain.dem import DEM
from src.domain.meteo_data import MeteoData
from src.utils import localize_datetime_string


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
