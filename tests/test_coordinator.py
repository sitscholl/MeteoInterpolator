import asyncio
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Point

from src.coordinator import (
    DistanceFieldPrecomputeRequest,
    InterpolationCoordinator,
)
from src.interpolate import DistanceField
from src.meteo.base import Station


class _RecordingDistanceCalculator:
    def __init__(self):
        self.calls = []

    def calculate_fields(self, dem, x_coords, y_coords, point_ids):
        self.calls.append(
            {
                "dem": dem,
                "x_coords": list(x_coords),
                "y_coords": list(y_coords),
                "point_ids": list(point_ids),
            }
        )
        return DistanceField(
            "recorded",
            xr.DataArray(
                np.ones((len(point_ids), 2, 2)),
                dims=("id", "y", "x"),
                coords={
                    "id": list(point_ids),
                    "y": [0.0, 1.0],
                    "x": [0.0, 1.0],
                },
            ),
        )


class _RecordingMeteoLoader:
    freq = "D"

    def __init__(self):
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def get_stations_for_sensors(self, sensors):
        return {sensor: ["a"] for sensor in sensors}

    async def get_data(self, **kwargs):
        self.calls.append(kwargs)
        return Station(
            id="a",
            x=11.0,
            y=46.0,
            crs=4326,
            data=pd.DataFrame(
                {
                    "datetime": [pd.Timestamp("2026-01-01", tz=kwargs["target_timezone"])],
                    "station_id": ["a"],
                    "tair_2m": [1.0],
                }
            ),
        )


def _coordinator(cache_manager=None):
    station_catalog = gpd.GeoDataFrame(
        {"station_id": ["b", "a"]},
        geometry=[Point(1.0, 1.0), Point(0.0, 0.0)],
        crs=4326,
    ).set_index("station_id")
    distance_calculator = _RecordingDistanceCalculator()
    context = SimpleNamespace(
        station_ids=["b", "a"],
        station_catalog=station_catalog,
        distance_calculator=distance_calculator,
        cache_manager=cache_manager,
        dem=SimpleNamespace(crs=4326, data=xr.DataArray([[1.0]], dims=("y", "x"))),
    )
    return InterpolationCoordinator(context), distance_calculator


def test_load_meteo_data_passes_request_timezone_and_sensor_codes():
    meteo_loader = _RecordingMeteoLoader()
    context = SimpleNamespace(
        station_ids=["a"],
        meteo_loader=meteo_loader,
        interpolator=SimpleNamespace(min_sample_size=1),
    )
    coordinator = InterpolationCoordinator(context)
    tzinfo = ZoneInfo("Europe/Rome")

    result = asyncio.run(
        coordinator._load_meteo_data(
            ["a"],
            pd.Timestamp("2026-01-01", tz=tzinfo),
            pd.Timestamp("2026-01-02", tz=tzinfo),
            tzinfo,
            "tair_2m",
        )
    )

    assert result.n_stations == 1
    assert meteo_loader.calls[0]["target_timezone"] is tzinfo
    assert meteo_loader.calls[0]["sensor_codes"] == ["tair_2m"]


def test_prepare_distance_fields_uses_stable_sorted_station_set():
    coordinator, distance_calculator = _coordinator()

    coordinator._prepare_distance_fields(jobs=[object()])

    assert distance_calculator.calls[0]["point_ids"] == ["a", "b"]
    assert distance_calculator.calls[0]["x_coords"] == [0.0, 1.0]
    assert distance_calculator.calls[0]["y_coords"] == [0.0, 1.0]


def test_precompute_distance_fields_requires_cache_enabled():
    coordinator, _ = _coordinator(cache_manager=None)

    with pytest.raises(ValueError, match="cache.enabled=true"):
        coordinator._calculate_stable_distance_fields(require_cache=True)


def test_precompute_distance_fields_returns_requested_station_subset():
    coordinator, distance_calculator = _coordinator(cache_manager=object())

    result = asyncio.run(
        coordinator.precompute_distance_fields(
            DistanceFieldPrecomputeRequest(station_ids=["b"])
        )
    )

    assert result.station_ids == ["b"]
    assert result.n_sources == 1
    assert distance_calculator.calls[0]["point_ids"] == ["b"]
