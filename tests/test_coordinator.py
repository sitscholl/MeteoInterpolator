import asyncio
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Point

from src.coordinator import InterpolationCoordinator
from src.interpolate import DistanceField
from src.meteo.base import BaseMeteoHandler, Station
from src.runtime import RuntimeContext


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


class _RecordingMeteoLoader(BaseMeteoHandler):
    freq = "D"
    inclusive = "left"

    def __init__(self):
        self.calls = []
        self.sensor_calls = []
        self.entered = 0

    async def __aenter__(self):
        self.entered += 1
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def get_stations_for_sensors(self, sensors):
        self.sensor_calls.append(list(sensors))
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

    @classmethod
    def name(cls):
        return "recording"

    async def get_station_info(self, station_id=None):
        return {}

    async def get_raw_data(self, **kwargs):
        raise NotImplementedError

    def transform(self, raw_data, target_timezone):
        raise NotImplementedError


def _runtime_with_distance_calculator():
    station_catalog = gpd.GeoDataFrame(
        {"station_id": ["b", "a"]},
        geometry=[Point(1.0, 1.0), Point(0.0, 0.0)],
        crs=4326,
    ).set_index("station_id")
    distance_calculator = _RecordingDistanceCalculator()
    runtime = RuntimeContext(config={})
    runtime.station_catalog = station_catalog
    runtime.distance_calculator = distance_calculator
    runtime.dem = SimpleNamespace(
        crs=4326,
        data=xr.DataArray([[1.0]], dims=("y", "x")),
    )
    return runtime, distance_calculator


def test_handler_loads_supported_stations_with_cached_sensor_catalogue():
    meteo_loader = _RecordingMeteoLoader()
    tzinfo = ZoneInfo("Europe/Rome")

    async def load_twice():
        first = await meteo_loader.get_data_for_stations(
            station_ids=["a"],
            sensor_codes=["tair_2m"],
            start=pd.Timestamp("2026-01-01", tz=tzinfo),
            end=pd.Timestamp("2026-01-02", tz=tzinfo),
            target_timezone=tzinfo,
        )
        second = await meteo_loader.get_data_for_stations(
            station_ids=["a"],
            sensor_codes=["tair_2m"],
            start=pd.Timestamp("2026-01-01", tz=tzinfo),
            end=pd.Timestamp("2026-01-02", tz=tzinfo),
            target_timezone=tzinfo,
        )
        return first, second

    first, second = asyncio.run(load_twice())

    assert [station.id for station in first] == ["a"]
    assert [station.id for station in second] == ["a"]
    assert meteo_loader.sensor_calls == [["tair_2m"]]
    assert meteo_loader.entered == 2
    assert meteo_loader.calls[0]["target_timezone"] is tzinfo
    assert meteo_loader.calls[0]["sensor_codes"] == ["tair_2m"]


def test_build_distance_fields_uses_stable_sorted_station_set():
    runtime, distance_calculator = _runtime_with_distance_calculator()

    distance_fields = runtime.build_distance_fields()

    assert isinstance(distance_fields, DistanceField)
    assert distance_calculator.calls[0]["point_ids"] == ["a", "b"]
    assert distance_calculator.calls[0]["x_coords"] == [0.0, 1.0]
    assert distance_calculator.calls[0]["y_coords"] == [0.0, 1.0]


def test_build_distance_fields_requires_a_distance_calculator():
    runtime, _ = _runtime_with_distance_calculator()
    runtime.distance_calculator = None

    with pytest.raises(ValueError, match="no distance calculator"):
        runtime.build_distance_fields()
