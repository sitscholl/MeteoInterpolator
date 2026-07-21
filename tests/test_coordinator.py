import asyncio
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import Point

from src.coordinator import (
    DistanceFieldPrecomputeRequest,
    InterpolationCoordinator,
)
from src.interpolate import DistanceField


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
