import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
import pytest
import rioxarray  # noqa: F401
import xarray as xr

from src.database.db import InterpolationDB
from src.domain.dem import DEM
from src.meteo.ingest import MeteoIngestor
from src.meteo.resample import MeteoResampler
from src.meteo.types import Station


class _FakeLoader:
    freq = "10min"

    def __init__(self, station: Station):
        self.station = station
        self.requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    async def get_data(self, **kwargs) -> Station:
        self.requests.append(kwargs)
        return self.station


@pytest.fixture
def database():
    db = InterpolationDB("sqlite:///:memory:")
    yield db
    db.close()


@pytest.fixture
def dem():
    data = xr.DataArray(
        [[1000.0]],
        coords={"y": [46.0], "x": [11.0]},
        dims=("y", "x"),
    ).rio.write_crs(4326)
    return DEM(path="test-dem", data=data, fingerprint="test")


def _station() -> Station:
    return Station(
        id="station-a",
        x=11.0,
        y=46.0,
        crs=4326,
        elevation=1000.0,
        data=pd.DataFrame(
            {
                "station_id": ["station-a", "station-a"],
                "datetime": pd.to_datetime(
                    ["2026-01-01 00:00", "2026-01-01 00:10"],
                    utc=True,
                ),
                "tair_2m": [1.0, 3.0],
            }
        ),
    )


def test_get_start_date_uses_earliest_variable_and_overlap(database):
    ingestor = MeteoIngestor(
        loader=_FakeLoader(_station()),
        database=database,
        resampler=MeteoResampler(),
        overlap_days=7,
    )

    start = ingestor._get_start_date(
        {
            "tair_2m": SimpleNamespace(datetime=datetime(2026, 1, 10)),
            "relative_humidity": SimpleNamespace(
                datetime=datetime(2026, 1, 12, tzinfo=UTC)
            ),
        }
    )

    assert start == pd.Timestamp("2026-01-03T00:00:00Z")


def test_get_start_date_uses_default_when_a_variable_has_no_data(database):
    default_start = datetime(2020, 1, 1, tzinfo=UTC)
    ingestor = MeteoIngestor(
        loader=_FakeLoader(_station()),
        database=database,
        resampler=MeteoResampler(),
        default_start_date=default_start,
    )

    start = ingestor._get_start_date(
        {"tair_2m": SimpleNamespace(datetime=datetime(2026, 1, 10)), "wind": None}
    )

    assert start == pd.Timestamp(default_start)


def test_refresh_station_inserts_data_then_uses_overlap_watermark(database, dem):
    loader = _FakeLoader(_station())
    ingestor = MeteoIngestor(
        loader=loader,
        database=database,
        resampler=MeteoResampler(),
        overlap_days=2,
    )
    first_end = datetime(2026, 1, 2, tzinfo=UTC)

    first = asyncio.run(
        ingestor.refresh_station("station-a", ["tair_2m"], dem, end=first_end)
    )

    assert first.status == "completed"
    assert first.observations_written == 1
    assert loader.requests[0]["start"] == pd.Timestamp("2020-01-01T00:00:00Z")

    second = asyncio.run(
        ingestor.refresh_station(
            "station-a",
            ["tair_2m"],
            dem,
            end=datetime(2026, 1, 3, tzinfo=UTC),
        )
    )

    assert second.status == "completed"
    assert loader.requests[1]["start"] == pd.Timestamp("2025-12-30T00:00:00Z")
