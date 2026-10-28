import asyncio
from datetime import UTC, datetime

import pandas as pd
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from sqlalchemy import event, select

from src.database import models
from src.database.db import InterpolationDB
from src.domain.dem import DEM
from src.meteo.ingest import MeteoIngestor
from src.meteo.resample import MeteoResampler
from src.meteo.types import Station


class _FakeLoader:
    freq = "10min"
    provider_name = "province_api"

    def __init__(self, station: Station):
        self.station = station
        self.requests = []
        self.discovery_requests = []
        self.sensor_codes = {station.id: {"LT"}}
        self.enters = 0
        self.exits = 0
        self.active = False

    async def __aenter__(self):
        self.enters += 1
        self.active = True
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.exits += 1
        self.active = False
        return None

    async def get_sensors_for_station(self, station_code):
        assert self.active, "Sensor discovery requires an active loader"
        self.discovery_requests.append(station_code)
        codes = self.sensor_codes[station_code]
        if isinstance(codes, Exception):
            raise codes
        return codes

    async def get_data(self, **kwargs) -> Station:
        assert self.active, "Data loading requires an active loader"
        self.requests.append(kwargs)
        return self.station


@pytest.fixture
def database():
    db = InterpolationDB("sqlite:///:memory:")
    db.sync_catalog(pd.DataFrame([{
        'station_id': 'station-a', 'name': 'Catalog station',
        'latitude': 46.0, 'longitude': 11.0, 'elevation': 1000.0,
        'crs': 4326, 'status': 'active',
    }]))
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
                "station_id": ["station-a"] * 6,
                "datetime": pd.to_datetime(
                    [f"2026-01-01 00:{minute:02}" for minute in range(0, 60, 10)],
                    utc=True,
                ),
                "tair_2m": [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
            }
        ),
    )


@pytest.mark.parametrize("last_updated", [datetime(2026, 1, 10), datetime(2026, 1, 10, tzinfo=UTC)])
def test_get_start_date_uses_latest_measurement_and_overlap(database, last_updated):
    ingestor = MeteoIngestor(
        loader=_FakeLoader(_station()),
        database=database,
        resampler=MeteoResampler(),
        overlap_days=7,
    )

    start = ingestor._get_start_date(last_updated)

    assert start == pd.Timestamp("2026-01-03T00:00:00Z")


def test_get_start_date_uses_default_when_station_has_no_data(database):
    default_start = datetime(2020, 1, 1, tzinfo=UTC)
    ingestor = MeteoIngestor(
        loader=_FakeLoader(_station()),
        database=database,
        resampler=MeteoResampler(),
        default_start_date=default_start,
    )

    start = ingestor._get_start_date(None)

    assert start == pd.Timestamp(default_start)


def test_upsert_station_inserts_derived_data_then_uses_overlap_watermark(database, dem):
    loader = _FakeLoader(_station())
    ingestor = MeteoIngestor(
        loader=loader,
        database=database,
        resampler=MeteoResampler(),
        overlap_days=2,
    )
    first_end = datetime(2026, 1, 2, tzinfo=UTC)

    first = asyncio.run(
        ingestor.upsert_station("station-a", dem, end=first_end)
    )

    assert first.status == "completed"
    assert first.observations_written == 1
    assert loader.requests[0]["start"] == pd.Timestamp("2020-01-01T00:00:00Z")

    second = asyncio.run(
        ingestor.upsert_station(
            "station-a",
            dem,
            end=datetime(2026, 1, 3, tzinfo=UTC),
        )
    )

    assert second.status == "completed"
    assert second.observations_written == 1
    assert loader.requests[1]["start"] == pd.Timestamp("2025-12-30T00:00:00Z")
    assert loader.discovery_requests == ["station-a", "station-a"]
    stored = database.get_data("station-a", pd.Timestamp("2026-01-01", tz="UTC"))
    assert len(stored) == 1
    assert stored["tair_2m"].tolist() == [6.0]
    assert stored["tair_2m_min"].tolist() == [1.0]
    assert stored["tair_2m_max"].tolist() == [11.0]


def test_automatic_sensors_are_discovered_inside_loader_context(database, dem):
    loader = _FakeLoader(_station())
    ingestor = MeteoIngestor(loader, database, MeteoResampler())
    result = asyncio.run(ingestor.upsert_station(
        "station-a", dem, end=datetime(2026, 1, 2, tzinfo=UTC)
    ))
    assert result.status == "completed"
    assert loader.discovery_requests == ["station-a"]
    assert loader.requests[0]["sensor_codes"] == ["tair_2m"]
    assert (loader.enters, loader.exits) == (1, 1)


def test_upsert_reuses_supplied_session_without_closing_it(database, dem):
    loader = _FakeLoader(_station())
    ingestor = MeteoIngestor(loader, database, MeteoResampler())

    async def run():
        async with loader:
            result = await ingestor.upsert_station(
                "station-a", dem, session=loader,
                end=datetime(2026, 1, 2, tzinfo=UTC),
            )
            assert loader.active
            assert loader.exits == 0
            return result

    assert asyncio.run(run()).status == "completed"
    assert (loader.enters, loader.exits) == (1, 1)


def test_empty_sensors_skip_without_fetching_or_inserting(database, dem):
    loader = _FakeLoader(_station())
    ingestor = MeteoIngestor(loader, database, MeteoResampler())
    loader.sensor_codes["station-a"] = set()
    result = asyncio.run(ingestor.upsert_station("station-a", dem, persist_requested_end=True))
    assert result.status == "skipped"
    assert loader.requests == []
    assert database.get_last_row("station-a") is None
    assert database.get_last_ingestion_task("station-a") is None


def test_insufficient_coverage_preserves_existing_values(database, dem):
    loader = _FakeLoader(_station())
    ingestor = MeteoIngestor(loader, database, MeteoResampler())
    kwargs = dict(end=datetime(2026, 1, 2, tzinfo=UTC))
    assert asyncio.run(ingestor.upsert_station("station-a", dem, **kwargs)).observations_written == 1
    loader.station.data = loader.station.data.iloc[:2].copy()
    result = asyncio.run(ingestor.upsert_station("station-a", dem, **kwargs))
    assert result.status == "completed"
    assert result.observations_written == 0
    stored = database.get_data("station-a", pd.Timestamp("2026-01-01", tz="UTC"))
    assert stored["tair_2m"].tolist() == [6.0]
    assert stored["tair_2m_min"].tolist() == [1.0]
    assert stored["tair_2m_max"].tolist() == [11.0]


@pytest.mark.parametrize("source_freq", ["D", "0min", "-10min"])
def test_invalid_frequency_fails_before_fetch_or_write(database, dem, source_freq):
    loader = _FakeLoader(_station())
    loader.freq = source_freq
    result = asyncio.run(MeteoIngestor(loader, database, MeteoResampler()).upsert_station(
        "station-a", dem
    ))
    assert result.status == "failed"
    assert loader.requests == []
    assert database.get_last_row("station-a") is None


def test_utc_hour_boundaries_are_used_for_non_hour_offset(database, dem):
    loader = _FakeLoader(_station())
    ingestor = MeteoIngestor(
        loader, database, MeteoResampler(),
        default_start_date=pd.Timestamp("2026-01-01 06:15", tz="Asia/Kathmandu"),
    )
    result = asyncio.run(ingestor.upsert_station(
        "station-a", dem,
        end=pd.Timestamp("2026-01-02 06:15", tz="Asia/Kathmandu"),
    ))
    assert result.status == "completed"
    assert result.requested_start == pd.Timestamp("2026-01-01 00:00", tz="UTC")
    assert result.requested_end == pd.Timestamp("2026-01-02 00:00", tz="UTC")


def test_refresh_continues_after_discovery_failure_and_reuses_session(database):
    for station_id in ["station-b", "station-c"]:
        database.add_station(station_id, 46, 11, 1000, crs=4326, status='active')
    database.add_station('inactive-station', 46, 11, 1000, crs=4326, status='inactive')
    loader = _FakeLoader(_station())
    loader.sensor_codes = {
        "station-a": {"LT", "LF"},
        "station-b": RuntimeError("Sensor catalogue unavailable"),
        "station-c": set(),
    }
    result = asyncio.run(MeteoIngestor(loader, database, MeteoResampler()).refresh_database(
        end=datetime(2026, 1, 2, 0, 29, tzinfo=UTC)
    ))
    assert result.n_completed == 1
    assert result.n_failed == 1
    assert result.n_skipped == 1
    assert result.written_rows == 1
    assert set(result.processed_stations) == {"station-a", "station-b", "station-c"}
    assert set(loader.requests[0]["sensor_codes"]) == {"tair_2m", "relative_humidity"}
    assert len(loader.requests) == 1
    assert (loader.enters, loader.exits) == (1, 1)
    for item in result.station_results:
        if item.status == "failed":
            # Discovery failed before the requested interval was validated.
            assert "Sensor catalogue unavailable" in item.error
        else:
            assert item.requested_end == pd.Timestamp("2026-01-02", tz="UTC")


def test_data_failure_leaves_catalog_station_without_observations(database, dem, monkeypatch):
    loader = _FakeLoader(_station())

    async def fail(**kwargs):
        raise RuntimeError("A required request failed")

    monkeypatch.setattr(loader, "get_data", fail)
    result = asyncio.run(MeteoIngestor(loader, database, MeteoResampler()).upsert_station(
        "station-a", dem
    ))
    assert result.status == "failed"
    assert "required request failed" in result.error
    assert database.station_exists("station-a")
    assert database.get_last_row("station-a") is None


@pytest.mark.parametrize("provide_dem", [False, True])
def test_missing_elevation_requires_dem(database, dem, provide_dem):
    station = _station()
    station.elevation = None
    loader = _FakeLoader(station)
    result = asyncio.run(MeteoIngestor(loader, database, MeteoResampler()).upsert_station(
        "station-a", dem=dem if provide_dem else None,
        end=datetime(2026, 1, 2, tzinfo=UTC),
    ))
    if provide_dem:
        assert result.status == "completed"
        assert database.get_stations("station-a")[0].elevation == 1000.0
    else:
        assert result.status == "failed"
        assert "no elevation" in result.error
        assert database.get_last_row("station-a") is None


def test_ingestion_requests_all_available_supported_sensors(database, dem):
    station = _station()
    station.data["relative_humidity"] = 75.0
    station.data["water_level"] = 2.0
    loader = _FakeLoader(station)
    loader.sensor_codes[station.id] = {"LT", "LF", "W", "unknown-provider-code"}
    result = asyncio.run(MeteoIngestor(loader, database, MeteoResampler()).upsert_station(
        station.id, dem, end=datetime(2026, 1, 2, tzinfo=UTC)
    ))
    assert result.status == "completed"
    assert result.observations_written == 1
    assert set(loader.requests[0]["sensor_codes"]) == {"tair_2m", "relative_humidity", "water_level"}
    row = database.get_last_row(station.id)
    assert row.tair_2m == 6.0
    assert row.relative_humidity == 75.0
    assert row.water_level == 2.0
    assert database.get_last_ingestion_task(station.id) is None


@pytest.mark.parametrize("empty_response", [False, True])
def test_refresh_uses_persisted_end_and_overlap_even_with_no_observations(database, dem, empty_response):
    station = _station()
    if empty_response:
        station.data = None
    loader = _FakeLoader(station)
    ingestor = MeteoIngestor(loader, database, MeteoResampler(), overlap_days=2)
    first_end = datetime(2026, 1, 10, tzinfo=UTC)
    first = asyncio.run(ingestor.upsert_station(
        station.id, dem, end=first_end, persist_requested_end=True,
    ))
    assert first.status == "completed"
    assert first.observations_written == (0 if empty_response else 1)
    assert database.get_last_ingestion_task(station.id).requested_end == datetime(2026, 1, 10)
    second = asyncio.run(ingestor.refresh_database(
        end=datetime(2026, 1, 11, tzinfo=UTC), persist_requested_end=True,
    ))
    assert second.n_completed == 1
    assert loader.requests[1]["start"] == pd.Timestamp("2026-01-08", tz="UTC")
    assert database.get_last_ingestion_task(station.id).requested_end == datetime(2026, 1, 11)


def test_progress_insert_failure_rolls_back_measurement_inserts_and_updates(database, dem):
    station = _station()
    loader = _FakeLoader(station)
    ingestor = MeteoIngestor(loader, database, MeteoResampler())
    end = datetime(2026, 1, 2, tzinfo=UTC)
    assert asyncio.run(ingestor.upsert_station(station.id, dem, end=end)).status == "completed"
    original = station.data.copy()
    station.data["tair_2m"] = 100.0
    additional = original.copy()
    additional["datetime"] += pd.Timedelta(hours=1)
    station.data = pd.concat([station.data, additional], ignore_index=True)

    def reject_progress(mapper, connection, target):
        raise RuntimeError("forced progress failure")

    event.listen(models.IngestionRequest, "before_insert", reject_progress)
    try:
        result = asyncio.run(ingestor.upsert_station(
            station.id, dem, end=end, persist_requested_end=True,
        ))
    finally:
        event.remove(models.IngestionRequest, "before_insert", reject_progress)

    assert result.status == "failed"
    assert result.observations_written == 0
    assert "forced progress failure" in result.error
    with database.Session() as session:
        rows = session.scalars(select(models.Measurement)).all()
        assert len(rows) == 1
        assert rows[0].tair_2m == 6.0
        assert session.scalars(select(models.IngestionRequest)).all() == []


def test_explicit_start_preserves_catalog_metadata(database, dem):
    database.update_station('station-a', only_missing=False, elevation=900)
    loader = _FakeLoader(_station())
    result = asyncio.run(MeteoIngestor(loader, database, MeteoResampler()).upsert_station(
        "station-a", dem, start=datetime(2025, 12, 31, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
    ))
    assert result.status == "completed"
    assert loader.requests[0]["start"] == pd.Timestamp("2025-12-31", tz="UTC")
    stored = database.get_stations("station-a")[0]
    assert stored.elevation == 900.0
    assert stored.crs == 4326


def test_ingestion_requires_catalog_station_before_fetching(database, dem):
    loader = _FakeLoader(_station())
    ingestor = MeteoIngestor(loader, database, MeteoResampler())
    with pytest.raises(ValueError, match='does not exist'):
        asyncio.run(ingestor.upsert_station('unknown', dem))
    assert loader.enters == 0
    assert loader.requests == []
    assert not database.station_exists('unknown')
