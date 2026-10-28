import math
from datetime import UTC, datetime

import pandas as pd
import pytest
from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError

from src.database import models
from src.database.db import InterpolationDB
from src.database.models import DATABASE_VARIABLES
from src.meteo.retrieve import retrieve_stations


@pytest.fixture
def database():
    db = InterpolationDB("sqlite:///:memory:")
    yield db
    db.close()


@pytest.mark.parametrize(
    ("latitude", "longitude", "elevation"),
    [
        (91, 11, 300),
        (-91, 11, 300),
        (46, 181, 300),
        (46, -181, 300),
        (46, 11, None),
        (46, 11, math.nan),
    ],
)
def test_add_station_rejects_invalid_required_coordinates(
    database, latitude, longitude, elevation
):
    with pytest.raises(ValueError):
        database.add_station("station-1", latitude, longitude, elevation, crs=4326, status="active")


def test_sqlite_enforces_measurement_foreign_keys(database):
    with pytest.raises(IntegrityError):
        with database.engine.begin() as connection:
            connection.execute(
                insert(models.Measurement).values(
                    station_id="unknown-station",
                    datetime=datetime(2026, 1, 1, tzinfo=UTC),
                    tair_2m=1.0,
                    retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
                )
            )


def test_add_station_data_stores_retrieval_timestamp(database):
    database.add_station("station-1", 46, 11, 300, crs=4326, status="active")
    database.add_station_data(
        "station-1",
        pd.DataFrame(
            {
                "station_id": ["station-1"],
                "datetime": ["2026-01-01T00:00:00Z"],
                "tair_2m": [2.5],
            }
        ),
    )
    database.add_station_data(
        "station-1",
        pd.DataFrame(
            {
                "station_id": ["station-1"],
                "datetime": ["2026-01-01T00:00:00Z"],
                "tair_2m": [3.0],
            }
        ),
    )

    data = database.get_data(
        "station-1",
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-02T00:00:00Z"),
    )

    assert len(data) == 1
    assert data["tair_2m"].tolist() == [3.0]
    assert data["retrieved_at"].notna().all()


def test_wide_row_stores_every_supported_sensor(database):
    database.add_station("station-1", 46, 11, 300, crs=4326, status="active")
    values = {name: float(i) for i, name in enumerate(DATABASE_VARIABLES)}
    written = database.add_station_data(
        "station-1", pd.DataFrame([{"datetime": "2026-01-01T00:00:00Z", **values}])
    )
    data = database.get_data("station-1", pd.Timestamp("2026-01-01", tz="UTC"))
    assert written == len(data) == 1
    assert data.iloc[0][DATABASE_VARIABLES].to_dict() == values
    assert database.get_stations("station-1")[0].crs == 4326


def test_partial_upsert_preserves_missing_values_and_accepts_zero(database):
    database.add_station("station-1", 46, 11, 300, crs=4326, status="active")
    timestamp = "2026-01-01T00:00:00Z"
    database.add_station_data("station-1", pd.DataFrame([{
        "datetime": timestamp, "tair_2m": 5.0, "relative_humidity": 75.0,
        "precipitation": 2.0,
    }]))
    # One missing column, one explicit missing value, and a valid zero.
    database.add_station_data("station-1", pd.DataFrame([{
        "datetime": timestamp, "tair_2m": float("nan"), "precipitation": 0.0,
    }]))
    row = database.get_data("station-1", pd.Timestamp(timestamp)).iloc[0]
    assert row["tair_2m"] == 5.0
    assert row["relative_humidity"] == 75.0
    assert row["precipitation"] == 0.0


def test_missing_sensor_in_one_row_does_not_drop_other_values(database):
    database.add_station("station-1", 46, 11, 300, crs=4326, status="active")
    written = database.add_station_data("station-1", pd.DataFrame({
        "datetime": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
        "tair_2m": [1.0, None, None],
        "relative_humidity": [None, 75.0, None],
    }))
    assert written == 2
    data = database.get_data("station-1", pd.Timestamp("2026-01-01", tz="UTC"))
    assert len(data) == 2
    assert data["tair_2m"].dropna().tolist() == [1.0]
    assert data["relative_humidity"].dropna().tolist() == [75.0]


def test_invalid_columns_can_be_rejected_or_dropped(database):
    database.add_station("station-1", 46, 11, 300, crs=4326, status="active")
    data = pd.DataFrame([{
        "datetime": "2026-01-01T00:00:00Z", "tair_2m": 5.0, "unknown_sensor": 2.0,
    }])
    with pytest.raises(ValueError, match="unknown_sensor"):
        database.add_station_data("station-1", data)
    assert database.add_station_data("station-1", data, drop_invalid_variables=True) == 1
    assert "unknown_sensor" in data.columns  # Caller-owned data is unchanged.


def test_get_last_row_and_end_exclusive_multi_station_query(database):
    for station_id in ["station-1", "station-2"]:
        database.add_station(station_id, 46, 11, 300, crs=4326, status="active")
        assert database.get_last_row(station_id) is None
        database.add_station_data(station_id, pd.DataFrame({
            "datetime": pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
            "tair_2m": [1.0, 2.0],
        }))
    assert database.get_last_row("unknown-station") is None
    assert database.get_last_row("station-1").datetime == datetime(2026, 1, 1, 1)
    data = database.get_data(
        ["station-1", "station-2"],
        pd.Timestamp("2026-01-01", tz="UTC"),
        pd.Timestamp("2026-01-01 01:00", tz="UTC"),
    )
    assert set(data["station_id"]) == {"station-1", "station-2"}
    assert data["tair_2m"].tolist() == [1.0, 1.0]


def test_station_crs_is_normalized_on_insert_and_update(database):
    station = database.add_station("station-1", 46, 11, 300, crs="4326", status="active")
    assert station.crs == 4326
    updated = database.update_station(
        "station-1", only_missing=False, crs="4326", name="Updated station"
    )
    assert updated.crs == 4326
    assert updated.name == "Updated station"
    with pytest.raises(ValueError, match="4326"):
        database.update_station("station-1", only_missing=False, crs=3857)
    assert database.get_stations("station-1")[0].crs == 4326


def test_retrieve_stations_selects_sensors_and_restores_crs(database):
    database.add_station("station-1", 46, 11, 300, crs=4326, status="active")
    database.add_station_data("station-1", pd.DataFrame([{
        "datetime": "2026-01-01T00:00:00Z", "tair_2m": 5.0, "water_level": 2.0,
    }]))
    stations = retrieve_stations(
        database, ["station-1"], ["water_level"],
        pd.Timestamp("2026-01-01", tz="UTC"),
        pd.Timestamp("2026-01-02", tz="UTC"), target_timezone="Europe/Rome",
    )
    assert len(stations) == 1
    assert stations[0].crs == 4326
    assert set(stations[0].data.columns) == {"station_id", "datetime", "water_level"}
    assert stations[0].data["water_level"].tolist() == [2.0]
    assert str(stations[0].data["datetime"].dt.tz) == "Europe/Rome"


def _catalog(*entries):
    return pd.DataFrame([
        dict(station_id=station_id, name=station_id, latitude=46.0,
             longitude=11.0, elevation=1000.0, crs=4326, status=status)
        for station_id, status in entries
    ])


@pytest.mark.parametrize('backend', ['memory', 'file'])
def test_sync_catalog_updates_membership_and_preserves_history(tmp_path, backend):
    engine = 'sqlite:///:memory:' if backend == 'memory' else f"sqlite:///{(tmp_path / 'sync.db').as_posix()}"
    database = InterpolationDB(engine)
    try:
        database.add_station('removed', 46, 11, 900, 4326, status='active')
        database.add_station_data('removed', pd.DataFrame([
            {'datetime': '2026-01-01T00:00:00Z', 'tair_2m': 5.0},
        ]))
        catalog = _catalog(('a', 'active'), ('b', 'inactive'))
        database.sync_catalog(catalog)
        assert [station.id for station in database.get_stations()] == ['a']
        assert {station.id: station.status for station in database.get_stations(include_inactive=True)} == {
            'a': 'active', 'b': 'inactive', 'removed': 'inactive',
        }
        assert database.station_exists('b')
        assert database.get_last_row('removed').tair_2m == 5.0

        catalog.loc[0, ['status', 'name']] = ['inactive', 'Updated name']
        catalog.loc[1, 'status'] = 'active'
        database.sync_catalog(catalog)
        database.sync_catalog(catalog)  # Repeating the same catalog is idempotent.
        assert [station.id for station in database.get_stations()] == ['b']
        assert database.get_stations('a', include_inactive=True)[0].name == 'Updated name'
        assert len(database.get_stations(include_inactive=True)) == 3
    finally:
        database.close()


def test_sync_catalog_rolls_back_prior_inserts_and_updates(database):
    database.add_station('a', 46, 11, 900, 4326, status='active', name='Original')
    database.add_station('removed', 46, 11, 900, 4326, status='active')
    catalog = _catalog(('a', 'inactive'), ('new', 'active'), ('invalid', 'active'))
    catalog.loc[2, 'latitude'] = 999.0
    with pytest.raises(ValueError):
        database.sync_catalog(catalog)
    stations = {station.id: station for station in database.get_stations(include_inactive=True)}
    assert set(stations) == {'a', 'removed'}
    assert stations['a'].name == 'Original'
    assert stations['a'].elevation == 900
    assert stations['a'].status == stations['removed'].status == 'active'


def test_station_and_measurement_helpers_share_outer_transaction(database):
    with pytest.raises(RuntimeError, match='force rollback'):
        with database.Session.begin() as session:
            database.add_station('new', 46, 11, 1000, 4326, status='active', db_session=session)
            assert database.station_exists('new', db_session=session)
            database.add_station_data('new', pd.DataFrame([
                {'datetime': '2026-01-01T00:00:00Z', 'tair_2m': 5.0},
            ]), db_session=session)
            raise RuntimeError('force rollback')
    assert database.get_stations(include_inactive=True) == []
    with database.Session() as session:
        assert session.query(models.Measurement).count() == 0


@pytest.mark.parametrize('operation', ['add', 'update'])
def test_station_status_is_normalized_and_validated(database, operation):
    if operation == 'add':
        station = database.add_station('a', 46, 11, 1000, 4326, status='ACTIVE')
        with pytest.raises(ValueError, match='Invalid status'):
            database.add_station('invalid', 46, 11, 1000, 4326, status='unknown')
    else:
        database.add_station('a', 46, 11, 1000, 4326, status='inactive')
        station = database.update_station('a', only_missing=False, status='ACTIVE')
        with pytest.raises(ValueError, match='Invalid status'):
            database.update_station('a', only_missing=False, status='unknown')
    assert station.status == 'active'
    assert database.get_stations('a')[0].status == 'active'
