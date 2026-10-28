import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.database.db import InterpolationDB


@pytest.fixture
def station_client(monkeypatch, tmp_path):
    # Import requires a config path; station endpoints do not initialize runtime.
    config = Path(__file__).resolve().parents[1] / "config.example.yaml"
    monkeypatch.setenv("CONFIG_FILE", str(config))
    api = importlib.import_module("src.api.app")
    database = InterpolationDB(f"sqlite:///{(tmp_path / 'stations.db').as_posix()}")
    monkeypatch.setattr(
        api.app.state, "runtime", SimpleNamespace(db=database), raising=False
    )
    # Without a context manager, TestClient skips the app's expensive lifespan.
    client = TestClient(api.app)
    try:
        yield client, database
    finally:
        client.close()
        database.close()


def test_list_catalog_station_preserves_metadata(station_client):
    client, database = station_client
    payload = {
        "id": "station-1", "latitude": 46.0, "longitude": 11.0,
        "elevation": 300.0, "crs": 4326, "name": "Test station", "status": "active",
    }
    database.add_station(**payload)
    response = client.get("/api/stations", params={"station_id": "station-1"})
    assert response.status_code == 200
    assert response.json() == [payload]
    assert database.get_stations("station-1")[0].crs == 4326


def test_station_creation_endpoint_is_disabled(station_client):
    client, database = station_client
    response = client.post("/api/stations", json={
        "id": "station-1", "latitude": 46.0, "longitude": 11.0, "elevation": 300.0,
    })
    assert response.status_code == 405
    assert not database.station_exists("station-1")


@pytest.mark.parametrize('status,expected', [
    (None, ['active-station', 'inactive-station']),
    ('active', ['active-station']), ('inactive', ['inactive-station']),
])
def test_list_stations_filters_status(station_client, status, expected):
    client, database = station_client
    database.add_station('active-station', 46, 11, 1000, 4326, status='active')
    database.add_station('inactive-station', 46, 11, 1000, 4326, status='inactive')
    params = {} if status is None else {'status': status}
    response = client.get('/api/stations', params=params)
    assert response.status_code == 200
    assert sorted(station['id'] for station in response.json()) == expected


def test_inactive_station_is_available_by_id(station_client):
    client, database = station_client
    database.add_station('inactive-station', 46, 11, 1000, 4326, status='inactive')
    response = client.get('/api/stations', params={'station_id': 'inactive-station'})
    assert response.status_code == 200
    assert response.json()[0]['status'] == 'inactive'


@pytest.mark.parametrize('status', [None, 'active', 'inactive'])
def test_unknown_station_returns_not_found(station_client, status):
    client, _ = station_client
    params = {'station_id': 'unknown'}
    if status is not None:
        params['status'] = status
    assert client.get('/api/stations', params=params).status_code == 404
