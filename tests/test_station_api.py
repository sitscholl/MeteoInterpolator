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


def test_create_and_list_station_preserve_crs(station_client):
    client, database = station_client
    payload = {
        "id": "station-1", "latitude": 46.0, "longitude": 11.0,
        "elevation": 300.0, "crs": 4326, "name": "Test station",
    }
    response = client.post("/api/stations", json=payload)
    assert response.status_code == 201
    assert response.json() == payload
    response = client.get("/api/stations", params={"station_id": "station-1"})
    assert response.status_code == 200
    assert response.json() == [payload]
    assert database.get_stations("station-1")[0].crs == 4326


def test_station_creation_requires_crs(station_client):
    client, database = station_client
    response = client.post("/api/stations", json={
        "id": "station-1", "latitude": 46.0, "longitude": 11.0, "elevation": 300.0,
    })
    assert response.status_code == 422
    assert not database.station_exists("station-1")
