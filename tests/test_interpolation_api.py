from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.schemas import InterpolationRequest, InterpolationSubmission


@pytest.fixture
def interpolation_client(monkeypatch):
    submit = AsyncMock(return_value=InterpolationSubmission(request_id="run-1", status="queued"))
    monkeypatch.setattr(app.state, "coordinator", SimpleNamespace(submit=submit), raising=False)
    # Skip lifespan: these endpoint tests require no database or scheduler.
    client = TestClient(app, raise_server_exceptions=False)
    try:
        yield client, submit
    finally:
        client.close()


@pytest.mark.parametrize("freq,expected", [(None, "D"), ("h", "h"), ("D", "D")])
def test_interpolation_endpoint_forwards_frequency(interpolation_client, freq, expected):
    client, submit = interpolation_client
    params = {"param": "tair_2m", "start": "2026-01-01", "end": "2026-01-02"}
    if freq is not None:
        params["freq"] = freq
    response = client.get("/api/interpolate", params=params)
    assert response.status_code == 200
    assert response.json() == {"request_id": "run-1", "status": "queued"}
    submit.assert_awaited_once()
    request = submit.await_args.args[0]
    assert isinstance(request, InterpolationRequest)
    assert request.freq == expected


@pytest.mark.parametrize("freq", ["2h", "30min", "ME", "d", "H", ""])
def test_interpolation_endpoint_rejects_unsupported_frequency(interpolation_client, freq):
    client, submit = interpolation_client
    response = client.get("/api/interpolate", params={
        "param": "tair_2m", "start": "2026-01-01", "end": "2026-01-02", "freq": freq,
    })
    assert response.status_code == 422
    assert any(error["loc"] == ["query", "freq"] for error in response.json()["detail"])
    submit.assert_not_awaited()
