import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.schemas import InterpolationRequest, InterpolationSubmission
from src.coordinator import InterpolationCoordinator, PreparedInterpolationRun
from src.meteo.resample import MeteoResampler


@pytest.fixture
def interpolation_client(monkeypatch):
    submit = AsyncMock(return_value=InterpolationSubmission(request_id="run-1", status="queued"))
    monkeypatch.setattr(app.state, "coordinator", SimpleNamespace(submit=submit), raising=False)
    monkeypatch.setattr(
        app.state, "runtime",
        SimpleNamespace(interpolation_status=SimpleNamespace(enabled=True, reason=None)),
        raising=False,
    )
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
    response = client.post("/api/interpolate", params=params)
    assert response.status_code == 202
    assert response.json() == {"request_id": "run-1", "status": "queued"}
    submit.assert_awaited_once()
    request = submit.await_args.args[0]
    assert isinstance(request, InterpolationRequest)
    assert request.freq == expected


@pytest.mark.parametrize("freq", ["2h", "30min", "ME", "d", "H", ""])
def test_interpolation_endpoint_rejects_unsupported_frequency(interpolation_client, freq):
    client, submit = interpolation_client
    response = client.post("/api/interpolate", params={
        "param": "tair_2m", "start": "2026-01-01", "end": "2026-01-02", "freq": freq,
    })
    assert response.status_code == 422
    assert any(error["loc"] == ["query", "freq"] for error in response.json()["detail"])
    submit.assert_not_awaited()


def test_interpolation_endpoint_rejects_submission_when_cache_unavailable(
    interpolation_client, monkeypatch
):
    client, submit = interpolation_client
    monkeypatch.setattr(
        app.state, "runtime",
        SimpleNamespace(interpolation_status=SimpleNamespace(
            enabled=False, reason="No distance fields available."
        )),
    )

    response = client.post("/api/interpolate", params={
        "param": "tair_2m", "start": "2026-01-01", "end": "2026-01-02",
    })

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Interpolation is not available, reason: No distance fields available."
    )
    submit.assert_not_awaited()


@pytest.mark.parametrize("fail", [False, True])
def test_interpolation_responds_and_serves_health_while_work_is_blocked(monkeypatch, fail):
    preparing = threading.Event()
    release_preparation = threading.Event()
    interpolating = threading.Event()
    release_interpolation = threading.Event()
    context = SimpleNamespace(
        interpolation_status=SimpleNamespace(enabled=True),
        resampler=MeteoResampler(), gapfiller=None, config={},
        dem=SimpleNamespace(path="dem.tif", fingerprint="test"),
    )
    coordinator = InterpolationCoordinator(context)

    def prepare(request):
        preparing.set()
        if not release_preparation.wait(5):
            raise TimeoutError("test did not release preparation")
        return PreparedInterpolationRun(prediction_target=None, jobs=[])

    def jobs(prepared):
        interpolating.set()
        if not release_interpolation.wait(5):
            raise TimeoutError("test did not release interpolation")
        if fail:
            raise ValueError("interpolation failed")
        return iter([])

    monkeypatch.setattr(coordinator, "_prepare_request", prepare)
    monkeypatch.setattr(coordinator, "_submit_jobs", jobs)
    monkeypatch.setattr(app.state, "runtime", context, raising=False)
    monkeypatch.setattr(app.state, "coordinator", coordinator, raising=False)

    async def check():
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await asyncio.wait_for(client.post("/api/interpolate", params={
                    "param": "tair_2m", "start": "2026-01-01", "end": "2026-01-02",
                }), timeout=1)
                assert response.status_code == 202
                assert response.json()["status"] == "queued"
                request_id = response.json()["request_id"]
                assert await asyncio.to_thread(preparing.wait, 1)
                assert coordinator.get_status(request_id).finished_at is None
                assert (await asyncio.wait_for(client.get("/api/health"), 1)).status_code == 200

                release_preparation.set()
                assert await asyncio.to_thread(interpolating.wait, 1)
                assert (await asyncio.wait_for(client.get("/api/health"), 1)).status_code == 200

                shutdown = asyncio.create_task(coordinator.shutdown())
                await asyncio.sleep(0)
                assert not shutdown.done()
                release_interpolation.set()
                await asyncio.wait_for(shutdown, 2)
                state = coordinator.get_status(request_id)
                assert state.finished_at is not None
                if fail:
                    assert state.status == "failed"
                    assert state.error == "interpolation failed"
                else:
                    assert state.n_jobs == 0
                    assert state.error is None
        finally:
            release_preparation.set()
            release_interpolation.set()
            await coordinator.shutdown()

    asyncio.run(check())
