import asyncio
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
import pytest

from src.meteo.province import ProvinceAPI


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _FakeResponse(self.payload)


def test_get_stations_for_sensors_filters_catalogue_by_requested_sensors():
    payload = [
        {"TYPE": "LT", "SCODE": "station-a"},
        {"TYPE": "LT", "SCODE": "station-a"},
        {"TYPE": "LT", "SCODE": "station-b"},
        {"TYPE": "N", "SCODE": "station-c"},
        {"TYPE": "WG", "SCODE": "station-d"},
    ]
    client = _FakeClient(payload)
    api = ProvinceAPI()
    api._client = client

    result = asyncio.run(api.get_stations_for_sensors(["tair_2m", "precipitation"]))

    assert result == {
        "tair_2m": ["station-a", "station-b"],
        "precipitation": ["station-c"],
    }
    assert len(client.calls) == 1
    assert client.calls[0][0] == api.sensors_url


@pytest.mark.parametrize("failure", ["http", "network", "json"])
def test_chunk_failure_raises_after_draining_queue_without_returning_partial_data(failure):
    api = ProvinceAPI(chunk_size_days=1, max_concurrent_requests=1, sleep_time=0)
    api.station_info = {"station-a": {"id": "station-a", "x": 11, "y": 46}}
    api.station_sensors = {"station-a": {"LT"}}
    calls = []

    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            if failure == "network":
                raise httpx.ConnectError("Connection failed", request=request)
            if failure == "http":
                return httpx.Response(503, json={"error": "Unavailable"})
            return httpx.Response(200, text="invalid json")
        return httpx.Response(200, json=[{"DATE": "2026-01-02T01:00:00CET", "VALUE": 2.0}])

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            api._client = client
            # Four chunks exceed the queue capacity of two, exercising worker survival.
            return await asyncio.wait_for(api.get_raw_data(
                "station-a", datetime(2026, 1, 1, tzinfo=UTC),
                datetime(2026, 1, 5, tzinfo=UTC), sensor_codes=["tair_2m"],
            ), timeout=5)

    with pytest.raises(RuntimeError, match="1 request\\(s\\) failed for station station-a") as caught:
        asyncio.run(run())
    assert len(calls) == 4
    assert caught.value.__cause__ is not None
    if failure == "http":
        assert isinstance(caught.value.__cause__, httpx.HTTPStatusError)
    elif failure == "network":
        assert isinstance(caught.value.__cause__, httpx.ConnectError)


def test_successful_empty_api_response_is_not_a_request_failure():
    api = ProvinceAPI(sleep_time=0)
    api.station_info = {"station-a": {"id": "station-a", "x": 11, "y": 46}}
    api.station_sensors = {"station-a": {"LT"}}

    async def run():
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=[]))
        async with httpx.AsyncClient(transport=transport) as client:
            api._client = client
            return await api.get_raw_data(
                "station-a", datetime(2026, 1, 1, tzinfo=UTC),
                datetime(2026, 1, 2, tzinfo=UTC), sensor_codes=["tair_2m"],
            )

    data, metadata = asyncio.run(run())
    assert data is None
    assert metadata["id"] == "station-a"


def test_transform_raises_on_invalid_datetime():
    raw = pd.DataFrame({
        "DATE": ["invalidCET"], "station_id": ["station-a"],
        "sensor": ["LT"], "VALUE": [2.0],
    })
    with pytest.raises(ValueError):
        ProvinceAPI().transform(raw, ZoneInfo("UTC"))


@pytest.mark.parametrize("with_temperature", [False, True])
def test_precipitation_sums_into_interval_end_bins_without_shifting_temperature(with_temperature):
    raw = pd.DataFrame({
        "DATE": [f"2026-01-01T00:{minute:02}:00CET" for minute in [0, 5, 10, 15]],
        "station_id": ["station-a"] * 4,
        "sensor": ["N"] * 4,
        "VALUE": [1.0, 2.0, 3.0, 4.0],
    })
    if with_temperature:
        raw = pd.concat([raw, pd.DataFrame({
            "DATE": ["2026-01-01T00:00:00CET"], "station_id": ["station-a"],
            "sensor": ["LT"], "VALUE": [5.0],
        })], ignore_index=True)
    result = ProvinceAPI().transform(raw, ZoneInfo("UTC"))
    assert result["datetime"].tolist() == pd.date_range(
        "2025-12-31 23:00", periods=3, freq="10min", tz="UTC"
    ).tolist()
    assert result["precipitation"].tolist() == [1.0, 5.0, 4.0]
    assert result["precipitation"].sum() == raw[raw["sensor"] == "N"]["VALUE"].sum()
    if with_temperature:
        assert result.loc[0, "tair_2m"] == 5.0
        assert result.loc[1:, "tair_2m"].isna().all()


def test_precipitation_handles_five_and_ten_minute_stations_independently():
    raw = pd.DataFrame({
        "DATE": [
            "2026-01-01T00:05:00CET", "2026-01-01T00:10:00CET",
            "2026-01-01T00:00:00CET", "2026-01-01T00:10:00CET",
        ],
        "station_id": ["five", "five", "ten", "ten"],
        "sensor": ["N"] * 4,
        "VALUE": [1.0, 2.0, 4.0, 5.0],
    })

    result = ProvinceAPI().transform(raw, ZoneInfo("Europe/Rome"))

    assert result.loc[result["station_id"] == "five", "precipitation"].tolist() == [3.0]
    assert result.loc[result["station_id"] == "ten", "precipitation"].tolist() == [4.0, 5.0]
    assert result.loc[result["station_id"] == "five", "datetime"].tolist() == [
        pd.Timestamp("2026-01-01 00:10", tz="Europe/Rome")
    ]


def test_precipitation_preserves_zero_and_partial_bins_but_not_missing_as_zero():
    raw = pd.DataFrame({
        "DATE": [
            "2026-01-01T00:05:00CET", "2026-01-01T00:10:00CET",
            "2026-01-01T00:15:00CET", "2026-01-01T00:20:00CET",
            "2026-01-01T00:25:00CET", "2026-01-01T00:30:00CET",
            "2026-01-01T00:50:00CET",
        ],
        "station_id": ["a"] * 7,
        "sensor": ["N"] * 7,
        "VALUE": [0.0, 0.0, 2.0, None, None, None, 3.0],
    })

    result = ProvinceAPI().transform(raw, ZoneInfo("Europe/Rome"))

    assert result["datetime"].dt.minute.tolist() == [10, 20, 30, 40, 50]
    assert result.loc[:1, "precipitation"].tolist() == [0.0, 2.0]
    assert result.loc[2:3, "precipitation"].isna().all()
    assert result.loc[4, "precipitation"] == 3.0


def test_precipitation_keeps_repeated_dst_hours_separate():
    raw = pd.DataFrame({
        "DATE": [
            "2026-10-25T02:05:00CEST", "2026-10-25T02:10:00CEST",
            "2026-10-25T02:05:00CET", "2026-10-25T02:10:00CET",
        ],
        "station_id": ["a"] * 4,
        "sensor": ["N"] * 4,
        "VALUE": [1.0, 2.0, 4.0, 5.0],
    })

    result = ProvinceAPI().transform(raw, ZoneInfo("Europe/Rome"))
    observed = result.dropna(subset=["precipitation"])

    assert observed["datetime"].dt.tz_convert("UTC").tolist() == [
        pd.Timestamp("2026-10-25 00:10", tz="UTC"),
        pd.Timestamp("2026-10-25 01:10", tz="UTC"),
    ]
    assert observed["precipitation"].tolist() == [3.0, 9.0]
    assert not result.duplicated(["station_id", "datetime"]).any()


def test_transform_without_precipitation_preserves_on_grid_temperature():
    raw = pd.DataFrame({
        "DATE": ["2026-01-01T00:00:00CET", "2026-01-01T00:10:00CET"],
        "station_id": ["a", "a"],
        "sensor": ["LT", "LT"],
        "VALUE": [5.0, 6.0],
    })

    result = ProvinceAPI().transform(raw, ZoneInfo("Europe/Rome"))

    assert result["datetime"].dt.minute.tolist() == [0, 10]
    assert result["tair_2m"].tolist() == [5.0, 6.0]
    assert "precipitation" not in result.columns
