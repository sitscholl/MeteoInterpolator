import asyncio

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
