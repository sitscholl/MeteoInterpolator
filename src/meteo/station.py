from dataclasses import dataclass
import logging
from typing import Optional, Tuple

import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class Station:
    id: str
    x: float
    y: float
    elevation: Optional[float]
    data: pd.DataFrame

    def __post_init__(self):
        if -90 > self.y or self.y > 90:
            raise ValueError("Latitude must be between -90 and 90")
        if -180 > self.x or self.x > 180:
            raise ValueError("Longitude must be between -180 and 180")

    @classmethod
    async def create(cls, id, x, y, data, elevation: Optional[float] = None, client: Optional[httpx.AsyncClient] = None):
        if elevation is None:
            try:
                elevation = await cls.fetch_elevation(x, y, client=client)
            except Exception as e:
                logger.warning(f"Fetching elevation for station {id} failed with error: {e}")
        return cls(id = id, x = x, y = y, elevation = elevation, data = data)

    @staticmethod
    async def fetch_elevation(x: float, y: float, client: Optional[httpx.AsyncClient] = None) -> float:
        api_template = "https://api.opentopodata.org/v1/eudem25m?locations={lat},{lon}"
        url = api_template.format(lat=y, lon=x)

        if client is None:
            async with httpx.AsyncClient() as temp_client:
                response = await temp_client.get(url)
                response.raise_for_status()
                return response.json()["results"][0]["elevation"]

        response = await client.get(url)
        response.raise_for_status()
        return response.json()["results"][0]["elevation"]

@dataclass
class MeteoData:
    ids: list[str]
    coords: list[Tuple[float, float]]
    elevation: list[float]
    data: list[pd.DataFrame]

    def __post_init__(self):
        if not len(self.ids) == len(self.coords) == len(self.elevation) == len(self.data):
            raise ValueError("Length mismatch in MeteoData. Make sure all attributes have the same number of elements")

        for lst,nam in zip([self.ids, self.coords], ['ids', 'coords']):
            self._assert_unique(lst, name = nam)

        for tbl in self.data:
            if "datetime" in tbl.columns:
                tbl["datetime"] = pd.to_datetime(tbl["datetime"], utc = True)

    def __repr__(self):
        return "MeteoData"

    @classmethod
    def from_list(cls, lst: list[Station | None]):
        stations = [st for st in lst if st is not None]
        ids = [str(st.id) for st in stations]
        coords = [(st.x, st.y) for st in stations]
        elevation = [st.elevation for st in stations]
        data = [st.data for st in stations]
        return cls(ids=ids, coords=coords, elevation=elevation, data=data)

    @property
    def n_stations(self):
        return len(self.ids)

    @property
    def available_stations(self):
        return self.ids

    @staticmethod
    def _assert_unique(lst, name: str):
        if len(lst) != len(set(lst)):
            raise ValueError(f"Found duplicated elements for attribute {name} in MeteoData.")

    def get_parameter_data(self, parameter_names: list[str], idx_cols = ['datetime', 'station_id']):
        if isinstance(parameter_names, str):
            parameter_names = [parameter_names]

        parameter_list = []
        for tbl in self.data:
            _params = [i for i in parameter_names if i in tbl.columns]
            if len(_params) == 0:
                continue
            if not all([i in tbl.columns for i in idx_cols]):
                raise ValueError(f"Not all idx_cols found in table: Got {tbl.columns}")
            parameter_list.append(tbl[idx_cols + _params])

        if len(parameter_list) > 0:
            return pd.concat(parameter_list, ignore_index = True).set_index(idx_cols)
        else:
            logger.warning(f"Could find no stations with data for {parameter_names}")
            return None

    def get_xy_data(self, parameter: str, timestamp) -> tuple[np.ndarray, np.ndarray]:
        if parameter is None:
            raise ValueError("Parameter cannot be None")

        ts = pd.to_datetime(timestamp, utc = True)
        values = []
        elevations = []

        for elev, tbl in zip(self.elevation, self.data):
            if elev is None:
                continue
            if parameter not in tbl.columns:
                continue
            if "datetime" not in tbl.columns:
                raise ValueError(f"Missing 'datetime' column in station data. Got columns: {list(tbl.columns)}")

            row = tbl.loc[tbl["datetime"] == ts, parameter]
            if row.empty:
                continue
            row = row.dropna()
            if row.empty:
                continue

            values.append(row.mean())
            elevations.append(elev)

        if len(values) == 0:
            logger.warning(f"No data found for parameter '{parameter}' at {ts}")
            return np.array([]), np.array([])

        X = np.asarray(values, dtype=float).reshape(-1, 1)
        y = np.asarray(elevations, dtype=float)
        return X, y

    def get_station_data(self, station_id: str):
        if station_id not in self.ids:
            logger.warning(f"No data available for station {station_id}")
            return None
        station_idx = self.ids.index(station_id)
        return self.data[station_idx]

    def iter_samples(self, start, end, params: list[str], freq: str = 'D'):

        if isinstance(params, str):
            params = [params]

        for interp_date in pd.date_range(start, end, freq = freq):
            for param in params:
                X, y = self.get_xy_data(param, interp_date)
                yield (param, interp_date, X, y)
