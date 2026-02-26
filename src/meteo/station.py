from dataclasses import dataclass
import logging
from typing import Optional, Tuple

import httpx
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

logger = logging.getLogger(__name__)

@dataclass
class Station:
    id: str
    x: float
    y: float
    crs: int
    elevation: Optional[float]
    data: pd.DataFrame

    def __post_init__(self):

        if self.id is None:
            raise ValueError("Station id cannot be None.")
        if self.x is None:
            raise ValueError("Station x-coordinate cannot be None.")
        if self.y is None:
            raise ValueError("Station y-coordinate cannot be None.")
        if self.crs is None:
            raise ValueError("Station crs cannot be None.")

        if self.crs != 4326:
            raise NotImplementedError(f"Station crs is {self.crs}. Only 4326 is implemented for now. Make sure the MeteoHandler returns coordinates in this crs.")

        if -90 > self.y or self.y > 90:
            raise ValueError("Latitude must be between -90 and 90")
        if -180 > self.x or self.x > 180:
            raise ValueError("Longitude must be between -180 and 180")

    @classmethod
    async def create(cls, id, x, y, data, crs, elevation: Optional[float] = None, client: Optional[httpx.AsyncClient] = None):
        if elevation is None:
            try:
                elevation = await cls.fetch_elevation(x, y, client=client)
            except Exception as e:
                logger.warning(f"Fetching elevation for station {id} failed with error: {e}")
        return cls(id = id, x = x, y = y, crs = crs, elevation = elevation, data = data)

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
    crs: int
    data: list[pd.DataFrame]

    def __post_init__(self):
        if not len(self.ids) == len(self.coords) == len(self.elevation) == len(self.data):
            raise ValueError("Length mismatch in MeteoData. Make sure all attributes have the same number of elements")

        if self.n_stations > 0:
            if self.crs is None:
                raise ValueError("MeteoData crs cannot be None when stations are present.")
            if self.crs != 4326:
                raise NotImplementedError(
                    f"MeteoData crs is {self.crs}. Only 4326 is implemented for now."
                )
        elif self.crs is None:
            self.crs = 4326

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
        if len(stations) == 0:
            return cls(ids=[], coords=[], elevation=[], data=[], crs=4326)

        ids = [str(st.id) for st in stations]
        coords = [(st.x, st.y) for st in stations]
        elevation = [st.elevation for st in stations]
        data = [st.data for st in stations]
        
        crs = set([st.crs for st in stations])
        if len(crs) > 1:
            raise ValueError(f"Cannot construct MeteoData from stations with different coordinate systems. Got {crs}")

        return cls(ids=ids, coords=coords, elevation=elevation, data=data, crs=list(crs)[0])

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

    def to_geodataframe(self):
        return gpd.GeoDataFrame(
            {'id': self.ids, "geometry": [Point(x[0], x[1]) for x in self.coords]},
            crs = self.crs
        )

    def to_dataframe(self, include_coords: bool = False) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for station_id, (x, y), elev, tbl in zip(self.ids, self.coords, self.elevation, self.data):
            if tbl is None or tbl.empty:
                continue
            df = tbl.copy()
            df["station_id"] = station_id
            df["elevation"] = elev
            if include_coords:
                df["x"] = x
                df["y"] = y
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            frames.append(df)

        if len(frames) == 0:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def get_station_data(self, station_id: str):
        if station_id not in self.ids:
            logger.warning(f"No data available for station {station_id}")
            return None
        station_idx = self.ids.index(station_id)
        return self.data[station_idx]

    def iter_samples(self, start, end, params: list[str], freq: str = "D"):
        if isinstance(params, str):
            params = [params]

        df = self.to_dataframe()
        if df.empty:
            return
        if "datetime" not in df.columns:
            raise ValueError("Missing 'datetime' column in MeteoData dataframes.")

        start_ts = pd.to_datetime(start, utc=True)
        end_ts = pd.to_datetime(end, utc=True)
        df = df[(df["datetime"] >= start_ts) & (df["datetime"] <= end_ts)]

        for interp_date in pd.date_range(start_ts, end_ts, freq=freq):
            ts = pd.to_datetime(interp_date, utc=True)
            subset = df[df["datetime"] == ts]
            for param in params:
                if param not in subset.columns:
                    continue
                series = subset[param].dropna()
                if series.empty:
                    logger.warning(f"No data found for parameter '{param}' on {ts.date()}")
                    yield (param, interp_date, np.array([]), np.array([]))
                    continue

                y = series.to_numpy(dtype=float)
                elevations = subset.loc[series.index, "elevation"].to_numpy(dtype=float)
                X = elevations.reshape(-1, 1)
                yield (param, interp_date, X, y)
