from dataclasses import dataclass
import logging
from typing import Optional, Tuple

import httpx
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import rioxarray  # noqa: F401
from shapely.geometry import Point

from ..interpolate import InterpolationJob
from ..interpolate.interpolator import _REQUIRED_COLUMNS

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
                tbl["datetime"] = pd.to_datetime(tbl["datetime"])

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
                df["datetime"] = pd.to_datetime(df["datetime"])
            frames.append(df)

        if len(frames) == 0:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def _project_station_coordinates(self, target_grid: xr.DataArray) -> dict[str, tuple[float, float]]:
        target_crs = target_grid.rio.crs
        if target_crs is None:
            raise ValueError("Target grid must have an explicit CRS before interpolation jobs can be built.")

        target_epsg = target_crs.to_epsg()
        if target_epsg == self.crs:
            return {station_id: coords for station_id, coords in zip(self.ids, self.coords)}

        stations = gpd.GeoDataFrame(
            {"station_id": self.ids},
            geometry=[Point(x, y) for x, y in self.coords],
            crs=f"EPSG:{self.crs}",
        ).to_crs(target_crs)

        return {
            station_id: (float(geometry.x), float(geometry.y))
            for station_id, geometry in zip(stations["station_id"], stations.geometry)
        }

    def get_station_data(self, station_id: str):
        if station_id not in self.ids:
            logger.warning(f"No data available for station {station_id}")
            return None
        station_idx = self.ids.index(station_id)
        return self.data[station_idx]

    def build_jobs(self, start: pd.Timestamp, end: pd.Timestamp, param: str, target_grid: xr.DataArray):

        df = self.to_dataframe(include_coords = True)
        if df.empty:
            return
        if "datetime" not in df.columns:
            raise ValueError("Missing 'datetime' column in MeteoData dataframes.")
        if param not in df.columns:
            raise ValueError(f"{param} not found in MeteoData columns. Choose one of {df.columns}")
        missing_cols = [i for i in _REQUIRED_COLUMNS if i not in df.columns]
        if missing_cols:
            raise ValueError(f"Creating InterpolationJob requires the following columns to be present in the data {_REQUIRED_COLUMNS}. Got {df.columns}. Missing: {missing_cols}")

        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
        df = df[(df["datetime"] >= start_ts) & (df["datetime"] < end_ts)]
        projected_coords = self._project_station_coordinates(target_grid)
        df["x"] = df["station_id"].map(lambda station_id: projected_coords[station_id][0])
        df["y"] = df["station_id"].map(lambda station_id: projected_coords[station_id][1])

        for interp_date, subset in df.groupby('datetime'):
            ts = pd.to_datetime(interp_date)
            
            n_rows_before = len(subset)

            obs = subset.dropna(subset = [param, *_REQUIRED_COLUMNS])
            if obs.empty:
                logger.warning(f"No data found for parameter '{param}' and timestamp {ts}")
                continue

            n_rows_after = len(obs)
            if n_rows_after < n_rows_before:
                logger.warning(f"Dropped {n_rows_before - n_rows_after} rows with NaN values for parameter {param} on timestamp {ts}")

            job = InterpolationJob(
                timestamp = ts,
                parameter = param,
                observations = obs,
                target_grid = target_grid,
                crs = target_grid.rio.crs,
            )
            yield job
