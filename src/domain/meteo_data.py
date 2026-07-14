from dataclasses import dataclass
import logging

import pandas as pd
import geopandas as gpd
import rioxarray  # noqa: F401
from shapely.geometry import Point
from pyproj import CRS

from ..meteo.base import Station
from ..domain.schemas import _STATION_DATA_SCHEMA, _OBSERVATION_POINTS_SCHEMA
from ..interpolate import InterpolationJob
from ..interpolate.interpolator import _REQUIRED_COLUMNS

logger = logging.getLogger(__name__)

@dataclass
class MeteoData:
    stations: gpd.GeoDataFrame
    observations: pd.DataFrame

    def __post_init__(self):
        _STATION_DATA_SCHEMA.validate(self.observations)
        _OBSERVATION_POINTS_SCHEMA.validate(self.stations)

        stations_ids = self.stations.index
        observation_ids = self.observations['station_id'].unique()

        missing_ids_observations = stations_ids.difference(observation_ids)
        if missing_ids_observations:
            raise ValueError(f"The following ids are in the stations table but not in the observations table: {missing_ids_observations}")

        missing_ids_stations = observation_ids.difference(stations_ids)
        if missing_ids_stations:
            raise ValueError(f"The following ids are in the observations table but not in the stations table: {missing_ids_stations}")

        if self.stations.crs is None:
            raise ValueError("Crs of stations table is None")

        if self.stations.index.duplicated().any():
            raise ValueError("Stations table contains duplicated entries")

    @classmethod
    def from_list(cls, lst: list[Station]):

        stations = [st for st in lst if st.data is not None]
        crs = set([st.crs for st in stations])
        if len(crs) > 1:
            raise ValueError(f"Cannot construct MeteoData from stations with different coordinate systems. Got {crs}")

        coords = [(st.x, st.y) for st in stations]
        stations_table = gpd.GeoDataFrame(
            {
                'station_id': [i.id for i in stations],
                'elevation': [i.elevation for i in stations],
            },
            crs = list(crs)[0],
            geometry = [Point(x, y) for x,y in coords]
        ).set_index('station_id')

        observations_table = pd.concat([st.data for st in stations], ignore_index=True)
        
        return cls(stations=stations_table, observations=observations_table)

    @property
    def n_stations(self):
        return len(self.stations)

    @property
    def available_stations(self):
        return self.stations.index.values

    def to_crs(self, crs: CRS | str | int) -> "MeteoData":

        target_crs = CRS.from_user_input(crs)
        if target_crs == CRS.from_user_input(self.stations.crs):
            return type(self)(stations=self.stations, observations=self.observations)

        return type(self)(stations=self.stations.to_crs(target_crs), observations=self.observations)

    def get_station_data(self, station_id: str):
        station_id = str(station_id)

        if station_id not in self.observations['station_id'].unique():
            logger.warning(f"No data available for station {station_id}")
            return pd.DataFrame(columns = self.observations.columns)

        return self.observations.loc[self.observations['station_id'] == station_id]

    def _get_projected_stations(self, crs: CRS) -> gpd.GeoDataFrame:
        return self.stations.to_crs(crs)

    def _get_observations(self, start: pd.Timestamp, end: pd.Timestamp, param: str):
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
        return self.observations.loc[
            (self.observations["datetime"] >= start_ts) & 
            (self.observations["datetime"] < end_ts),
            ['station_id', 'datetime', param]]

    def build_jobs(self, start: pd.Timestamp, end: pd.Timestamp, param: str):
        
        if param not in self.observations.columns:
            raise ValueError(f"{param} not found in MeteoData observations. Choose one of {self.observations.columns}")
        observations = self._get_observations(start = start, end = end, param = param)

        missing_cols = [i for i in _REQUIRED_COLUMNS if i not in observations.columns]
        if missing_cols:
            raise ValueError(f"Creating InterpolationJob requires the following columns to be present in the data {_REQUIRED_COLUMNS}. Got {observations.columns}. Missing: {missing_cols}")

        for interp_date, subset in observations.groupby('datetime'):
            ts = pd.to_datetime(interp_date)
            
            n_rows_before = len(subset)

            obs = subset.dropna(subset = [param, *_REQUIRED_COLUMNS])
            if obs.empty:
                logger.warning(f"No data found for parameter '{param}' and timestamp {ts}")
                continue

            n_rows_after = len(obs)
            if n_rows_after < n_rows_before:
                logger.warning(f"Dropped {n_rows_before - n_rows_after} rows with NaN values for parameter {param} on timestamp {ts}")

            obs_stations = self.stations.loc[self.stations.index.isin([obs['station_id'].unique()])]
            job = InterpolationJob(
                timestamp = ts,
                parameter = param,
                observations = obs,
                training_points = obs_stations,
            )
            yield job