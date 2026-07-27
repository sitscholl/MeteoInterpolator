from dataclasses import dataclass
import logging

import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray  # noqa: F401
from shapely.geometry import Point
from pyproj import CRS

from ..aoi import AOI
from ..domain.dem import DEM
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
        if not self.observations.empty:
            self.observations = _STATION_DATA_SCHEMA.validate(self.observations)
        if not self.stations.empty:
            self.stations = _OBSERVATION_POINTS_SCHEMA.validate(self.stations)

        stations_ids = pd.Index(self.stations.index.astype(str))
        observation_ids = pd.Index(self.observations['station_id'].astype(str).unique())

        missing_ids_observations = stations_ids.difference(observation_ids)
        if len(missing_ids_observations) > 0:
            raise ValueError(f"The following ids are in the stations table but not in the observations table: {missing_ids_observations}")

        missing_ids_stations = observation_ids.difference(stations_ids)
        if len(missing_ids_stations) > 0:
            raise ValueError(f"The following ids are in the observations table but not in the stations table: {missing_ids_stations}")

        if self.stations.crs is None:
            raise ValueError("Crs of stations table is None")

        if self.stations.index.duplicated().any():
            raise ValueError("Stations table contains duplicated entries")

    @classmethod
    def from_list(cls, lst: list[Station]):

        stations = [st for st in lst if st.data is not None]
        if not stations:
            empty_stations = gpd.GeoDataFrame(
                {"elevation": []},
                geometry=[],
                crs=4326,
            )
            empty_stations.index = pd.Index([], name="station_id")
            empty_observations = pd.DataFrame(columns=["datetime", "station_id", "tair_2m"])
            return cls(stations=empty_stations, observations=empty_observations)

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

    def filter_bbox(self, aoi: AOI, buffer_m: int | float | None = None) -> tuple["MeteoData", int]:
        stations = aoi.filter_bbox(self.stations, buffer_m=buffer_m)
        station_ids = stations.index.astype(str)
        observations = self.observations.loc[
            self.observations["station_id"].astype(str).isin(station_ids)
        ].copy()
        n_dropped = self.n_stations - len(stations)
        return type(self)(stations=stations, observations=observations), n_dropped

    def to_crs(self, crs: CRS | str | int) -> "MeteoData":

        target_crs = CRS.from_user_input(crs)
        if target_crs == CRS.from_user_input(self.stations.crs):
            return type(self)(stations=self.stations, observations=self.observations)

        return type(self)(stations=self.stations.to_crs(target_crs), observations=self.observations)

    def update_elevation(self, dem: DEM, overwrite: bool = False) -> "MeteoData":
        if not isinstance(dem, DEM):
            raise TypeError(f"dem must be a DEM. Got {type(dem)}")

        stations = self.stations.copy()
        if "elevation" not in stations.columns:
            stations["elevation"] = pd.NA

        fill_mask = pd.Series(overwrite, index=stations.index) if overwrite else stations["elevation"].isna()
        if not fill_mask.any():
            return type(self)(stations=stations, observations=self.observations)

        dem_data = dem.data
        projected = stations.loc[fill_mask].to_crs(dem.crs)
        x_coords = projected.geometry.x.astype(float)
        y_coords = projected.geometry.y.astype(float)

        x_values = dem_data.coords["x"].values
        y_values = dem_data.coords["y"].values
        outside = projected.index[
            ~(
                x_coords.between(min(x_values), max(x_values))
                & y_coords.between(min(y_values), max(y_values))
            )
        ].astype(str).tolist()
        if outside:
            raise ValueError(
                "Cannot update station elevation because station points are outside the DEM extent. "
                f"Check CRS and DEM extent for ids: {outside}"
            )

        x_indexer = xr.DataArray(x_coords.to_numpy(dtype=float), dims=("station_id",))
        y_indexer = xr.DataArray(y_coords.to_numpy(dtype=float), dims=("station_id",))
        sampled_elevation = dem_data.sel(x=x_indexer, y=y_indexer, method="nearest").to_numpy()
        stations.loc[fill_mask, "elevation"] = sampled_elevation

        return type(self)(stations=stations, observations=self.observations)

    def get_projected_station_coords(self, target: CRS | str | int | object) -> dict[str, tuple[float, float]]:
        target_crs = getattr(getattr(target, "rio", None), "crs", None) or target
        projected = self.stations.to_crs(CRS.from_user_input(target_crs))
        return {
            str(station_id): (float(row.geometry.x), float(row.geometry.y))
            for station_id, row in projected.iterrows()
        }

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

        station_attrs = self.stations.copy()
        station_attrs["station_id"] = station_attrs.index.astype(str)
        station_attrs["x"] = station_attrs.geometry.x.astype(float)
        station_attrs["y"] = station_attrs.geometry.y.astype(float)
        station_attrs = pd.DataFrame(station_attrs.drop(columns="geometry")).reset_index(drop=True)
        station_attrs = station_attrs[["station_id", "elevation", "x", "y"]]
        observations = observations.merge(station_attrs, on="station_id", how="left", validate="many_to_one")

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

            obs_station_ids = obs["station_id"].astype(str).tolist()
            obs_stations = self.stations.loc[obs_station_ids]
            job = InterpolationJob(
                timestamp = ts,
                parameter = param,
                observations = obs,
                training_points = obs_stations,
            )
            yield job
