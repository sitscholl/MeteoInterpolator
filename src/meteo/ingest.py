from dataclasses import dataclass
from datetime import datetime
from typing import Literal
import logging

import pandas as pd
import numpy as np
from pyproj import Transformer, CRS
from zoneinfo import ZoneInfo

from ..database.db import InterpolationDB
from ..domain.dem import DEM
from .base import BaseMeteoHandler
from .types import Station
from ..utils import validate_station_coordinates
from .resample import MeteoResampler

logger = logging.getLogger(__name__)

@dataclass
class StationIngestResult:
    station_id: str
    status: Literal["completed", "skipped", "failed"]
    requested_start: datetime
    requested_end: datetime
    observations_written: int = 0
    error: str | None = None

class MeteoIngestor:
    def __init__(
        self,
        handler: BaseMeteoHandler,
        database: InterpolationDB,
        resampler: MeteoResampler,
        overlap_days: int = 7,
        default_start_date: datetime = datetime(2000, 1, 1, tzinfo = ZoneInfo('UTC'))
    ):
        self.handler = handler
        self.database = database
        self.resampler = resampler
        self.overlap_days = overlap_days
        self.default_start_date = default_start_date

    def _update_elevation(self, station: Station, dem: DEM, overwrite: bool = False) -> Station:
        if not isinstance(dem, DEM):
            raise TypeError(f"dem must be a DEM. Got {type(dem)}")

        if station.elevation is not None and not overwrite:
            return station

        if station.crs is None or dem.crs is None:
            raise ValueError(f"Cannot update station elevation because both station and the dem need to have a valid crs. Station: {station.crs}, DEM: {dem.crs}")

        source_crs = CRS.from_user_input(station.crs)
        target_crs = CRS.from_user_input(dem.crs)
        if source_crs != target_crs:
            transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
            x,y = transformer.transform(station.x, station.y)
        else:
            x,y = station.x, station.y

        dem_data = dem.data
        x_values = dem_data.coords["x"].values
        y_values = dem_data.coords["y"].values
        
        outside = (
            (x < min(x_values) or x > max(x_values)) or
            (y < min(y_values) or y > max(y_values))
            )
  
        if outside:
            raise ValueError(
                "Cannot update station elevation because station %s is outside the DEM extent.", station.id
            )

        sampled_elevation = dem_data.sel(x=x, y=y, method="nearest").item()

        if not np.isfinite(sampled_elevation):
            raise ValueError(f"Invalid elevation sampled from DEM when updating elevation: {sampled_elevation}")

        station.elevation = sampled_elevation

        return station

    async def refresh_station(
        self,
        station_id: str,
        sensors: list[str],
        dem: DEM,
        end: datetime | None = None,
        update_station_metadata: bool = False,
        add_missing_variables: bool = True,
    ) -> StationIngestResult:
        if isinstance(sensors, str):
            sensors = [sensors]

        last_database_row = self.database.get_last_row(station_id)
        if last_database_row is not None:
            requested_start = last_database_row.datetime - pd.Timedelta(self.overlap_days)
        else:
            requested_start = self.default_start_date

        if end is not None:
            if end.tzinfo is None:
                raise ValueError("End time must be timezone aware")
            requested_end = pd.Timestamp(end).tz_convert(ZoneInfo('UTC'))
        else:
            requested_end = pd.Timestamp.now(tz = ZoneInfo('UTC'))
            
        try:
            if requested_end <= requested_start:
                raise ValueError(f"Requested end must be before requested start. Got {requested_end} vs {requested_start}")

            station = await self.loader.get_data(
                station_id = station_id,
                sensor_codes = sensors,
                start = requested_start, 
                end = requested_end,
                target_timezone = ZoneInfo('UTC'),
            )

            station = self._update_elevation(station, dem, overwrite=False)
            _ = validate_station_coordinates(station.y, station.x, station.elevation)

            if not self.database.station_exists(station_id):
                self.database.add_station(
                    id = station.id, latitude = station.y, longitude = station.x, elevation = station.elevation
                )
            elif update_station_metadata:
                self.database.update_station(
                    id = station.id, latitude = station.y, longitude = station.x, elevation = station.elevation, only_missing = False
                )

            station_variables = [i for i in station.data.columns if i not in ['datetime', 'station_id']]
            variables_not_in_db = [i for i in station_variables if not self.database.variable_exists(i)]

            if len(variables_not_in_db) > 0 and add_missing_variables:
                for var in variables_not_in_db:
                    ##TODO: Find way to handle units of variables
                    self.database.add_variable(name = var)

            ##Resample to 1h resolution to reduce database size
            ##TODO: Add resample_station function to resampler class
            station = self.resampler.resample_station(
                station, freq = 'H', datetime_col = 'datetime', source_freq = self.loader.freq, min_sample_size = 1,
                )
            
            self.database.add_station_data(id = station.id, data = station.data, ignore_missing_variables = True)

            return StationIngestResult(
                station_id = station.id,
                status = 'completed',
                requested_start = requested_start,
                requested_end = requested_end,
                observations_written = len(station.data),
                error = None
            )
        except Exception as e:
            logger.exception("Refreshing station failed")
            return StationIngestResult(
                station_id = station.id,
                status = 'failed',
                requested_start = requested_start,
                requested_end = requested_end,
                observations_written = 0,
                error = str(e)
            )

    async def refresh_database(
        self,
        dem: DEM,
        update_station_metadata: bool = False,
        add_missing_variables: bool = True,
    ):
        ##Query station_ids and sensor_codes form database and refresh them all
        pass