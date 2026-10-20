import logging
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import Tick
from pyproj import CRS, Transformer

from ..database.db import InterpolationDB
from ..domain.dem import DEM
from ..utils import validate_station_coordinates
from .base import BaseMeteoHandler
from .resample import MeteoResampler
from .sensors import SENSORS
from .types import Station

logger = logging.getLogger(__name__)

##Do not change!! Setting this to daily produces wrong daily aggregates before interpolation!
_INGESTION_FREQ = 'h'

@dataclass
class StationIngestResult:
    station_id: str
    status: Literal["completed", "skipped", "failed"]
    requested_start: datetime | None
    requested_end: datetime | None
    observations_written: int = 0
    error: str | None = None

@dataclass
class DatabaseRefreshResult:
    station_results: list[StationIngestResult]

    @property
    def processed_stations(self):
        return [i.station_id for i in self.station_results]
    @property
    def n_completed(self):
        return len([i for i in self.station_results if i.status == 'completed'])
    @property
    def n_skipped(self):
        return len([i for i in self.station_results if i.status == 'skipped'])
    @property
    def n_failed(self):
        return len([i for i in self.station_results if i.status == 'failed'])
    @property
    def written_rows(self):
        return sum([i.observations_written for i in self.station_results])

class MeteoIngestor:
    def __init__(
        self,
        loader: BaseMeteoHandler,
        database: InterpolationDB,
        resampler: MeteoResampler,
        overlap_days: int = 7,
        default_start_date: datetime = datetime(2020, 1, 1, tzinfo = ZoneInfo('UTC'))
    ):

        if overlap_days is None or not np.isfinite(overlap_days) or overlap_days < 1:
            raise ValueError("Overlap days must be a positive number greater than 0")

        self.loader = loader
        self.database = database
        self.resampler = resampler
        self.overlap_days = overlap_days
        self.default_start_date = default_start_date

    @staticmethod
    def _validate_frequency(source_freq, target_freq):
        source = to_offset(source_freq)
        target = to_offset(target_freq)

        if not isinstance(target, Tick) or target.n <= 0:
            raise ValueError("Target frequency must be a positive, fixed duration.")

        if (
            not isinstance(source, Tick)
            or source.n <= 0
            or source.nanos > target.nanos
        ):
            raise ValueError(
                f"Cannot add data with source frequency {source_freq} to database. "
                f"Raw data must have intervals no longer than {target_freq}."
            )

    def _validate_end(self, end: datetime):
        if end is not None:
            if end.tzinfo is None:
                raise ValueError("End time must be timezone aware")
            requested_end = pd.Timestamp(end).tz_convert(ZoneInfo('UTC')).floor(_INGESTION_FREQ)
        else:
            requested_end = pd.Timestamp.now(tz = ZoneInfo('UTC')).floor(_INGESTION_FREQ)
        return requested_end

    def _update_elevation(self, station: Station, dem: DEM, overwrite: bool = False) -> Station:
        if station.elevation is not None and not overwrite:
            return station

        if not isinstance(dem, DEM):
            raise TypeError(f"dem must be a DEM. Got {type(dem)}")

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
            raise ValueError(f"Cannot update station elevation because station {station.id} is outside the DEM extent.")

        sampled_elevation = dem_data.sel(x=x, y=y, method="nearest").item()

        if not np.isfinite(sampled_elevation):
            raise ValueError(f"Invalid elevation sampled from DEM when updating elevation: {sampled_elevation}")

        station.elevation = sampled_elevation

        return station

    def _get_start_date(
        self,
        last_rows: dict | None,
    ) -> pd.Timestamp:
        default_start = pd.Timestamp(self.default_start_date)
        if default_start.tzinfo is None:
            raise ValueError("default_start_date must be timezone aware")
        default_start = default_start.tz_convert("UTC").floor(_INGESTION_FREQ)

        if not last_rows or any(
            row is None or row.datetime is None
            for row in last_rows.values()
        ):
            return default_start

        timestamps = []
        for row in last_rows.values():
            timestamp = pd.Timestamp(row.datetime)
            timestamp = (
                timestamp.tz_localize("UTC")
                if timestamp.tzinfo is None
                else timestamp.tz_convert("UTC")
            )
            timestamps.append(timestamp)

        return (min(timestamps) - pd.Timedelta(days=self.overlap_days)).floor(_INGESTION_FREQ)

    async def _get_variables_for_station(
        self,
        station_id: str,
        session: BaseMeteoHandler,
    ) -> set[str]:
        codes = await session.get_sensors_for_station(station_id)
        mapping = SENSORS.sensor_maps[session.provider_name]
        return {mapping[code] for code in codes if code in mapping}

    async def upsert_station(
        self,
        station_id: str,
        dem: DEM | None = None,
        end: datetime | None = None,
        sensors: list[str] | None = None,
        update_station_metadata: bool = False,
        add_missing_variables: bool = True,
        session: BaseMeteoHandler | None = None,
    ) -> StationIngestResult:
        if sensors is not None and isinstance(sensors, str):
            sensors = [sensors]

        requested_start = None
        requested_end = None
        try:
            context = (
                self.loader
                if session is None
                else nullcontext(session)
            )

            async with context as meteo_loader:
                source_freq = meteo_loader.freq
                self._validate_frequency(source_freq, _INGESTION_FREQ)

                if sensors is None:
                    sensors = sorted(
                        await self._get_variables_for_station(
                            station_id, session=meteo_loader
                        )
                    )

                requested_end = self._validate_end(end)

                if not sensors:
                    return StationIngestResult(
                        station_id=station_id,
                        status="skipped",
                        requested_start=None,
                        requested_end=requested_end,
                    )

                last_rows = self.database.get_last_row_per_variable(
                    station_id, sensors
                )
                requested_start = self._get_start_date(last_rows)

                if requested_end <= requested_start:
                    raise ValueError(
                        "Requested end must be after requested start. "
                        f"Got {requested_end} vs {requested_start}"
                    )

                station = await meteo_loader.get_data(
                    station_id=station_id,
                    sensor_codes=sensors,
                    start=requested_start,
                    end=requested_end,
                    target_timezone=ZoneInfo("UTC"),
                )

            if station.data is None:
                logger.warning("No data found for station %s. Skipping database insertion", station_id)
                return StationIngestResult(
                        station_id = station_id,
                        status = 'skipped',
                        requested_start = requested_start,
                        requested_end = requested_end,
                        observations_written = 0,
                        error = None
                    )

            if station.elevation is None and dem is None:
                raise ValueError(f"Station {station.id} has no elevation and no DEM provided. Cannot add data to database")
            
            if dem is not None:
                station = self._update_elevation(station, dem, overwrite=False)
            _ = validate_station_coordinates(station.y, station.x, station.elevation)

            if not self.database.station_exists(station_id):
                self.database.add_station(
                    id = station.id, latitude = station.y, longitude = station.x, elevation = station.elevation, name = station.name
                )
            elif update_station_metadata:
                self.database.update_station(
                    id = station.id, latitude = station.y, longitude = station.x, elevation = station.elevation, name = station.name, only_missing = False
                )

            ##Resample to 1h resolution to reduce database size
            station = self.resampler.resample_station(
                station, freq = _INGESTION_FREQ, datetime_col = 'datetime', source_freq = source_freq,
                )

            station_variables = [i for i in station.data.columns if i not in ['datetime', 'station_id']]
            variables_not_in_db = [i for i in station_variables if not self.database.variable_exists(i)]

            if len(variables_not_in_db) > 0 and add_missing_variables:
                for var in variables_not_in_db:
                    self.database.add_variable(name = var)
            
            written_rows = self.database.add_station_data(id = station.id, data = station.data, ignore_missing_variables = True)

            return StationIngestResult(
                station_id = station.id,
                status = 'completed',
                requested_start = requested_start,
                requested_end = requested_end,
                observations_written = written_rows,
                error = None
            )
        except Exception as e:
            logger.exception("Refreshing station failed")
            return StationIngestResult(
                station_id = station_id,
                status = 'failed',
                requested_start = requested_start,
                requested_end = requested_end,
                observations_written = 0,
                error = str(e)
            )

    async def refresh_database(
        self,
        end: datetime | None = None,
        update_station_metadata: bool = False,
        add_missing_variables: bool = False,
    ):
        ##Query station_ids and sensor_codes from database and refresh them all
        requested_end = self._validate_end(end)     
        database_stations = self.database.get_stations()
        database_variables = {var.name for var in self.database.get_variables()}
        
        results = []
        async with self.loader as meteo_loader:
            for station in database_stations:
                try:
                    station_variables = await self._get_variables_for_station(station.id, session = meteo_loader)
                    requested_variables = sorted(station_variables & database_variables) #drops derived variables, such as tair_2m_max...
                except Exception as e:
                    logger.exception(f"Database refresh failed for station {station.id}")
                    ingest_result = StationIngestResult(
                            station_id = station.id,
                            status = 'failed',
                            requested_start = None,
                            requested_end = requested_end,
                            observations_written = 0,
                            error = str(e)
                        )
                else:
                    ingest_result = await self.upsert_station(
                        station.id, 
                        end = requested_end, 
                        sensors = requested_variables, 
                        update_station_metadata=update_station_metadata, 
                        add_missing_variables=add_missing_variables,
                        session = meteo_loader
                    )
                results.append(ingest_result)

        refresh_result = DatabaseRefreshResult(results)
        logger.info(
            "Database refresh completed. Processed %s stations (skipped=%s, failed=%s) and wrote %s total rows to database", 
            len(refresh_result.processed_stations), 
            refresh_result.n_skipped, 
            refresh_result.n_failed, 
            refresh_result.written_rows
            )

        return refresh_result
