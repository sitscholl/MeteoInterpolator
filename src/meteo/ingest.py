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
from ..database.models import DATABASE_VARIABLES
from ..domain.dem import DEM
from ..utils import to_utc_timestamp, validate_station_coordinates
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

    def _validate_start(self, start: datetime | None):
        if start is None:
            return start
        if start.tzinfo is None:
            raise ValueError("Start time must be timezone aware")
        return pd.Timestamp(start).tz_convert(ZoneInfo('UTC')).floor(_INGESTION_FREQ)

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
        self, last_updated_at: datetime | None,
    ) -> pd.Timestamp:
        default_start = to_utc_timestamp(self.default_start_date, raise_missing_timezone=True)
        default_start = default_start.floor(_INGESTION_FREQ)

        if last_updated_at is None:
            return default_start

        return (to_utc_timestamp(last_updated_at) - pd.Timedelta(days=self.overlap_days)).floor(_INGESTION_FREQ)

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
        start: datetime | None = None,
        end: datetime | None = None,
        session: BaseMeteoHandler | None = None,
        persist_requested_end: bool = False,
    ) -> StationIngestResult:

        requested_start = None
        requested_end = None

        if not self.database.station_exists(station_id):
            raise ValueError(f"Cannot upsert station because station {station_id} does not exist in the database. Add a new station by modifying the station catalog file and restarting the application.")
        if self.database.get_stations(station_id)[0].status == 'inactive':
            logger.warning("Requesting data for inactive station %s", station_id)
        
        try:
            context = (
                self.loader
                if session is None
                else nullcontext(session)
            )

            async with context as meteo_loader:
                source_freq = meteo_loader.freq
                self._validate_frequency(source_freq, _INGESTION_FREQ)

                station_sensors = sorted(
                    await self._get_variables_for_station(
                        station_id, session=meteo_loader
                        )
                    )
                valid_sensors = [i for i in DATABASE_VARIABLES if i in station_sensors]

                requested_start = self._validate_start(start)
                requested_end = self._validate_end(end)

                if not valid_sensors:
                    logger.debug("Skipping station %s: no supported sensors available",station_id)
                    return StationIngestResult(
                        station_id=station_id,
                        status="skipped",
                        requested_start=requested_start,
                        requested_end=requested_end,
                    )

                if requested_start is None:
                    last_row = self.database.get_last_row(station_id)
                    requested_start = self._get_start_date(last_row.datetime if last_row else None)

                if requested_end <= requested_start:
                    raise ValueError(
                        "Requested end must be after requested start. "
                        f"Got {requested_end} vs {requested_start}"
                    )

                station = await meteo_loader.get_data(
                    station_id=station_id,
                    sensor_codes=valid_sensors,
                    start=requested_start,
                    end=requested_end,
                    target_timezone=ZoneInfo("UTC"),
                )

            if station.elevation is None and dem is None:
                raise ValueError(f"Station {station.id} has no elevation and no DEM provided. Cannot add data to database")
            
            if dem is not None:
                station = self._update_elevation(station, dem, overwrite=False)
            _ = validate_station_coordinates(station.y, station.x, station.elevation)

            if station.data is None:
                logger.warning("No data found for station %s. Skipping database insertion", station_id)
                if persist_requested_end:
                    self.database.log_ingestion_task(
                        station_id = station.id, requested_start = requested_start, requested_end = requested_end, status = "completed",
                    )
                return StationIngestResult(
                        station_id = station_id,
                        status = 'completed',
                        requested_start = requested_start,
                        requested_end = requested_end,
                        observations_written = 0,
                        error = None
                    )

            ##Resample to 1h resolution to reduce database size
            station = self.resampler.resample_station(
                station, target_freq = _INGESTION_FREQ, datetime_col = 'datetime', source_freq = source_freq,
                )
            
            with self.database.Session.begin() as db_session:
                
                written_rows = self.database.add_station_data(
                    id = station.id, data = station.data, drop_invalid_variables=True, db_session=db_session
                    )
                if persist_requested_end:
                    self.database.log_ingestion_task(
                        station_id = station.id, 
                        requested_start = requested_start, 
                        requested_end = requested_end, 
                        status = "completed",
                        db_session=db_session
                    )
                logger.debug("Upserted %s rows to database for station %s", written_rows, station.id)

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
        persist_requested_end: bool = False,
    ):
        ##Refresh all active stations
        requested_end = self._validate_end(end)     
        database_stations = self.database.get_stations()
        
        results = []
        async with self.loader as meteo_loader:
            for station in database_stations:
                try:
                
                    last_refresh_task = self.database.get_last_ingestion_task(station.id)
                    if last_refresh_task is not None:
                        last_refresh_end = to_utc_timestamp(last_refresh_task.requested_end)
                        last_refresh_end = last_refresh_end - pd.Timedelta(days=self.overlap_days)
                    else:
                        last_refresh_end = None

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
                        start = last_refresh_end,
                        end = requested_end, 
                        session = meteo_loader,
                        persist_requested_end=persist_requested_end
                    )

                results.append(ingest_result)

        refresh_result = DatabaseRefreshResult(results)
        logger.info(
            "Database refresh completed. Processed %s stations (skipped=%s, failed=%s) and wrote %s total rows to the database", 
            len(refresh_result.processed_stations), 
            refresh_result.n_skipped, 
            refresh_result.n_failed, 
            refresh_result.written_rows
            )

        return refresh_result
