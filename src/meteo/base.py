import asyncio
import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .types import Station

logger = logging.getLogger(__name__)

class BaseMeteoHandler(ABC):
    """
    Abstract base class for meteorological data handlers.
    
    This class defines the interface for retrieving, processing, and validating
    meteorological data from various sources.
    """

    registry: dict[str, type["BaseMeteoHandler"]] = {}
    def __init__(self) -> None:
        self._sensor_catalogue: dict[str, list[str]] = {}
        self._batch_lock = asyncio.Lock()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            BaseMeteoHandler.registry[cls.name()] = cls

    @classmethod
    def get_handler(cls, name: str) -> type["BaseMeteoHandler"]:
        handler = cls.registry.get(name)
        if handler is None:
            available = ", ".join(sorted(cls.registry)) or "none"
            raise ValueError(f"Unknown meteo handler '{name}'. Available: {available}")
        return handler

    @classmethod
    def create(cls, name: str, **kwargs) -> "BaseMeteoHandler":
        handler_cls = cls.get_handler(name)
        return handler_cls(**kwargs)

    @classmethod
    @abstractmethod
    def name(cls):
        pass

    @property
    @abstractmethod
    def freq(self):
        """Return frequency string for datetime frequency of provider measurements"""
        pass

    @property
    @abstractmethod
    def inclusive(self):
        """Return string indicating if query calls to the provider are left or right inclusive or both. Must be one of 'left', 'right' or 'both'."""
        pass

    @abstractmethod
    async def __aenter__(self):
        """Enter async context management."""
        return self

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context management."""
        pass

    @abstractmethod
    async def get_station_info(self, station_id: str | None) -> dict[str, Any]:
        """
        Query information for a given station from the source, 
        such as elevation, latitude or longitude.

        Args:
            station_id (str): The unique identifier for the station.

        Returns:
            dict: A dictionary containing station information such as elevation, latitude, and longitude.
        """
        pass

    @abstractmethod
    async def get_stations_for_sensors(self, sensors: str | list[str]) -> dict[str, list[str]]:
        """
        Query station ids that expose the requested sensor code(s).

        Args:
            sensors: A sensor code or list of sensor codes, using either provider
                codes or harmonized names supported by the handler.

        Returns:
            dict: A mapping from requested sensor code to station ids.
        """
        pass

    @abstractmethod
    async def get_raw_data(self, **kwargs) -> pd.DataFrame:
        """
        Query the raw data from the source.
        
        Args:
            **kwargs: Parameters for data retrieval
            
        Returns:
            Any: Raw data from the source
        """
        pass

    @abstractmethod
    def transform(self, raw_data: Any, target_timezone: ZoneInfo) -> pd.DataFrame:
        """
        Transform the raw data into a standardized format.
        
        Args:
            raw_data: Raw data to be transformed
            
        Returns:
            pd.DataFrame: Transformed data in standardized format
        """
        pass

    async def get_data(self, target_timezone: ZoneInfo, **kwargs) -> Station :
        """
        Run the complete data processing pipeline.
        
        This method orchestrates the entire process:
        1. Get raw data
        2. Transform it
        3. Validate it
        
        Args:
            **kwargs: Parameters for the pipeline
            
        Returns:
            Station: A station object with validated data
        """
        raw_data, metadata = await self.get_raw_data(**kwargs)
        transformed_data = self.transform(raw_data, target_timezone)
            
        return Station(
            id = metadata.get('id'),
            x = metadata.get('x'),
            y = metadata.get('y'),
            elevation = metadata.get('elevation'),
            crs = metadata.get('crs'),
            data = transformed_data,
        )

    async def get_data_for_stations(
        self,
        station_ids: str | Sequence[str],
        sensor_codes: str | Sequence[str],
        start: datetime,
        end: datetime,
        target_timezone: ZoneInfo,
    ) -> list[Station]:
        """Load data for configured stations that provide a requested sensor.
        """
        requested_stations = {str(station_id) for station_id in ([station_ids] if isinstance(station_ids, str) else station_ids)}
        requested_sensors = [sensor_codes] if isinstance(sensor_codes, str) else list(sensor_codes)
        if not requested_stations:
            raise ValueError("At least one station_id is required.")
        if not requested_sensors:
            raise ValueError("At least one sensor code is required.")

        max_concurrent_requests = getattr(self, "max_concurrent_requests", 3)
        if not isinstance(max_concurrent_requests, int) or max_concurrent_requests < 1:
            raise ValueError(
                "max_concurrent_requests must be an integer greater than 0. "
                f"Got {max_concurrent_requests}"
            )

        async with self._batch_lock:
            async with self as meteo_loader:
                uncached_sensors = [
                    sensor
                    for sensor in requested_sensors
                    if sensor not in self._sensor_catalogue
                ]
                if uncached_sensors:
                    new_catalogue_entries = await meteo_loader.get_stations_for_sensors(
                        uncached_sensors
                    )
                    if new_catalogue_entries:
                        self._sensor_catalogue.update(new_catalogue_entries)

                stations_with_sensors = {
                    station
                    for sensor, station_ids in self._sensor_catalogue.items()
                    if sensor in requested_sensors
                    for station in station_ids
                }

                valid_stations = sorted(stations_with_sensors.intersection(requested_stations))
                if not valid_stations:
                    raise ValueError(
                        f"Found no stations for sensors {requested_sensors} within the "
                        f"{len(requested_stations)} requested stations."
                    )

                logger.info("Requesting data for %s stations.", len(valid_stations))

                semaphore = asyncio.Semaphore(max_concurrent_requests)

                async def load_station(station_id: str):
                    async with semaphore:
                        return await meteo_loader.get_data(
                            station_id=station_id,
                            start=start,
                            end=end,
                            sensor_codes=requested_sensors,
                            target_timezone=target_timezone
                        )

                return await asyncio.gather(
                    *(load_station(station_id) for station_id in valid_stations)
                )
