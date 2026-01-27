from abc import ABC, abstractmethod
import inspect
import pandas as pd
from typing import Any, Dict

from .station import Station
from ..validate.meteo import MeteoValidator

class BaseMeteoHandler(ABC):
    """
    Abstract base class for meteorological data handlers.
    
    This class defines the interface for retrieving, processing, and validating
    meteorological data from various sources.
    """

    registry: dict[str, type["BaseMeteoHandler"]] = {}

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
    async def get_station_info(self, station_id: str | None) -> Dict[str, Any]:
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
    def transform(self, raw_data: Any) -> pd.DataFrame:
        """
        Transform the raw data into a standardized format.
        
        Args:
            raw_data: Raw data to be transformed
            
        Returns:
            pd.DataFrame: Transformed data in standardized format
        """
        pass

    async def get_data(self, validator: MeteoValidator, **kwargs) -> Station | None:
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
        transformed_data = self.transform(raw_data)

        if transformed_data is None:
            return None

        if validator is not None:
            transformed_data = validator.validate(transformed_data)
                    
        client = getattr(self, "_client", None)
        return await Station.create(
            id = metadata.get('id'),
            x = metadata.get('x'),
            y = metadata.get('y'),
            elevation = metadata.get('elevation'),
            data = transformed_data,
            client = client
        )
