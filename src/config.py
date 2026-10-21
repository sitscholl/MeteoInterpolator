from dataclasses import dataclass

from .meteo import BaseMeteoHandler, MeteoResampler
from .database.db import InterpolationDB

@dataclass(frozen=True)
class MeteoDataConfig:
    loader_config: dict
    scheduler_config: dict
    ingestor_config: dict
    resampler_config: dict
    database_config: dict

    @classmethod
    def from_config(cls, config: dict):
        return cls(
            loader_config=config.get("loader", {}),
            scheduler_config=config.get("scheduler", {}),
            ingestor_config=config.get("database_ingestion", {}),
            resampler_config=config.get("resampling", {}),
            database_config=config.get("database", {}),
        )

    @property
    def configured_stations(self):
        return self.loader_config.get('stations')

    def create_meteo_loader(self) -> BaseMeteoHandler:
        loader_config = dict(self.loader_config) 
        handler_name = loader_config.pop('handler')
        return BaseMeteoHandler.create(handler_name, **loader_config)

    def create_database_handler(self) -> InterpolationDB:
        return InterpolationDB(**self.database_config)

    def create_meteo_resampler(self) -> MeteoResampler:
        return MeteoResampler(**self.resampler_config)