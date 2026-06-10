from pathlib import Path
from dataclasses import dataclass
import yaml

import logging

from .aoi import AOI
from .array.base_grid import load_base_grid
from .array.cache import CacheManager
from .meteo.base import BaseMeteoHandler
from .resample import MeteoResampler
from .validate.meteo import MeteoValidator
from .datagaps import Gapfiller
from .interpolate import Interpolator
from .array.writer import GridWriter
from .database.db import InterpolationDB

logger = logging.getLogger(__name__)

def load_config_file(config_file: str | Path) -> dict:
    config_path = Path(config_file)
    if not config_path.exists():
        raise ValueError(f"Could not find config file at {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config

@dataclass
class RuntimeContext:
    config: dict
    config_file: str | Path | None = None

    @classmethod
    def from_config_file(cls, config_file: str | Path):
        config = load_config_file(config_file)
        return cls(config=config, config_file=config_file)

    def __post_init__(self):
        if self.config is None:
            raise ValueError("RuntimeContext requires a config dictionary")
        self.initialize_runtime(self.config)

    def initialize_runtime(self, config: dict):

        ## General
        general_config = config['general']            
        self.timezone = general_config['timezone']
        self.require_stations_in_aoi = general_config.get('require_stations_in_aoi', True)
       
        self.aoi = AOI(**config['aoi'])
        logger.info(f'Initialized aoi with bounds {self.aoi.bounds}')

        ## Cache
        cache_config = config.get("cache", {})
        cache_enabled = cache_config.get("enabled", True)
        self.cache_manager = CacheManager(cache_config.get("cache_dir", "data/cache")) if cache_enabled else None

        ## Base Grid
        base_grid_config = dict(config['base_grid'])
        self.base_grid = load_base_grid(**base_grid_config, aoi = self.aoi, cache_manager=self.cache_manager)
        logger.info(f"Initialized Base grid {self.base_grid}")

        ## Meteo Data
        meteo_data_config = dict(config['meteo_data'])
        stations = meteo_data_config.get('stations')
        if stations is None:
            self.stations = None
        elif isinstance(stations, (list, tuple)):
            self.stations = list(stations)
        else:
            self.stations = [stations]

        handler_name = meteo_data_config.pop('handler')
        self.meteo_loader = BaseMeteoHandler.create(handler_name, target_timezone = self.timezone, **meteo_data_config)
        logger.info(f'Initialized {handler_name} meteo loader')

        ## Meteo resampler
        resampler_config = config.get('resampling', {})
        self.resampler = MeteoResampler(**resampler_config)

        ## Meteo Validator
        self.meteo_validator = MeteoValidator(timezone = self.timezone)

        ## Gapfiller
        gapfiller_config = config.get('gapfilling')
        self.gapfiller = Gapfiller(**gapfiller_config) if gapfiller_config is not None else None
        if gapfiller_config is None:
            logger.info('No gapfiller configuration provided. Gaps will not be filled')
        else:
            logger.info("Gapfiller initialized")

        ## Interpolator
        self.interpolator = Interpolator.from_config(
            config['interpolation'],
            cache_manager=self.cache_manager,
        )

        ## Grid Writer
        output_config = config.get('output')
        if output_config is None:
            logger.info("No output configuration provided. Results will not be saved")
            self.grid_writer = None
        else:
            output_format = output_config['format']
            self.grid_writer = GridWriter.create(output_format, **output_config.get('options', {}))
            logger.info(f"Initialized grid writer with output format {output_format} pointing to {self.grid_writer.path}")

        ## Database
        db_config = config.get('database')
        self.db = InterpolationDB(**db_config) if db_config is not None else None
        if db_config is None:
            logger.info("No database configuration provided. Validation scores will not be persisted")
        else:
            logger.info(f"Initialized database connection at {db_config['path']}")

    def update_runtime(self, config_file: str | Path):
        self.config_file = Path(config_file)
        self.config = load_config_file(self.config_file)
        self.initialize_runtime(self.config)

if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG, force = True)
    runtime = RuntimeContext.from_config_file('config.example.yaml')
