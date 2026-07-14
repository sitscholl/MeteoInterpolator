from pathlib import Path
from dataclasses import dataclass
import yaml

import logging

from .aoi import AOI
from .domain.dem import load_dem
from .array.cache import CacheManager
from .meteo.base import BaseMeteoHandler
from .resample import MeteoResampler
from .validate.meteo import MeteoValidator
from .datagaps import Gapfiller
from .interpolate import BaseDistanceCalculator
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
       
        ## Cache
        cache_config = config.get("cache", {})
        cache_enabled = cache_config.get("enabled", True)
        self.cache_manager = CacheManager(cache_config.get("cache_dir", "data/cache")) if cache_enabled else None

        ## DEM
        dem_config = dict(config['dem'])
        self.dem = load_dem(**dem_config)
        logger.info(f"Initialized dem {self.dem}")

        ## AOI
        self.aoi = AOI.from_array(self.dem.data)
        logger.info(f'Initialized aoi with bounds {self.aoi.bounds}')

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

        ##Distance Calculator
        interpolation_config = config['interpolation']

        distance_config = interpolation_config.get('distance')
        if distance_config is None:
            self.distance_calculator = None
        else:
            distance_config = dict(distance_config)
            distance_handler = distance_config.pop("type")
            self.distance_calculator = BaseDistanceCalculator.create(
                distance_handler,
                cache_manager=self.cache_manager,
                **distance_config,
            )

        ## Interpolator
        self.interpolator = Interpolator.from_config(interpolation_config)

        cv_config = config.get("cross_validation")
        if cv_config is None or not cv_config.get("enabled", False):
            self.cross_validation_config = None
            logger.info("No cross-validation configuration provided. Cross-validation will be skipped")
        else:
            self.cross_validation_config = dict(cv_config)
            self.cross_validation_config.pop("enabled", None)

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
