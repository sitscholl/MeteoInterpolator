from pathlib import Path
from dataclasses import dataclass
import yaml
from pandas import to_datetime

import logging

from .aoi import AOI
from .array.base_grid import BaseGrid
from .meteo.base import BaseMeteoHandler
from .datagaps import Gapfiller
from .interpolate import Interpolator, VerticalModel, ResidualModel, InterpolationRegions, CrossValidator
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
        stations = general_config.get('stations')
        if stations is None:
            self.stations = None
        elif isinstance(stations, (list, tuple)):
            self.stations = list(stations)
        else:
            self.stations = [stations]
        self.timezone = general_config['timezone']
        self.start = self._localize_datetime(general_config['start'])
        self.end = self._localize_datetime(general_config['end'])
        parameters = general_config['parameters']
        if isinstance(parameters, (list, tuple)):
            self.parameters = list(parameters)
        else:
            self.parameters = [parameters]
        logger.info(f"Start initializing runtime context. General settings: Start = {self.start}, End = {self.end}, Parameters = {self.parameters}")

        self.aoi = AOI(**config['aoi'])
        logger.info(f'Initialized aoi with bounds {self.aoi.bounds}')

        ## Base Grid
        self.base_grid = BaseGrid(**config['base_grid'], aoi = self.aoi)
        logger.info(f"Initialized Base grid {self.base_grid}")

        ## Meteo Loader
        handler_config = dict(config['meteo_input'])
        handler_name = handler_config.pop('handler')
        self.meteo_loader = BaseMeteoHandler.create(handler_name, **handler_config)
        logger.info(f'Initialized {handler_name} meteo loader')

        ## Gapfiller
        gapfiller_config = config.get('gapfilling')
        self.gapfiller = Gapfiller(**gapfiller_config) if gapfiller_config is not None else None
        if gapfiller_config is None:
            logger.info('No gapfiller configuration provided. Gaps will not be filled')
        else:
            logger.info("Gapfiller initialized")

        ## Interpolation Regions
        region_config = config.get('interpolation_regions')
        interpolation_regions = InterpolationRegions(**region_config) if region_config is not None else None
        if region_config is None:
            logger.info("No interpolation regions specified.")
        else:
            logger.info('Interpolation regions initialized')

        ## Cross validation
        cv_config = config.get('cross_validation')
        cross_validator = CrossValidator(**cv_config) if cv_config is not None else None
        if cv_config is None:
            logger.info('No cross validation configuration provided. Cross validation will be skipped')
        else:
            logger.info('Cross validator initialized.')

        ## Vertical Model
        vertical_model = VerticalModel(**config.get('vertical_model', {}))
        logger.info("Vertical model initialized")

        ## Residual Model
        residual_config = config.get('residual_model')
        residual_model = ResidualModel(**residual_config) if residual_config is not None else None
        if residual_config is None:
            logger.info("No residual model configuration provided. Residuals will not be interpolated")
        else:
            logger.info('Residual Model initialized')

        ## Interpolator
        self.interpolator = Interpolator(
            vertical_model = vertical_model,
            residual_model = residual_model,
            regions = interpolation_regions,
            cross_validator = cross_validator
        )

        ## Grid Writer
        output_config = config.get('output')
        self.grid_writer = GridWriter(**output_config) if output_config is not None else None
        if output_config is None:
            logger.info("No output configuration provided. Results will not be saved")
        else:
            logger.info(f"Initialized grid writer pointing to {output_config['path']}")

        ## Database
        db_config = config.get('database')
        self.db = InterpolationDB(**db_config) if db_config is not None else None
        if db_config is None:
            logger.info("No database configuration provided. Validation scores will not be persisted")
        else:
            logger.info(f"Initialized database connection at {db_config['path']}")

    def _localize_datetime(self, value):
        ts = to_datetime(value, dayfirst = True)
        if ts.tzinfo is None:
            try:
                return ts.tz_localize(self.timezone)
            except Exception:
                return ts.tz_localize(self.timezone, ambiguous = False, nonexistent = "shift_forward")
        return ts.tz_convert(self.timezone)

    def update_runtime(self, config_file: str | Path):
        self.config_file = Path(config_file)
        self.config = load_config_file(self.config_file)
        self.initialize_runtime(self.config)

if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG, force = True)
    runtime = RuntimeContext.from_config_file('config.example.yaml')
