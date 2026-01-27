from pathlib import Path
from dataclasses import dataclass
import yaml

from .aoi import AOI
from .array.base_grid import BaseGrid
from .meteo.base import BaseMeteoHandler
from .datagaps import Gapfiller
from .interpolate import Interpolator, VerticalModel, ResidualModel, InterpolationRegions, CrossValidator
from .array import GridWriter
from .database.db import InterpolationDB


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
        aoi = AOI(**config['aoi'])

        ## Base Grid
        self.base_grid = BaseGrid(**config['base_grid'], aoi = aoi)

        ## Meteo Loader
        self.meteo_loader = BaseMeteoHandler(**config['meteo_input']).get_handler()

        ## Gapfiller
        gapfiller_config = config.get('gapfilling')
        self.gapfiller = Gapfiller(**gapfiller_config) if gapfiller_config is not None else None

        ## Interpolation Regions
        region_config = config.get('interpolation_regions')
        interpolation_regions = InterpolationRegions(**region_config) if region_config is not None else None

        ## Cross validation
        cv_config = config.get('cross_validation')
        cross_validator = CrossValidator(**cv_config) if cv_config is not None else None

        ## Interpolator
        self.interpolator = Interpolator(
            vertical_model = VerticalModel(**config['vertical_model']),
            residual_model = ResidualModel(**config['residual_model']),
            regions = interpolation_regions,
            cross_validator = cross_validator
        )

        ## Grid Writer
        output_config = config.get('output')
        self.grid_writer = GridWriter(**output_config) if output_config is not None else None

        ## Database
        db_config = config.get('database')
        self.db = InterpolationDB(**db_config) if db_config is not None else None

    def update_runtime(self, config_file: str | Path):
        self.config_file = Path(config_file)
        self.config = load_config_file(self.config_file)
        self.initialize_runtime(self.config)
