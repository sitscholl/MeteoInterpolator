import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import geopandas as gpd
import yaml
from pyproj import CRS
from shapely.geometry import Point

from .aoi import AOI
from .array.cache import CacheManager
from .array.writer import GridWriter
from .database.db import InterpolationDB
from .datagaps import Gapfiller
from .domain.dem import load_dem
from .domain.schemas import _OBSERVATION_POINTS_SCHEMA
from .interpolate import BaseDistanceCalculator, DistanceField, Interpolator
from .interpolate.cv import validate_cross_validation_options
from .meteo import BaseMeteoHandler, MeteoIngestor, MeteoResampler

logger = logging.getLogger(__name__)

def load_config_file(config_file: str | Path) -> dict:
    config_path = Path(config_file)
    if not config_path.exists():
        raise ValueError(f"Could not find config file at {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config

@dataclass(frozen=True)
class CrossValidationConfig:
    enabled: bool = False
    param_grid: dict[str, list[float]] = field(default_factory=dict)
    n_splits: int | None = None
    min_relative_score_improvement: float | None = None
    improvement_latency_steps: int = 3
    select_by: str = 'rmse'

    def __post_init__(self):
        if self.enabled:
            validate_cross_validation_options(
                param_grid=self.param_grid,
                n_splits=self.n_splits,
                min_relative_score_improvement=self.min_relative_score_improvement,
                improvement_latency_steps=self.improvement_latency_steps,
                metric=self.select_by,
            )

@dataclass(frozen=True)
class MeteoDataConfig:
    loader_config: dict
    scheduler_config: dict
    ingestor_config: dict
    resampler_config: dict
    database_config: dict

    @classmethod
    def from_config(cls, config: dict):
        loader_config = config['meteo_data'].get('loader', {})
        scheduler_config = config['meteo_data'].get('scheduler', {})
        ingestor_config = config['meteo_data'].get('database_ingestion', {})
        resampler_config = config['meteo_Data'].get('resampling', {})
        database_config = config['meteo_data'].get('database', {})

        return cls(loader_config, scheduler_config, ingestor_config, resampler_config, database_config)

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

@dataclass
class RuntimeContext:
    config: dict
    config_file: str | Path | None = None
    initialized: bool = field(default=False, init=False)

    @classmethod
    async def from_config_file(cls, config_file: str | Path):
        config = load_config_file(config_file)
        runtime = cls(config=config, config_file=config_file)
        await runtime.initialize_runtime(config)
        return runtime

    def __post_init__(self):
        if self.config is None:
            raise ValueError("RuntimeContext requires a config dictionary")

    async def build_station_catalog(
        self, configured_stations: str | list[str] | None, aoi: AOI, crs: CRS | str | int
        ):
        async with self.meteo_loader as meteo_loader:
            all_stations = await meteo_loader.get_station_info()

        if configured_stations is None:
            stations = all_stations
        else:
            if isinstance(configured_stations, str):
                configured_stations = [configured_stations]
            stations = {st: info for st, info in all_stations.items() if st in configured_stations}

        gdf_data = list(stations.values())
        coords = [Point(i['x'], i['y']) for i in gdf_data]
        station_gdf = gpd.GeoDataFrame(gdf_data, geometry = coords, crs = 4326)
        station_gdf = station_gdf.to_crs(crs)
        station_gdf.rename(columns = {'id': 'station_id'}, inplace = True)

        stations_in_dem = aoi.filter_bbox(station_gdf)

        if len(stations_in_dem) == 0:
            raise ValueError("No stations are within supplied dem.")

        dropped_stations = [i for i in station_gdf['station_id'].unique() if i not in stations_in_dem['station_id'].unique()]
        if len(dropped_stations) > 0 and configured_stations is not None:
            logger.warning(f"The following stations are ignored because they are outside the provided dem: {dropped_stations}")
        
        stations_in_dem = stations_in_dem.set_index("station_id", drop=False)
        stations_in_dem = _OBSERVATION_POINTS_SCHEMA.validate(stations_in_dem)

        return stations_in_dem

    def build_distance_fields(self) -> DistanceField:
        if self.distance_calculator is None:
            raise ValueError("Distance fields required by configuration but no distance calculator provided")

        station_catalog = self.station_catalog.sort_index().to_crs(
            self.dem.crs
        )
        x_coords = station_catalog.geometry.x.astype(float).tolist()
        y_coords = station_catalog.geometry.y.astype(float).tolist()

        logger.info(
            "Preparing distance fields for %s source station(s).",
            len(station_catalog),
        )

        return self.distance_calculator.calculate_fields(
            self.dem,
            x_coords,
            y_coords,
            station_catalog.index,
        )

    async def initialize_runtime(self, config: dict):
       
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

        ## Meteo Config
        self.meteo_data_config = MeteoDataConfig.from_config(config['meteo_data'])

        ## Meteo Loader
        self.meteo_loader = self.meteo_data_config.create_meteo_loader()
        logger.info('Initialized meteo loader.')

        ## Stations
        configured_stations = self.meteo_data_config.configured_stations()
        self.station_catalog = await self.build_station_catalog(
            configured_stations, aoi = self.aoi, crs = CRS.from_user_input(self.dem.crs)
        )
        self.station_ids = self.station_catalog.index.astype(str).tolist()
        logger.info("Initialized %s stations inside dem for interpolation", len(self.station_ids))

        ## Meteo resampler
        self.resampler = self.meteo_data_config.create_meteo_resampler()
        logger.info("Initialized meteo resampler with min coverage %s", self.resampler.min_coverage)

        ## Database
        self.db = self.meteo_data_config.create_database_handler()
        logger.info("Initialized database connection at %s", self.db.engine)

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

        ## Cross Validation
        self.cross_validation = CrossValidationConfig(**config.get("cross_validation", {}))
        if self.cross_validation.enabled:
            if self.interpolator.residual_model is None:
                raise ValueError(
                    "Cross validation tunes lam_value for residual interpolation, "
                    "but no residual model is configured."
                )
            if self.distance_calculator is None:
                raise ValueError(
                    "Cross validation is enabled but no distance calculator is configured."
                )

        ## Distance fields
        self.distance_fields = None
        distance_fields_required = self.cross_validation.enabled or self.interpolator.residual_model is not None
        if distance_fields_required:
            self.distance_fields = self.build_distance_fields()

        ## Grid Writer
        output_config = config.get('output')
        if output_config is None:
            logger.info("No output configuration provided. Results will not be saved")
            self.grid_writer = None
        else:
            output_format = output_config['format']
            self.grid_writer = GridWriter.create(output_format, **output_config.get('options', {}))
            output_target = getattr(self.grid_writer, "root", getattr(self.grid_writer, "path", None))
            logger.info(f"Initialized grid writer with output format {output_format} pointing to {output_target}")

        self.initialized = True


    async def update_runtime(self, config_file: str | Path):
        self.config_file = Path(config_file)
        self.config = load_config_file(self.config_file)
        await self.initialize_runtime(self.config)

if __name__ == '__main__':
    import asyncio

    logging.basicConfig(level = logging.DEBUG, force = True)
    runtime = asyncio.run(RuntimeContext.from_config_file('config.example.yaml'))
