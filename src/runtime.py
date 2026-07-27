import geopandas as gpd
from shapely.geometry import Point
from pyproj import CRS

from pathlib import Path
from dataclasses import dataclass, field
import yaml

import logging

from .aoi import AOI
from .domain.dem import load_dem
from .array.cache import CacheManager
from .meteo.base import BaseMeteoHandler
from .resample import MeteoResampler
from .datagaps import Gapfiller
from .interpolate import BaseDistanceCalculator
from .interpolate import Interpolator, CrossValidator
from .array.writer import GridWriter
from .database.db import InterpolationDB
from .domain.schemas import _OBSERVATION_POINTS_SCHEMA

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

        ## Meteo Loader
        meteo_data_config = dict(config['meteo_data'])
        handler_name = meteo_data_config.pop('handler')
        self.meteo_loader = BaseMeteoHandler.create(handler_name, **meteo_data_config)
        logger.info(f'Initialized {handler_name} meteo loader')

        ## Stations
        configured_stations = meteo_data_config.get('stations')
        self.station_catalog = await self.build_station_catalog(
            configured_stations, aoi = self.aoi, crs = CRS.from_user_input(self.dem.crs)
        )
        self.station_ids = self.station_catalog.index.astype(str).tolist()
        logger.info("Initialized %s stations inside dem for interpolation", len(self.station_ids))

        ## Meteo resampler
        resampler_config = config.get('resampling', {})
        self.resampler = MeteoResampler(**resampler_config)

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
        cv_config = config.get("cross_validation", {})
        self.cross_validator = CrossValidator(**cv_config)

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

        ## Database
        db_config = config.get('database')
        self.db = InterpolationDB(**db_config) if db_config is not None else None
        if db_config is None:
            logger.info("No database configuration provided. Validation scores will not be persisted")
        else:
            logger.info(f"Initialized database connection at {db_config['path']}")
        self.initialized = True

    async def update_runtime(self, config_file: str | Path):
        self.config_file = Path(config_file)
        self.config = load_config_file(self.config_file)
        await self.initialize_runtime(self.config)

if __name__ == '__main__':
    import asyncio

    logging.basicConfig(level = logging.DEBUG, force = True)
    runtime = asyncio.run(RuntimeContext.from_config_file('config.example.yaml'))
