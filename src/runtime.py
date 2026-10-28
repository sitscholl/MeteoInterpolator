import logging
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml
from pyproj import CRS

from .aoi import AOI
from .array.cache import CacheManager
from .config import MeteoDataConfig
from .datagaps import Gapfiller
from .domain.dem import load_dem
from .domain.schemas import _STATION_CATALOG_SCHEMA
from .interpolate import BaseDistanceCalculator, DistanceField, Interpolator
from .interpolate.cv import validate_cross_validation_options

logger = logging.getLogger(__name__)

def load_config_file(config_file: str | Path) -> dict:
    config_path = Path(config_file)
    if not config_path.exists():
        raise ValueError(f"Could not find config file at {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config

def load_station_catalog(file: str | Path, aoi: AOI, crs: CRS | str | int):
    
    file = Path(file)

    if not file.exists():
        raise ValueError(f"Cannot find station cataloge file at {file}. Please upload a valid csv file.")
    if not file.suffix == '.csv':
        raise ValueError(f"Cataloge file is not a csv file. Got {file}")

    catalog = pd.read_csv(file, dtype={'station_id': str})
    if catalog.empty:
        raise ValueError("No active stations found in station catalogue.")
    catalog = _STATION_CATALOG_SCHEMA.validate(catalog)

    n_active = sum(catalog['status'] == 'active')
    if n_active == 0:
        raise ValueError("No active stations found in station catalogue.")

    coords = gpd.points_from_xy(
        catalog["longitude"], catalog["latitude"]
    )
    station_gdf = gpd.GeoDataFrame(
        catalog, geometry = coords, crs = 4326
        )
    station_gdf = station_gdf.to_crs(crs)

    stations_in_dem = aoi.filter_bbox(station_gdf)
    stations_outside_dem = [i for i in station_gdf['station_id'].unique() if i not in stations_in_dem['station_id'].unique()]
    if len(stations_outside_dem) > 0:
        raise ValueError(f"All stations must be within the DEM. The following stations are outside the provided dem: {stations_outside_dem}")

    return stations_in_dem

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
class InterpolationStatus:
    enabled: bool
    reason: str | None = None

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

    def load_cached_distance_fields(self) -> DistanceField:
        if self.distance_calculator is None:
            raise ValueError("Distance fields required by configuration but no distance calculator provided")

        distance_catalog = self.station_catalog.loc[self.station_catalog['station_id'].isin(self.active_stations)]
        
        distance_catalog = distance_catalog.sort_values('station_id').to_crs(self.dem.crs)
        x_coords = distance_catalog.geometry.x.astype(float).tolist()
        y_coords = distance_catalog.geometry.y.astype(float).tolist()

        logger.info(
            "Preparing distance fields for %s source station(s).",
            len(distance_catalog),
        )

        return self.distance_calculator.load_cache(
            self.dem,
            x_coords,
            y_coords,
            distance_catalog.station_id,
        )

    async def initialize_runtime(self, config: dict):
       
        self.dem = None
        self.db = None

        try:

            ## Meteo Config
            self.meteo_data_config = MeteoDataConfig.model_validate(config['meteo_data'])

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
            self.meteo_loader = self.meteo_data_config.create_meteo_loader()
            logger.info('Initialized meteo loader.')

            ## Stations
            self.station_catalog = load_station_catalog(
                self.meteo_data_config.station_catalog, aoi = self.aoi, crs = CRS.from_user_input(self.dem.crs)
            )
            self.active_stations = self.station_catalog.loc[self.station_catalog['status'] == 'active', 'station_id'].tolist()
            logger.info("Initialized %s stations inside dem for interpolation", len(self.active_stations))

            ## Meteo resampler
            self.resampler = self.meteo_data_config.create_meteo_resampler()
            logger.info("Initialized meteo resampler with min coverage %s", self.resampler.min_coverage)

            ## Database
            self.db = self.meteo_data_config.create_database_handler()
            logger.info("Initialized database connection at %s", self.db.engine)
            self.db.sync_catalog(self.station_catalog.drop('geometry', axis = 1))

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
            distance_fields_required = self.cross_validation.enabled or self.interpolator.residual_model is not None
            self.distance_fields = None
            if distance_fields_required:
                if self.interpolator.residual_model is None:
                    raise ValueError("Distance field required but no residual model available.")

                self.distance_fields = self.load_cached_distance_fields()
                if self.distance_fields is None:
                    logger.warning("No cached distance fields available but required by configuration. Interpolation will be disabled")
                    
            ## Grid Writer
            self.output_dir = Path('results')
            try:
                self.output_dir.mkdir(exist_ok = True, parents=True)
            except Exception as e:
                logger.exception("Failed to initialize output directory")
                raise
            
            if distance_fields_required and self.distance_fields is None:
                self.interpolation_status = InterpolationStatus(enabled = False, reason='No distance fields available.')
            else:
                self.interpolation_status = InterpolationStatus(enabled = True)

            self.initialized = True

        except Exception as e:
            logger.exception("Error during runtime initialization")
            try:
                if self.dem is not None:
                    self.dem.data.close()
            finally:
                try:
                    if self.db is not None:
                        self.db.close()
                finally:
                    raise

    async def update_runtime(self, config_file: str | Path):
        self.config_file = Path(config_file)
        self.config = load_config_file(self.config_file)
        await self.initialize_runtime(self.config)

if __name__ == '__main__':
    import asyncio

    logging.basicConfig(level = logging.DEBUG, force = True)
    runtime = asyncio.run(RuntimeContext.from_config_file('config.example.yaml'))
