import argparse
import logging
from datetime import datetime

from pyproj import CRS

from src.aoi import AOI
from src.array.cache import CacheManager
from src.domain.dem import load_dem
from src.interpolate import BaseDistanceCalculator
from src.runtime import load_config_file, load_station_catalog

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute and cache distance fields for configured interpolation stations."
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        help="Path to configuration file.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Turn on verbose logging.",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        force=True,
        format="%(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("rasterio").setLevel(logging.WARNING)

    logger.info("=" * 50)
    logger.info("Starting distance field precompute at %s", datetime.now().strftime("%H:%M:%S"))
    logger.info("=" * 50)

    config = load_config_file(args.config)

    dem = load_dem(**config['dem'])
    aoi = AOI.from_array(dem.data)
    station_catalog = load_station_catalog(config['meteo_data']['station_catalog'], aoi = aoi, crs = CRS.from_user_input(dem.crs))

    cache_config = config.get("cache", {})
    cache_manager = CacheManager(cache_config.get("cache_dir", "data/cache"))

    distance_config = config['interpolation'].get('distance')
    if distance_config is None:
        raise ValueError('Distance configuration in yaml file required to calculate distance fields.')
    distance_config = dict(distance_config)
    distance_handler = distance_config.pop("type")
    distance_calculator = BaseDistanceCalculator.create(
        distance_handler,
        cache_manager=cache_manager,
        **distance_config,
    )

    distance_catalog = station_catalog.sort_values('station_id').to_crs(dem.crs)
    x_coords = distance_catalog.geometry.x.astype(float).tolist()
    y_coords = distance_catalog.geometry.y.astype(float).tolist()

    logger.info(
        "Preparing distance fields for %s source station(s).",
        len(distance_catalog),
    )

    distance_fields = distance_calculator.calculate_fields(
        dem,
        x_coords,
        y_coords,
        distance_catalog.station_id,
    )

    if distance_fields.data is None:
        raise ValueError("Error when calculating distance fields, data is None.")

    sizes = distance_fields.data.sizes
    logger.info(
        "Prepared distance fields for %s station(s) with sizes %s.",
        len(distance_fields.data.id.values),
        dict(sizes),
    )

    logger.info("=" * 50)
    logger.info("Finished distance field precompute at %s", datetime.now().strftime("%H:%M:%S"))
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
