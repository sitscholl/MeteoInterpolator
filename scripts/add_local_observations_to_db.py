import argparse
import asyncio
import logging
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

from src.config import MeteoDataConfig
from src.domain.dem import load_dem
from src.meteo import MeteoIngestor
from src.runtime import load_config_file

logger = logging.getLogger(__name__)


# from dataclasses import dataclass
# @dataclass
# class Dummy:
#     observations_path = r"C:\Users\tscho\OneDrive - Scientific Network South Tyrol\Daten\StationData\Province\original/*.csv"
#     metadata_path = "data/province/stations.csv"
#     config = "config.example.yaml"
#     verbose = True

# args = Dummy()

async def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Add local station observations to the configured database."
    )
    parser.add_argument(
        "-f",
        "--observations_path",
        default=None,
        help='Override meteo_data.loader.observations_path; supports glob patterns.'
    )
    parser.add_argument(
        "-m",
        "--metadata_path",
        default=None,
        help="Override meteo_data.loader.station_metadata_path.",
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
    logger.info("Starting script to add local observations to db at %s", datetime.now().strftime("%H:%M:%S"))
    logger.info("=" * 50)

    config = load_config_file(args.config)
    meteo_config = MeteoDataConfig.from_config(config['meteo_data'])
    loader_config = dict(meteo_config.loader_config)
    loader_config['handler'] = 'local_file'
    loader_config.setdefault('observations_path', 'data/province/station_files/*.csv')
    loader_config.setdefault('station_metadata_path', 'data/province/stations.csv')
    loader_config.setdefault('timezone', 'UTC')
    if args.observations_path is not None:
        loader_config['observations_path'] = args.observations_path
    if args.metadata_path is not None:
        loader_config['station_metadata_path'] = args.metadata_path
    meteo_config = replace(meteo_config, loader_config=loader_config)
    db_handler = meteo_config.create_database_handler()

    dem = None
    try:
        local_loader = meteo_config.create_meteo_loader()
        resampler = meteo_config.create_meteo_resampler()
        dem = load_dem(**config['dem'])

        ingestor_config = dict(meteo_config.ingestor_config)
        ingestor_config.setdefault('default_start_date', datetime(2000, 1, 1, tzinfo=ZoneInfo('UTC')))
        ingestor = MeteoIngestor(
            loader = local_loader,
            database = db_handler,
            resampler = resampler,
            **ingestor_config,
        )

        station_info = await local_loader.get_station_info()

        results = []
        for st_id in station_info:
            refresh_result = await ingestor.upsert_station(
                station_id = st_id,
                dem = dem,
                update_station_metadata = True,
            )
            results.append(refresh_result)
        
        failed_results = [i for i in results if i.status == 'failed']
        if failed_results:
            logger.warning("Ingestion for %s stations failed; check logs", len(failed_results))

        skipped_results = [i for i in results if i.status == 'skipped']
        if skipped_results:
            logger.info("Skipped %s stations for insertion into database", len(skipped_results))
        
        logger.info("=" * 50)
        logger.info("Finished adding data to database at %s", datetime.now().strftime("%H:%M:%S"))
        logger.info("=" * 50)
    finally:
        try:
            db_handler.close()
        finally:
            if dem is not None:
                dem.data.close()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
