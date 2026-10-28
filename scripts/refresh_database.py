import argparse
import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from src.config import MeteoDataConfig
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
        description="Update local database."
    )
    parser.add_argument(
        "-f",
        "--observations_path",
        default=None,
        help='Override meteo_data.loader.observations_path for a local-file loader.'
    )
    parser.add_argument(
        "-m",
        "--metadata_path",
        default=None,
        help="Override meteo_data.loader.station_metadata_path for a local-file loader.",
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
    logger.info("Starting script to refresh local db at %s", datetime.now().strftime("%H:%M:%S"))
    logger.info("=" * 50)

    config = load_config_file(args.config)
    meteo_options = dict(config['meteo_data'])
    if args.observations_path is not None or args.metadata_path is not None:
        loader_config = dict(meteo_options['loader'])
        if loader_config.get('handler') != 'local_file':
            raise ValueError('Observation and metadata path overrides require a local_file loader.')
        if args.observations_path is not None:
            loader_config['observations_path'] = args.observations_path
        if args.metadata_path is not None:
            loader_config['station_metadata_path'] = args.metadata_path
        meteo_options['loader'] = loader_config
    meteo_config = MeteoDataConfig.model_validate(meteo_options)
    db_handler = meteo_config.create_database_handler()

    try:
        loader = meteo_config.create_meteo_loader()
        resampler = meteo_config.create_meteo_resampler()
        ingestor_config = meteo_config.ingestor_options
        ingestor_config.setdefault('default_start_date', datetime(2000, 1, 1, tzinfo=ZoneInfo('UTC')))
        ingestor = MeteoIngestor(
            loader = loader,
            database = db_handler,
            resampler = resampler,
            **ingestor_config,
        )

        refresh_result = await ingestor.refresh_database(persist_requested_end=True)
        if refresh_result.n_failed:
            logger.warning("Database refresh failed for %s stations; check logs", refresh_result.n_failed)

        logger.info("=" * 50)
        logger.info("Finished database refresh at %s", datetime.now().strftime("%H:%M:%S"))
        logger.info("=" * 50)

    finally:
        db_handler.close()

def main() -> None:
    asyncio.run(_main())

if __name__ == "__main__":
    main()
