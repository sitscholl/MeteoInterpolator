import argparse
import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from src.database.db import InterpolationDB
from src.domain.dem import load_dem
from src.meteo import LocalFileMeteoHandler, MeteoIngestor, MeteoResampler
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
        default="data/province/station_files/*.csv",
        help='Path to station observation files, possibly include glob patterns'
    )
    parser.add_argument(
        "-m",
        "--metadata_path",
        default="data/province/stations.csv",
        help="Path to station metadata file.",
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
    local_loader = LocalFileMeteoHandler(
        station_metadata_path = args.metadata_path,
        observations_path = args.observations_path,
        timezone = "UTC",
    )
    db_handler = InterpolationDB(**config.get('database', {}))
    resampler = MeteoResampler(**config['resampling'])

    try:
        ingestor = MeteoIngestor(
            loader = local_loader,
            database = db_handler,
            resampler = resampler,
            default_start_date = datetime(2000, 1, 1, tzinfo = ZoneInfo('UTC')),
        )

        refresh_result = await ingestor.refresh_database(update_station_metadata=False, persist_requested_end=True)

        logger.info("=" * 50)
        logger.info("Finished database refresh at %s", datetime.now().strftime("%H:%M:%S"))
        logger.info("=" * 50)

    finally:
        db_handler.close()

def main() -> None:
    asyncio.run(_main())

if __name__ == "__main__":
    main()