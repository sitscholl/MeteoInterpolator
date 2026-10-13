import argparse
import asyncio
import logging
from datetime import datetime

from src.runtime import RuntimeContext

logger = logging.getLogger(__name__)


async def _main() -> None:
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

    runtime = await RuntimeContext.from_config_file(args.config)
    if runtime.cache_manager is None:
        raise ValueError("Distance field precomputation requires cache.enabled=true.")

    distance_fields = runtime.distance_fields
    if distance_fields is None:
        distance_fields = runtime.build_distance_fields()

    sizes = distance_fields.data.sizes if distance_fields.data is not None else {}
    logger.info(
        "Prepared distance fields for %s station(s) with sizes %s.",
        len(runtime.station_ids),
        dict(sizes),
    )

    logger.info("=" * 50)
    logger.info("Finished distance field precompute at %s", datetime.now().strftime("%H:%M:%S"))
    logger.info("=" * 50)


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
