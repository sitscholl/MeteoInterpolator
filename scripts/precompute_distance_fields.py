import argparse
import asyncio
import logging
from datetime import datetime

from src.coordinator import (
    DistanceFieldPrecomputeRequest,
    InterpolationCoordinator,
)
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
        "--stations",
        nargs="+",
        default=None,
        help="Optional station IDs to precompute. Defaults to all configured stations inside the DEM.",
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
    coordinator = InterpolationCoordinator(runtime)
    result = await coordinator.precompute_distance_fields(
        DistanceFieldPrecomputeRequest(station_ids=args.stations)
    )

    sizes = result.distance_fields.data.sizes if result.distance_fields.data is not None else {}
    logger.info(
        "Prepared distance fields for %s station(s) with sizes %s.",
        result.n_sources,
        dict(sizes),
    )

    logger.info("=" * 50)
    logger.info("Finished distance field precompute at %s", datetime.now().strftime("%H:%M:%S"))
    logger.info("=" * 50)


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
