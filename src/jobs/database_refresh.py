import asyncio
import logging

from ..runtime import MeteoDataConfig
from ..meteo import MeteoIngestor

logger = logging.getLogger(__name__)

def run_database_refresh(config: MeteoDataConfig):
    # Runs in a scheduler worker thread.
    asyncio.run(_refresh_on_worker(config))

async def _refresh_on_worker(config: MeteoDataConfig):
    """
    Create new instances of loader, handler and ingestor and pass them to workers
    """
    try:
        meteo_loader = config.create_meteo_loader()
        db_handler = config.create_database_handler()
        resampler = config.create_meteo_resampler()

        ingestor = MeteoIngestor(
            loader = meteo_loader,
            database = db_handler,
            resampler = resampler,
            **config.ingestor_config
        )

        refresh_result = await ingestor.refresh_database(persist_requested_end=True)

        if refresh_result.n_failed > 0:
            logger.warning(f"Database refresh failed for {refresh_result.n_failed} stations.")
    finally:
        db_handler.close()