import asyncio
import logging

from ..config import MeteoDataConfig
from ..meteo import MeteoIngestor

logger = logging.getLogger(__name__)

def run_database_refresh(config: MeteoDataConfig):
    # Runs in a scheduler worker thread.
    return asyncio.run(_refresh_on_worker(config))

async def _refresh_on_worker(config: MeteoDataConfig):
    """
    Create new instances of loader, handler and run refresh on separate worker
    """
    db_handler = config.create_database_handler()
    try:
        meteo_loader = config.create_meteo_loader()
        resampler = config.create_meteo_resampler()

        ingestor = MeteoIngestor(
            loader = meteo_loader,
            database = db_handler,
            resampler = resampler,
            **config.ingestor_options
        )

        refresh_result = await ingestor.refresh_database(persist_requested_end=True)

        if refresh_result.n_failed > 0:
            logger.warning(f"Database refresh failed for {refresh_result.n_failed} stations.")

        return refresh_result
    finally:
        db_handler.close()