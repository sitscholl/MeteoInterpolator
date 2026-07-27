import pandas as pd
import asyncio

import logging

from src.runtime import RuntimeContext
from src.coordinator import InterpolationCoordinator, InterpolationRequest

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG, force = True, format='[%(asctime)s] %(name)s - %(levelname)s : %(message)s')
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('rasterio').setLevel(logging.WARNING)

    async def run_interpolation():
        runtime = await RuntimeContext.from_config_file('config.example.yaml')
        coordinator = InterpolationCoordinator(runtime)
        start = "2021-12-01"
        end = "2021-12-02"

        request = InterpolationRequest('tair_2m', start, end)

        await coordinator.run(request)

    logger.info("="*50)
    logger.info('Starting Interpolation Workflow')
    logger.info("="*50)

    asyncio.run(run_interpolation())

    logger.info("="*50)
    logger.info('Finished Interpolation Workflow')
    logger.info("="*50)