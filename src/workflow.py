from uuid import uuid4
from datetime import datetime
import asyncio

import logging

from .runtime import RuntimeContext
from .validate.meteo import MeteoValidator

logger = logging.getLogger(__name__)

class InterpolationWorkflow:

    def __init__(self, runtime_context: RuntimeContext):
        self.id = uuid4()
        self.timestamp = None
        self.context = runtime_context

        logger.info("Initialized InterpolationWorkflow")

    def _validate_context(self):
        pass

    async def run(self):
        self.timestamp = datetime.now()
        self._validate_context()

        if self.context.stations is None:
            async with self.context.meteo_loader as meteo_loader:
                stations = await meteo_loader.get_station_codes()
        else:
            stations = self.context.stations
        
        logger.info(f"Requesting data for {len(stations)} stations.")
        async with self.context.meteo_loader as meteo_loader:
            async def load_station(st: str):
                async with asyncio.Semaphore(3):
                    return await meteo_loader.get_data(
                        station_id = st, 
                        start = self.context.start, 
                        end = self.context.end, 
                        sensor_codes = self.context.parameters, 
                        validator = MeteoValidator()
                        )
            tasks = [asyncio.create_task(load_station(st)) for st in stations]
            station_data = await asyncio.gather(*tasks)

        ##TODO: Transform station_data list into a MeteoData object?
        station_data = [i for i in station_data if i is not None]
        if len(station_data) == 0:
            raise ValueError("Could not load data for any station.")
        logger.info(f"Loaded data for {len(station_data)} stations.")

        if self.context.gapfiller is not None:
            station_data = self.context.gapfiller.fill_gaps(station_data)

        interpolated_grid, cv_results = self.context.interpolator.interpolate(station_data, target = self.context.base_grid)

        if self.context.grid_writer is not None:
            self.context.grid_writer.write(interpolated_grid)

        if self.context.db is not None:
            self.context.db.store_cv_results(cv_results, workflow_id = self.id, timestamp = self.timestamp)

if __name__ == '__main__':

    logging.basicConfig(level = logging.DEBUG, force = True)

    async def test_workflow():
        runtime = RuntimeContext.from_config_file('config.example.yaml')
        workflow = InterpolationWorkflow(runtime)
        await workflow.run()

    logger.info("="*50)
    logger.info('Starting Interpolation Workflow')
    logger.info("="*50)

    asyncio.run(test_workflow())

    logger.info("="*50)
    logger.info('Finished Interpolation Workflow')
    logger.info("="*50)