from uuid import uuid4
from datetime import datetime
import asyncio
from pandas import to_datetime

import logging

from .runtime import RuntimeContext
from .meteo.station import MeteoData

logger = logging.getLogger(__name__)

##Fixed parameters for now, make configurable later
_FREQ = 'D'
_MIN_SAMPLE_SIZE = 60

class InterpolationWorkflow:

    def __init__(self, runtime_context: RuntimeContext):
        self.context = runtime_context

        logger.info("Initialized InterpolationWorkflow")

    def _validate_dates(self, start: datetime, end: datetime):
        if start >= end:
            raise ValueError("start date must be before end date")
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("Start and end time must both be timezone aware.")
        if start.tzinfo != end.tzinfo:
            raise ValueError(f"start and end date must have the same timezone. Got {start.tzinfo} vs {end.tzinfo}")
        
    def _interpolate_param(self, meteo_data: MeteoData, param: str, start: datetime, end: datetime):

        results = []
        for date, X, y in meteo_data.iter_samples(start, end, param):
            logger.debug(f'Starting interpolation for {date}')

            interpolated_grid, cv_results = self.context.interpolator.interpolate(
                X, y, target_grid = self.context.base_grid.data
                )

            if self.context.grid_writer is not None:
                self.context.grid_writer.write(interpolated_grid)

            if self.context.db is not None:
                self.context.db.store_cv_results(cv_results)

            results.append(interpolated_grid)

        return results

    async def run(self, param: str, start: datetime, end: datetime):       
        self._validate_dates(start, end)

        if self.context.stations is None:
            async with self.context.meteo_loader as meteo_loader:
                stations = await meteo_loader.get_station_codes()
        else:
            stations = self.context.stations
        
        logger.info(f"Requesting data for {len(stations)} stations.")
        async with self.context.meteo_loader as meteo_loader:
            semaphore = asyncio.Semaphore(3)
            async def load_station(st: str):
                async with semaphore:
                    return await meteo_loader.get_data(
                        station_id = st, 
                        start = start, 
                        end = end, 
                        sensor_codes = self.context.parameters, 
                        validator = self.context.meteo_validator
                        )
            tasks = [asyncio.create_task(load_station(st)) for st in stations]
            station_data = await asyncio.gather(*tasks)

        meteo_data = MeteoData.from_list(station_data)
        if meteo_data.n_stations == 0:
            raise ValueError("Could not load data for any station.")

        if meteo_data.n_stations < 3:
            raise ValueError(f"Model fitting requires at least 3 stations. Got {meteo_data.n_stations}")
        logger.info(f"Loaded data for {meteo_data.n_stations} stations.")

        ## Check if any stations are within AOI, otherwise raise
        stations_within_aoi = self.context.aoi.filter_bbox(meteo_data.to_geodataframe())
        if self.context.require_stations_in_aoi and len(stations_within_aoi) == 0:
            raise ValueError("No stations are within defined bounds.")

        if self.context.gapfiller is not None:
            meteo_data = self.context.gapfiller.fill_gaps(meteo_data)

        meteo_data = self.context.resampler.resample_meteo_data(
            meteo_data,
            freq=_FREQ,
            min_sample_size=_MIN_SAMPLE_SIZE,
            datetime_col = 'datetime',
            groupby_cols = ['station_id']
        )

        logger.info(f"Interpolating parameter {param} over period {start} - {end} with frequency {_FREQ}")
        results = self._interpolate_param(meteo_data, param)

        return results


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
