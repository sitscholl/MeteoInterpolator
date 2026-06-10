from uuid import uuid4
from datetime import datetime
import asyncio
import pandas as pd
import xarray as xr

import logging

from .runtime import RuntimeContext
from .meteo.station import MeteoData

logger = logging.getLogger(__name__)

##Fixed parameters for now, make configurable later
_FREQ = 'D'
_MIN_SAMPLE_SIZE = 60

class InterpolationWorkflow:

    def __init__(self, runtime_context: RuntimeContext):
        self.id = uuid4()
        self.timestamp = None
        self.context = runtime_context

        logger.info("Initialized InterpolationWorkflow")

    def _validate_dates(self, start: datetime, end: datetime):
        if start >= end:
            raise ValueError("start date must be before end date")
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("Start and end time must both be timezone aware.")
        if start.tzinfo != end.tzinfo:
            raise ValueError(f"start and end date must have the same timezone. Got {start.tzinfo} vs {end.tzinfo}")
        
    @staticmethod
    def _prepare_grid_for_output(interpolated_grid: xr.DataArray | xr.Dataset, param: str, interp_date):
        if isinstance(interpolated_grid, xr.Dataset):
            data = interpolated_grid
            if len(data.data_vars) == 1 and param not in data.data_vars:
                old_name = next(iter(data.data_vars))
                data = data.rename({old_name: param})
        elif isinstance(interpolated_grid, xr.DataArray):
            data = interpolated_grid.rename(param)
        else:
            raise TypeError(f"Interpolated grid must be an xarray DataArray or Dataset. Got {type(interpolated_grid)}")

        if 'time' not in data.dims:
            data = data.expand_dims(time=[pd.Timestamp(interp_date)])
        else:
            data = data.assign_coords(time=[pd.Timestamp(interp_date)])
        return data

    async def run(self, param: str, start: datetime, end: datetime):       
        self.timestamp = datetime.now()
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
                        sensor_codes = [param], 
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
        grid_writer = (
            self.context.grid_writer.initialize(param=param, start=start, end=end, freq=_FREQ)
            if self.context.grid_writer is not None
            else None
        )

        ## Calculate distance fields
        distance_calculator = self.context.interpolator.distance_calculator
        if distance_calculator is not None:
            run_stations = meteo_data.get_projected_station_coords(self.context.base_grid)
            run_stations = dict(sorted(run_stations.items())) #sort by station id
            run_distance_fields = distance_calculator.calculate_fields(
                self.context.base_grid, 
                [p[0] for p in run_stations.values()], 
                [p[1] for p in run_stations.values()], 
                list(run_stations.keys())
            )
        else:
            run_distance_fields = None

        ## Interpolate
        results = []
        for job in meteo_data.build_jobs(start, end, param, base_grid = self.context.base_grid):

            logger.info('Starting interpolation job %s', job)

            interpolation_result = self.context.interpolator.interpolate(job, distance_fields = run_distance_fields)
            if interpolation_result is None:
                continue

            #todo: do this inside interpolator and directly return prediction with correct structure
            output_grid = self._prepare_grid_for_output(interpolation_result.prediction, job.parameter, job.timestamp)
            if grid_writer is not None:
                grid_writer.write(output_grid)

            if self.context.db is not None:
                self.context.db.store_cv_results(interpolation_result.cv_results, timestamp = job.timestamp)

            results.append(output_grid)

        if len(results) == 0:
            raise ValueError(f"No interpolation results were produced for parameter {param} over period {start} - {end}.")

        return results


if __name__ == '__main__':

    logging.basicConfig(level = logging.DEBUG, force = True)

    async def test_workflow():
        runtime = RuntimeContext.from_config_file('config.example.yaml')
        workflow = InterpolationWorkflow(runtime)
        start = pd.Timestamp("2026-01-25", tz=runtime.timezone)
        end = pd.Timestamp("2026-01-26", tz=runtime.timezone)
        await workflow.run(param='tair_2m', start=start, end=end)

    logger.info("="*50)
    logger.info('Starting Interpolation Workflow')
    logger.info("="*50)

    asyncio.run(test_workflow())

    logger.info("="*50)
    logger.info('Finished Interpolation Workflow')
    logger.info("="*50)
