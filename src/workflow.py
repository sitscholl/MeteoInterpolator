from uuid import uuid4
from datetime import datetime
import asyncio
import pandas as pd
import xarray as xr

import logging

from .runtime import RuntimeContext
from .meteo.types import MeteoData
from .interpolate import cross_validate

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

    def prepare_distance_fields(self, jobs, meteo_data):
        distance_calculator = self.context.distance_calculator
        if distance_calculator is not None:
            job_station_ids = sorted(
                {
                    str(station_id)
                    for job in jobs
                    for station_id in job.observations["station_id"].values
                }
            )
            if job_station_ids:
                projected_stations = meteo_data.get_projected_station_coords(self.context.dem.data)
                run_stations = {
                    station_id: projected_stations[station_id]
                    for station_id in job_station_ids
                }
                return distance_calculator.calculate_fields(
                    self.context.dem, 
                    [p[0] for p in run_stations.values()], 
                    [p[1] for p in run_stations.values()], 
                    list(run_stations.keys())
                )
            else:
                return None
        else:
            return None

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
        logger.info(f"Loaded data for {meteo_data.n_stations} stations.")

        ## Filter stations that are inside DEM
        meteo_data, n_dropped = meteo_data.filter_bbox(aoi = self.context.aoi)
        if meteo_data.n_stations == 0:
            raise ValueError("No stations are within supplied dem.")
        if n_dropped > 0:
            logger.warning(f"Dropped {n_dropped} stations outside dem")

        if meteo_data.n_stations < self.context.interpolator.min_sample_size:
            raise ValueError(f"Only {meteo_data.n_stations} available, interpolation requires {self.context.interpolator.min_sample_size}")

        if self.context.gapfiller is not None:
            meteo_data = self.context.gapfiller.fill_gaps(meteo_data)

        meteo_data = self.context.resampler.resample_meteo_data(
            meteo_data,
            freq=_FREQ,
            min_sample_size=_MIN_SAMPLE_SIZE,
            datetime_col = 'datetime',
            groupby_cols = ['station_id']
        )

        ## Get station elevation from dem
        # meteo_data.update_elevation(self.context.dem)

        grid_writer = (
            self.context.grid_writer.initialize(param=param, start=start, end=end, freq=_FREQ)
            if self.context.grid_writer is not None
            else None
        )

        jobs = list(meteo_data.build_jobs(start, end, param, dem = self.context.dem))

        run_distance_fields = self.prepare_distance_fields(jobs, meteo_data)

        ## Interpolate
        logger.info(f"Interpolating parameter {param} over period {start} - {end} with frequency {_FREQ}")
        results = []
        for job in jobs:

            logger.info('Starting interpolation job %s', job)

            if len(job.observations) < self.context.interpolator.min_sample_size:
                logger.warning(
                    "Skipping interpolation job %s because only %s station sample(s) are available and %s are required.",
                    job,
                    len(job.observations),
                    self.context.interpolator.min_sample_size,
                )
                continue

            lam_value = None
            cv_result = None
            cross_validation_config = getattr(self.context, "cross_validation_config", None)
            if cross_validation_config is not None:
                cv_config = dict(cross_validation_config)
                apply_best_params = cv_config.pop("apply_best_params", True)
                cv_result = cross_validate(
                    self.context.interpolator,
                    job,
                    distance_fields=run_distance_fields,
                    **cv_config,
                )
                if apply_best_params:
                    lam_value = cv_result.best_params.get("lambda")
                logger.info(
                    "Cross-validation selected parameters for %s: %s (%s=%s)",
                    job,
                    cv_result.best_params,
                    cv_result.select_by,
                    cv_result.best_score,
                )

            prediction = self.context.interpolator.fit(job).predict(
                distance_fields=run_distance_fields,
                lam_value=lam_value,
            )

            output_grid = self._prepare_grid_for_output(prediction, job.parameter, job.timestamp)
            if grid_writer is not None:
                grid_writer.write(output_grid)

            if cv_result is not None and self.context.db is not None and hasattr(self.context.db, "store_cv_results"):
                self.context.db.store_cv_results(cv_result.fold_results, timestamp=job.timestamp)

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
