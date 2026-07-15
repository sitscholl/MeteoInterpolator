from uuid import uuid4
from datetime import datetime
import asyncio
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray  # noqa: F401
from pyproj import CRS

import logging

from .runtime import RuntimeContext
from .domain.meteo_data import MeteoData

logger = logging.getLogger(__name__)

##Fixed parameters for now, make configurable later
_FREQ = 'D'
_MIN_SAMPLE_SIZE = 60

class InterpolationWorkflow:

    def __init__(self, runtime_context: RuntimeContext, target_points: xr.DataArray | pd.DataFrame | gpd.GeoDataFrame | None = None):
        self.id = uuid4()
        self.timestamp = None
        self.context = runtime_context
        self.target_points = target_points

        self._sensor_catalogue = {}

        logger.info("Initialized InterpolationWorkflow")

    def _validate_dates(self, start: datetime, end: datetime):
        if start >= end:
            raise ValueError("start date must be before end date")
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("Start and end time must both be timezone aware.")
        if start.tzinfo != end.tzinfo:
            raise ValueError(f"start and end date must have the same timezone. Got {start.tzinfo} vs {end.tzinfo}")
        
    @staticmethod
    def _output_var_name(param: str, suffix: str | None = None) -> str:
        parts = [str(param).strip()]
        if suffix is not None:
            parts.append(str(suffix).strip().strip("_"))
        return "_".join(part for part in parts if part)

    @classmethod
    def _prepare_grid_for_output(cls, interpolated_grid: xr.DataArray | xr.Dataset, param: str, interp_date, suffix: str | None = None):
        new_name = cls._output_var_name(param, suffix)
        if isinstance(interpolated_grid, xr.Dataset):
            data = interpolated_grid
            data_vars = list(data.data_vars)
            if len(data_vars) == 1 and data_vars[0] != new_name:
                old_name = next(iter(data.data_vars))
                data = data.rename({old_name: new_name})
            elif param in data.data_vars and param != new_name:
                if new_name in data.data_vars:
                    raise ValueError(f"Cannot rename output variable {param!r} to existing variable {new_name!r}.")
                data = data.rename({param: new_name})
        elif isinstance(interpolated_grid, xr.DataArray):
            data = interpolated_grid.rename(new_name)
        else:
            raise TypeError(f"Interpolated grid must be an xarray DataArray or Dataset. Got {type(interpolated_grid)}")

        if 'time' not in data.dims:
            data = data.expand_dims(time=[pd.Timestamp(interp_date)])
        else:
            data = data.assign_coords(time=[pd.Timestamp(interp_date)])
        return data

    def _prepare_target_points(
        self,
        target_points: xr.DataArray | pd.DataFrame | gpd.GeoDataFrame | None,
    ) -> xr.DataArray | pd.DataFrame | gpd.GeoDataFrame:
        if target_points is None:
            return self.context.dem.data

        dem_crs = self.context.dem.crs

        if isinstance(target_points, xr.DataArray):
            if target_points.rio.crs is None:
                target_points = target_points.rio.write_crs(dem_crs, inplace=False)
            if CRS.from_user_input(target_points.rio.crs) != CRS.from_user_input(dem_crs):
                raise ValueError(
                    "Target grid CRS must match DEM CRS. Reproject/resample the target grid before interpolation."
                )
            return target_points

        if isinstance(target_points, gpd.GeoDataFrame):
            if target_points.crs is None:
                raise ValueError("Target point GeoDataFrame must define a CRS.")
            points = target_points.to_crs(dem_crs).copy()
            if "station_id" not in points.columns:
                points["station_id"] = points.index.astype(str)
            points["x"] = points.geometry.x.astype(float)
            points["y"] = points.geometry.y.astype(float)
        elif isinstance(target_points, pd.DataFrame):
            points = target_points.copy()
            point_crs = points.attrs.get("crs")
            if point_crs is not None and CRS.from_user_input(point_crs) != CRS.from_user_input(dem_crs):
                raise ValueError(
                    "Plain DataFrame target_points cannot be reprojected. "
                    "Pass a GeoDataFrame with geometry and CRS, or provide x/y in the DEM CRS."
                )
            points.attrs["crs"] = dem_crs
            if "station_id" not in points.columns:
                points["station_id"] = points.index.astype(str)
        else:
            raise TypeError(
                "target_points must be None, an xarray DataArray, a pandas DataFrame, "
                f"or a GeoDataFrame. Got {type(target_points)}"
            )

        missing_xy = [col for col in ("x", "y") if col not in points.columns]
        if missing_xy:
            raise ValueError(f"Point target predictions require x/y columns. Missing: {missing_xy}")

        if "elevation" not in points.columns:
            x_indexer = xr.DataArray(points["x"].to_numpy(dtype=float), dims=("station_id",))
            y_indexer = xr.DataArray(points["y"].to_numpy(dtype=float), dims=("station_id",))
            points["elevation"] = self.context.dem.data.sel(
                x=x_indexer,
                y=y_indexer,
                method="nearest",
            ).to_numpy()

        points.attrs["crs"] = dem_crs
        return points

    def _check_station_sample_size(self, n_stations: int):
        if n_stations < self.context.interpolator.min_sample_size:
            raise ValueError(f"Only {n_stations} available, interpolation requires {self.context.interpolator.min_sample_size}")

    async def _load_meteo_data(
        self, stations: str | list[str], start, end, sensor_codes: str | list[str]
        ):
        if isinstance(stations, str):
            stations = [stations]      
        if isinstance(sensor_codes, str):
            sensor_codes = [sensor_codes]

        requested_stations = set(stations)
        uncached_sensors = [i for i in sensor_codes if i not in self._sensor_catalogue.keys()]

        async with self.context.meteo_loader as meteo_loader:

            if len(uncached_sensors) > 0:
                new_catalogue_entries = await meteo_loader.get_stations_for_sensors(uncached_sensors)
                if len(new_catalogue_entries) > 0:
                    self._sensor_catalogue.update(new_catalogue_entries)

            stations_with_sensors = {st for sn, i in self._sensor_catalogue.items() for st in i if sn in sensor_codes}

            valid_stations = sorted(stations_with_sensors.intersection(requested_stations))
            if len(valid_stations) == 0:
                raise ValueError(f"Found no stations for sensors {sensor_codes} within the {len(requested_stations)} requested stations.")
            self._check_station_sample_size(len(valid_stations))

            logger.info(f"Requesting data for {len(valid_stations)} stations.")
        
            semaphore = asyncio.Semaphore(3)
            async def load_station(st: str):
                async with semaphore:
                    return await meteo_loader.get_data(
                        station_id = st, 
                        start = start, 
                        end = end, 
                        sensor_codes = sensor_codes, 
                        )
            tasks = [asyncio.create_task(load_station(st)) for st in valid_stations]
            station_data = await asyncio.gather(*tasks)

        meteo_data = MeteoData.from_list(station_data)

        if meteo_data.n_stations == 0:
            raise ValueError("Could not load data for any station.")
        self._check_station_sample_size(meteo_data.n_stations)
        logger.info(f"Loaded data for {meteo_data.n_stations} stations.")

        return meteo_data

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

    async def run(
        self,
        param: str,
        start: datetime,
        end: datetime,
        target_points: xr.DataArray | pd.DataFrame | gpd.GeoDataFrame | None = None,
    ):       
        self.timestamp = datetime.now()
        self._validate_dates(start, end)
        prediction_target = self._prepare_target_points(
            self.target_points if target_points is None else target_points
        )
        target_crs = (
            prediction_target.rio.crs
            if isinstance(prediction_target, xr.DataArray)
            else prediction_target.attrs["crs"]
        )

        meteo_data = await self._load_meteo_data(self.context.station_ids, start, end, param)
        meteo_data = meteo_data.to_crs(target_crs)
        
        if self.context.gapfiller is not None:
            meteo_data = self.context.gapfiller.fill_gaps(meteo_data)

        meteo_data = self.context.resampler.resample_meteo_data(
            meteo_data,
            freq=_FREQ,
            min_sample_size=_MIN_SAMPLE_SIZE,
            datetime_col = 'datetime',
            groupby_cols = ['station_id']
        )

        ## Use DEM-derived station elevations for consistency with target elevation values.
        meteo_data = meteo_data.update_elevation(self.context.dem, overwrite=True)

        grid_writer = (
            self.context.grid_writer.initialize(param=param, start=start, end=end, freq=_FREQ)
            if self.context.grid_writer is not None
            else None
        )

        jobs = list(meteo_data.build_jobs(start, end, param))

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

            cv_result = self.context.cross_validator.cross_validate(
                self.context.interpolator,
                job,
                distance_fields=run_distance_fields,
            )

            lam_value = None
            if cv_result is not None:
                lam_value = cv_result.best_params.get("lambda")
                logger.info(
                    "Cross-validation selected parameters for %s: %s (%s=%s)",
                    job,
                    cv_result.best_params,
                    cv_result.select_by,
                    cv_result.best_score,
                )

            fitted = self.context.interpolator.fit(job)
            prediction_result = self.context.interpolator.predict(
                fitted,
                prediction_target,
                distance_fields=run_distance_fields,
                lam_value=lam_value,
            )
            
            for suffix, result_var in zip([None, 'vertical', 'residual'],[prediction_result.prediction, prediction_result.vertical_prediction, prediction_result.residual_prediction]):
                if isinstance(result_var, (xr.DataArray, xr.Dataset)):
                    output = self._prepare_grid_for_output(result_var, job.parameter, job.timestamp, suffix)
                    if grid_writer is not None:
                        grid_writer.write(output)

            if cv_result is not None and self.context.db is not None and hasattr(self.context.db, "store_cv_results"):
                self.context.db.store_cv_results(cv_result.fold_results, timestamp=job.timestamp)

            results.append(output)

        if len(results) == 0:
            raise ValueError(f"No interpolation results were produced for parameter {param} over period {start} - {end}.")

        return results


if __name__ == '__main__':

    logging.basicConfig(level = logging.DEBUG, force = True, format='[%(asctime)s] %(name)s - %(levelname)s : %(message)s')
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('rasterio').setLevel(logging.WARNING)

    async def test_workflow():
        runtime = await RuntimeContext.from_config_file('config.example.yaml')
        workflow = InterpolationWorkflow(runtime)
        start = pd.Timestamp("2026-01-25", tz=runtime.timezone)
        end = pd.Timestamp("2026-01-30", tz=runtime.timezone)
        await workflow.run(param='tair_2m', start=start, end=end)

    logger.info("="*50)
    logger.info('Starting Interpolation Workflow')
    logger.info("="*50)

    asyncio.run(test_workflow())

    logger.info("="*50)
    logger.info('Finished Interpolation Workflow')
    logger.info("="*50)
