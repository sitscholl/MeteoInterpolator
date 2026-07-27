from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from typing import Literal
from uuid import uuid4

import geopandas as gpd
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr
from pyproj import CRS

from .api.schemas import InterpolationRequest, InterpolationSubmission
from .array.writer import GridWriter
from .domain.meteo_data import MeteoData
from .execute.base import JobExecutor
from .execute.serial import SerialExecutor
from .interpolate import DistanceField, InterpolationJob
from .runtime import RuntimeContext
from .utils import get_date_format_from_freq
from .worker import InterpolationJobResult, process_interpolation_job

logger = logging.getLogger(__name__)


RunStatus = Literal["queued", "running", "completed", "failed"]
BackgroundOperation = Literal["interpolation", "distance_precompute"]


@dataclass(frozen=True)
class DistanceFieldPrecomputeRequest:
    station_ids: Sequence[str] | None = None


@dataclass(frozen=True)
class InterpolationRunResult:
    request_id: str
    request: InterpolationRequest
    job_results: list[InterpolationJobResult]
    outputs: list[xr.Dataset | xr.DataArray] = field(default_factory=list)

    @property
    def status(self) -> RunStatus:
        return "completed" if any(result.status == "completed" for result in self.job_results) else "failed"


@dataclass(frozen=True)
class InterpolationRunState:
    request_id: str
    request: InterpolationRequest | DistanceFieldPrecomputeRequest
    operation: BackgroundOperation
    status: RunStatus
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    n_jobs: int | None = None
    n_completed: int | None = None
    n_skipped: int | None = None
    n_failed: int | None = None
    n_outputs: int | None = None
    n_sources: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class DistanceFieldPrecomputeResult:
    request_id: str
    request: DistanceFieldPrecomputeRequest
    distance_fields: DistanceField
    station_ids: list[str]

    @property
    def n_sources(self) -> int:
        return len(self.station_ids)


@dataclass(frozen=True)
class PreparedInterpolationRun:
    request: InterpolationRequest
    prediction_target: xr.DataArray | pd.DataFrame | gpd.GeoDataFrame
    jobs: list[InterpolationJob]
    distance_fields: DistanceField | xr.DataArray | None
    grid_writer: GridWriter | None
    target_freq: str


@dataclass
class InterpolationCoordinator:
    context: RuntimeContext
    executor: JobExecutor = field(default_factory=SerialExecutor)

    def __post_init__(self) -> None:
        self._sensor_catalogue: dict[str, list[str]] = {}
        self._sensor_catalogue_lock = asyncio.Lock()
        self._runs: dict[str, InterpolationRunState] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    async def run(
        self,
        request: InterpolationRequest,
        *,
        request_id: str | None = None,
    ) -> InterpolationRunResult:

        ##TODO: InterpolationRequest.target_points data type changed. Implement support for new input structure
        if request.param != 'tair_2m':
            raise NotImplementedError(f"Interpolation is currently only implemented for parameter 'tair_2m'. Got {request.param}")

        request_id = request_id or str(uuid4())
        prepared = await self._prepare_request(request)
        job_results = self._submit_jobs(prepared)

        outputs: list[xr.Dataset | xr.DataArray] = []
        for job_result in job_results:
            outputs.extend(self._finalize_job(job_result, prepared.grid_writer))

        if not any(result.status == "completed" for result in job_results):
            raise ValueError(
                "No interpolation results were produced for parameter "
                f"{request.param} over period {request.start} - {request.end}."
            )

        return InterpolationRunResult(
            request_id=request_id,
            request=request,
            job_results=job_results,
            outputs=outputs,
        )

    async def submit(self, request: InterpolationRequest) -> InterpolationSubmission:
        request_id = str(uuid4())
        self._runs[request_id] = InterpolationRunState(
            request_id=request_id,
            request=request,
            operation="interpolation",
            status="queued",
            submitted_at=datetime.now(),
        )
        self._tasks[request_id] = asyncio.create_task(
            self._run_submitted(request_id, request)
        )
        return InterpolationSubmission(request_id=request_id, status="queued")

    async def precompute_distance_fields(
        self,
        request: DistanceFieldPrecomputeRequest | None = None,
        *,
        request_id: str | None = None,
    ) -> DistanceFieldPrecomputeResult:
        request = request or DistanceFieldPrecomputeRequest()
        request_id = request_id or str(uuid4())
        distance_fields, station_ids = self._calculate_stable_distance_fields(
            request.station_ids,
            require_cache=True,
        )
        return DistanceFieldPrecomputeResult(
            request_id=request_id,
            request=request,
            distance_fields=distance_fields,
            station_ids=station_ids,
        )

    async def submit_distance_precompute(
        self,
        request: DistanceFieldPrecomputeRequest | None = None,
    ) -> InterpolationSubmission:
        request = request or DistanceFieldPrecomputeRequest()
        request_id = str(uuid4())
        self._runs[request_id] = InterpolationRunState(
            request_id=request_id,
            request=request,
            operation="distance_precompute",
            status="queued",
            submitted_at=datetime.now(),
        )
        self._tasks[request_id] = asyncio.create_task(
            self._run_submitted_distance_precompute(request_id, request)
        )
        return InterpolationSubmission(request_id=request_id, status="queued")

    def get_status(self, request_id: str) -> InterpolationRunState:
        try:
            return self._runs[request_id]
        except KeyError as exc:
            raise KeyError(f"Unknown interpolation request_id: {request_id}") from exc

    async def _run_submitted(self, request_id: str, request: InterpolationRequest) -> None:
        self._runs[request_id] = self._replace_run_state(
            request_id,
            status="running",
            started_at=datetime.now(),
        )
        try:
            result = await self.run(request, request_id=request_id)
        except Exception as exc:
            logger.exception("Interpolation request %s failed", request_id)
            self._runs[request_id] = self._replace_run_state(
                request_id,
                status="failed",
                finished_at=datetime.now(),
                error=str(exc),
            )
            return

        self._runs[request_id] = self._replace_run_state(
            request_id,
            status=result.status,
            finished_at=datetime.now(),
            n_jobs=len(result.job_results),
            n_completed=sum(1 for job_result in result.job_results if job_result.status == "completed"),
            n_skipped=sum(1 for job_result in result.job_results if job_result.status == "skipped"),
            n_failed=sum(1 for job_result in result.job_results if job_result.status == "failed"),
            n_outputs=len(result.outputs),
        )

    async def _run_submitted_distance_precompute(
        self,
        request_id: str,
        request: DistanceFieldPrecomputeRequest,
    ) -> None:
        self._runs[request_id] = self._replace_run_state(
            request_id,
            status="running",
            started_at=datetime.now(),
        )
        try:
            result = await self.precompute_distance_fields(
                request,
                request_id=request_id,
            )
        except Exception as exc:
            logger.exception("Distance field precompute request %s failed", request_id)
            self._runs[request_id] = self._replace_run_state(
                request_id,
                status="failed",
                finished_at=datetime.now(),
                error=str(exc),
            )
            return

        self._runs[request_id] = self._replace_run_state(
            request_id,
            status="completed",
            finished_at=datetime.now(),
            n_sources=result.n_sources,
        )

    def _replace_run_state(self, request_id: str, **changes) -> InterpolationRunState:
        current = self.get_status(request_id)
        values = {
            "request_id": current.request_id,
            "request": current.request,
            "operation": current.operation,
            "status": current.status,
            "submitted_at": current.submitted_at,
            "started_at": current.started_at,
            "finished_at": current.finished_at,
            "n_jobs": current.n_jobs,
            "n_completed": current.n_completed,
            "n_skipped": current.n_skipped,
            "n_failed": current.n_failed,
            "n_outputs": current.n_outputs,
            "n_sources": current.n_sources,
            "error": current.error,
        }
        values.update(changes)
        return InterpolationRunState(**values)

    async def _prepare_request(self, request: InterpolationRequest) -> PreparedInterpolationRun:
        self._validate_dates(request.start, request.end)
        prediction_target = self._prepare_target_points(request.target_points)
        target_crs = (
            prediction_target.rio.crs
            if isinstance(prediction_target, xr.DataArray)
            else prediction_target.attrs["crs"]
        )

        meteo_data = await self._load_meteo_data(
            self.context.station_ids,
            request.start,
            request.end,
            request.param,
        )
        meteo_data = meteo_data.to_crs(target_crs)

        if self.context.gapfiller is not None:
            meteo_data = self.context.gapfiller.fill_gaps(meteo_data)

        target_freq = self.context.resampler.target_freq
        meteo_data = self.context.resampler.resample_meteo_data(
            meteo_data,
            freq=target_freq,
            source_freq=self.context.meteo_loader.freq,
            datetime_col="datetime",
            groupby_cols=["station_id"],
        )
        meteo_data = meteo_data.update_elevation(self.context.dem, overwrite=True)

        jobs = list(meteo_data.build_jobs(request.start, request.end, request.param))
        distance_fields = self._prepare_distance_fields(jobs)
        grid_writer = (
            self.context.grid_writer.initialize(
                param=request.param,
                start=request.start,
                end=request.end,
                freq=target_freq,
            )
            if self.context.grid_writer is not None
            else None
        )

        datefmt = get_date_format_from_freq(target_freq)
        logger.info(
            "Prepared %s interpolation job(s) for parameter %s over period %s - %s with frequency %s",
            len(jobs),
            request.param,
            request.start.strftime(datefmt),
            request.end.strftime(datefmt),
            target_freq,
        )

        return PreparedInterpolationRun(
            request=request,
            prediction_target=prediction_target,
            jobs=jobs,
            distance_fields=distance_fields,
            grid_writer=grid_writer,
            target_freq=target_freq,
        )

    def _submit_jobs(self, prepared: PreparedInterpolationRun) -> list[InterpolationJobResult]:
        run_job = partial(
            process_interpolation_job,
            interpolator=self.context.interpolator,
            cross_validator=self.context.cross_validator,
            prediction_target=prepared.prediction_target,
            distance_fields=prepared.distance_fields,
        )
        return list(self.executor.map(run_job, prepared.jobs))

    def _finalize_job(
        self,
        job_result: InterpolationJobResult,
        grid_writer: GridWriter | None,
    ) -> list[xr.Dataset | xr.DataArray]:
        if job_result.status != "completed" or job_result.prediction is None:
            return []

        prediction = job_result.prediction
        outputs: list[xr.Dataset | xr.DataArray] = []
        for suffix, result_var in (
            (None, prediction.prediction),
            ("vertical", prediction.vertical_prediction),
            ("residual", prediction.residual_prediction),
        ):
            if not isinstance(result_var, (xr.DataArray, xr.Dataset)):
                continue
            output = GridWriter.prepare_grid_for_output(
                result_var,
                job_result.job.parameter,
                job_result.job.timestamp,
                suffix,
            )
            if grid_writer is not None:
                grid_writer.write(output)
            outputs.append(output)

        if (
            job_result.cv_result is not None
            and self.context.db is not None
            and hasattr(self.context.db, "store_cv_results")
        ):
            self.context.db.store_cv_results(
                job_result.cv_result.fold_results,
                timestamp=job_result.job.timestamp,
            )

        return outputs

    def _validate_dates(self, start: datetime, end: datetime) -> None:
        if start >= end:
            raise ValueError("start date must be before end date")
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("Start and end time must both be timezone aware.")
        if start.tzinfo != end.tzinfo:
            raise ValueError(
                f"start and end date must have the same timezone. Got {start.tzinfo} vs {end.tzinfo}"
            )

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

    def _check_station_sample_size(self, n_stations: int) -> None:
        if n_stations < self.context.interpolator.min_sample_size:
            raise ValueError(
                f"Only {n_stations} available, interpolation requires {self.context.interpolator.min_sample_size}"
            )

    async def _load_meteo_data(
        self,
        stations: str | list[str],
        start,
        end,
        sensor_codes: str | list[str],
    ) -> MeteoData:
        if isinstance(stations, str):
            stations = [stations]
        if isinstance(sensor_codes, str):
            sensor_codes = [sensor_codes]

        requested_stations = set(stations)
        async with self._sensor_catalogue_lock:
            uncached_sensors = [
                sensor for sensor in sensor_codes if sensor not in self._sensor_catalogue
            ]

            async with self.context.meteo_loader as meteo_loader:
                if uncached_sensors:
                    new_catalogue_entries = await meteo_loader.get_stations_for_sensors(
                        uncached_sensors
                    )
                    if new_catalogue_entries:
                        self._sensor_catalogue.update(new_catalogue_entries)

                stations_with_sensors = {
                    station
                    for sensor, station_ids in self._sensor_catalogue.items()
                    for station in station_ids
                    if sensor in sensor_codes
                }

                valid_stations = sorted(stations_with_sensors.intersection(requested_stations))
                if not valid_stations:
                    raise ValueError(
                        f"Found no stations for sensors {sensor_codes} within the "
                        f"{len(requested_stations)} requested stations."
                    )
                self._check_station_sample_size(len(valid_stations))

                logger.info("Requesting data for %s stations.", len(valid_stations))

                semaphore = asyncio.Semaphore(3)

                async def load_station(station_id: str):
                    async with semaphore:
                        return await meteo_loader.get_data(
                            station_id=station_id,
                            start=start,
                            end=end,
                            sensor_codes=sensor_codes,
                        )

                tasks = [asyncio.create_task(load_station(st)) for st in valid_stations]
                station_data = await asyncio.gather(*tasks)

        meteo_data = MeteoData.from_list(station_data)

        if meteo_data.n_stations == 0:
            raise ValueError("Could not load data for any station.")
        self._check_station_sample_size(meteo_data.n_stations)
        logger.info("Loaded data for %s stations.", meteo_data.n_stations)

        return meteo_data

    def _prepare_distance_fields(
        self,
        jobs: list[InterpolationJob],
    ) -> DistanceField | xr.DataArray | None:
        if self.context.distance_calculator is None:
            return None
        if not jobs:
            return None
        distance_fields, _ = self._calculate_stable_distance_fields(
            self.context.station_ids
        )
        return distance_fields

    def _calculate_stable_distance_fields(
        self,
        station_ids: Sequence[str] | None = None,
        *,
        require_cache: bool = False,
    ) -> tuple[DistanceField, list[str]]:
        distance_calculator = self.context.distance_calculator
        if distance_calculator is None:
            raise ValueError("No distance calculator is configured.")
        if require_cache and self.context.cache_manager is None:
            raise ValueError("Distance field precomputation requires cache.enabled=true.")

        resolved_station_ids = self._resolve_distance_station_ids(station_ids)
        station_catalog = self.context.station_catalog.loc[resolved_station_ids].to_crs(
            self.context.dem.crs
        )
        x_coords = station_catalog.geometry.x.astype(float).tolist()
        y_coords = station_catalog.geometry.y.astype(float).tolist()

        logger.info(
            "Preparing distance fields for %s stable source station(s).",
            len(resolved_station_ids),
        )
        distance_fields = distance_calculator.calculate_fields(
            self.context.dem,
            x_coords,
            y_coords,
            resolved_station_ids,
        )
        return distance_fields, resolved_station_ids

    def _resolve_distance_station_ids(
        self,
        station_ids: Sequence[str] | None,
    ) -> list[str]:
        configured_station_ids = {
            str(station_id) for station_id in self.context.station_catalog.index
        }
        requested_station_ids = (
            configured_station_ids
            if station_ids is None
            else {str(station_id) for station_id in station_ids}
        )
        missing_station_ids = sorted(requested_station_ids - configured_station_ids)
        if missing_station_ids:
            raise ValueError(
                "Cannot prepare distance fields for unknown station ids: "
                f"{missing_station_ids}"
            )
        resolved_station_ids = sorted(requested_station_ids)
        if not resolved_station_ids:
            raise ValueError("At least one station id is required to prepare distance fields.")
        return resolved_station_ids
