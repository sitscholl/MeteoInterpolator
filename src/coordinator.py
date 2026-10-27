from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Literal
from uuid import uuid4

import geopandas as gpd
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr

from .api.schemas import InterpolationRequest, InterpolationSubmission
from .domain.meteo_data import MeteoData
from .execute.base import JobExecutor
from .execute.serial import SerialExecutor
from .interpolate import InterpolationJob
from .meteo import retrieve_stations
from .meteo.ingest import _INGESTION_FREQ
from .runtime import RuntimeContext
from .utils import get_date_format_from_freq
from .worker import InterpolationJobResult, process_interpolation_job
from .write import write_job_result

logger = logging.getLogger(__name__)


RunStatus = Literal["queued", "running", "completed", "failed"]


@dataclass(frozen=True)
class PreparedInterpolationRun:
    prediction_target: xr.DataArray
    jobs: list[InterpolationJob]


@dataclass(frozen=True)
class InterpolationRunState:
    request_id: str
    request: InterpolationRequest
    status: RunStatus
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    n_jobs: int | None = None
    n_completed: int | None = None
    n_skipped: int | None = None
    n_failed: int | None = None
    n_outputs: int | None = None
    error: str | None = None

@dataclass(frozen=True)
class JobSummary:
    timestamp: pd.Timestamp
    parameter: str
    status: Literal['completed', 'skipped', 'failed']
    output_path: str | Path | None
    error: str | None = None

@dataclass(frozen=True)
class InterpolationRunResult:
    request_id: str
    request: InterpolationRequest
    job_results: list[JobSummary]

    @property
    def status(self) -> RunStatus:
        return "completed" if any(result.status == "completed" for result in self.job_results) else "failed"

@dataclass
class InterpolationCoordinator:
    context: RuntimeContext
    executor: JobExecutor = field(default_factory=SerialExecutor)

    def __post_init__(self) -> None:
        self._runs: dict[str, InterpolationRunState] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    async def _prepare_request(self, request: InterpolationRequest) -> PreparedInterpolationRun:

        if request.freq not in ['D', 'h']:
            raise ValueError(f"freq must be one of 'D' or 'h'. Got {request.freq}")

        self._validate_dates(request.start, request.end)
        prediction_target = self.context.dem.data
        target_crs = prediction_target.rio.crs

        stations = retrieve_stations(
            db = self.context.db,
            station_ids=self.context.station_ids,
            sensor_codes=[request.param],
            start=request.start,
            end=request.end,
            target_timezone=request.tzinfo,
        )
        meteo_data = MeteoData.from_list(stations)

        if meteo_data.n_stations == 0:
            raise ValueError("Could not load data for any station.")
        logger.info("Loaded data for %s stations.", meteo_data.n_stations)

        meteo_data = meteo_data.to_crs(target_crs)

        if self.context.gapfiller is not None:
            meteo_data = self.context.gapfiller.fill_gaps(meteo_data)

        target_freq = request.freq
        meteo_data = self.context.resampler.resample_meteo_data(
            meteo_data,
            target_freq=target_freq,
            source_freq=_INGESTION_FREQ,
            datetime_col="datetime",
            groupby_cols=["station_id"],
        )
        meteo_data = meteo_data.update_elevation(self.context.dem, overwrite=True)

        jobs = list(meteo_data.build_jobs(request.start, request.end, request.param))

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
            prediction_target=prediction_target,
            jobs=jobs,
        )

    async def run(
        self,
        request: InterpolationRequest,
        request_id: str | None = None,
    ) -> InterpolationRunResult:

        if request.param != 'tair_2m':
            raise NotImplementedError(f"Interpolation is currently only implemented for parameter 'tair_2m'. Got {request.param}")

        request_id = request_id or str(uuid4())
        prepared = await self._prepare_request(request)

        summaries = []
        for result in self._submit_jobs(prepared):
            output_path = None
            if result.status == "completed" and result.prediction is not None and self.context.output_dir is not None:
                    try:
                        request_metadata = {
                            "param": request.param,
                            "start": request.start.isoformat(),
                            "end": request.end.isoformat(),
                            "timezone": request.timezone,
                            "freq": request.freq,
                            "dem_path": str(self.context.dem.path),
                            "dem_fingerprint": self.context.dem.fingerprint
                        }
                        output_path = write_job_result(
                            result,
                            root=self.context.output_dir,
                            request_id=request_id,
                            metadata=request_metadata,
                        )
                    except Exception as e:
                        logger.exception(f"Error writing job result for request {request_id}")
                        summaries.append(
                            JobSummary(
                                timestamp=result.job.timestamp,
                                parameter=result.job.parameter,
                                status="failed",
                                error=str(e),
                                output_path=None,
                            )
                        )
                        del result
                        continue
                
            summaries.append(
                JobSummary(
                    timestamp=result.job.timestamp,
                    parameter=result.job.parameter,
                    status=result.status,
                    error=result.error,
                    output_path=output_path,
                )
            )

            del result #free up memory immediately

        return InterpolationRunResult(
            request_id=request_id,
            request=request,
            job_results=summaries,
        )

    async def submit(self, request: InterpolationRequest) -> InterpolationSubmission:
        request_id = str(uuid4())
        self._runs[request_id] = InterpolationRunState(
            request_id=request_id,
            request=request,
            status="queued",
            submitted_at=datetime.now(),
        )
        self._tasks[request_id] = asyncio.create_task(
            self._run_submitted(request_id, request)
        )
        return InterpolationSubmission(request_id=request_id, status="queued")

    def _submit_jobs(self, prepared: PreparedInterpolationRun) -> Iterable[InterpolationJobResult]:
        run_job = partial(
            process_interpolation_job,
            interpolator=self.context.interpolator,
            cross_validation_config=self.context.cross_validation,
            prediction_target=prepared.prediction_target,
            distance_fields=self.context.distance_fields,
        )
        return self.executor.map(run_job, prepared.jobs)

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
            self._runs[request_id] = self._replace_run_state(
                request_id,
                status=result.status,
                finished_at=datetime.now(),
                n_jobs=len(result.job_results),
                n_completed=sum(1 for job_result in result.job_results if job_result.status == "completed"),
                n_skipped=sum(1 for job_result in result.job_results if job_result.status == "skipped"),
                n_failed=sum(1 for job_result in result.job_results if job_result.status == "failed"),
                n_outputs=len([i for i in result.job_results if i.output_path is not None]),
            )
        except Exception as exc:
            logger.exception("Interpolation request %s failed", request_id)
            self._runs[request_id] = self._replace_run_state(
                request_id,
                status="failed",
                finished_at=datetime.now(),
                error=str(exc),
            )
            return

    def _replace_run_state(self, request_id: str, **changes) -> InterpolationRunState:
        return replace(self.get_status(request_id), **changes)

    def _validate_dates(self, start: datetime, end: datetime) -> None:
        if start >= end:
            raise ValueError("start date must be before end date")
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("Start and end time must both be timezone aware.")
        if start.tzinfo != end.tzinfo:
            raise ValueError(
                f"start and end date must have the same timezone. Got {start.tzinfo} vs {end.tzinfo}"
            )

    async def shutdown(self) -> None:
        tasks = list(self._tasks.values())

        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()