import asyncio
import datetime
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pandas import Timedelta

from ..coordinator import InterpolationCoordinator
from ..database.db import StationAlreadyExistsError
from ..jobs.database_refresh import run_database_refresh
from ..runtime import RuntimeContext
from .schemas import (
    DatabaseRefreshSubmission,
    InterpolationRequest,
    InterpolationSubmission,
    StationCreate,
    StationRead,
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load environment variables from .env file
    load_dotenv()

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    logger.info("Starting application")

    config_path = os.getenv('CONFIG_FILE')
    if config_path is None:
        raise ValueError("Missing environment value 'CONFIG_FILE'")
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Invalid configuration file {config_path}")

    runtime = await RuntimeContext.from_config_file(config_path)

    coordinator = None
    scheduler = None
    try:
        coordinator = InterpolationCoordinator(runtime)

        scheduler = BackgroundScheduler(timezone = 'UTC')
        scheduler.add_job(
            run_database_refresh,
            kwargs={'config': runtime.meteo_data_config},
            trigger = 'cron',
            timezone = 'UTC',
            id='database_refresh',
            max_instances=1,
            coalesce=True,
            **runtime.meteo_data_config.scheduler_options
        )
        scheduler.start()

        app.state.runtime = runtime
        app.state.coordinator = coordinator
        app.state.scheduler = scheduler

        yield
    
    finally:
        try:
            if scheduler is not None and scheduler.running:
                scheduler.pause()
                await asyncio.to_thread(scheduler.shutdown, wait=True)
        finally:
            try:
                if coordinator is not None:
                    await coordinator.shutdown()
            finally:
                try:
                    runtime.db.close()
                finally:
                    runtime.dem.data.close()


app = FastAPI(
    title="Meteorological Interpolation API",
    lifespan=lifespan,
)

@app.exception_handler(Exception)
async def handle_unexpected_error(
    request: Request,
    exc: Exception,
):
    logger.error(
        "Unhandled error during %s %s",
        request.method,
        request.url.path,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )

@app.get("/api/health")
async def health_check():
    """Lightweight liveness check for container health probes."""
    return {
        "status": "online"
    }

@app.get("/api/stations", response_model=list[StationRead])
def get_station_from_database(request: Request, station_id: str | None = None):
    
    db = request.app.state.runtime.db
    stations = db.get_stations(station_id)
    
    if station_id is not None and not stations:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Station not found")
        
    return [
        StationRead(
            id=station.id,
            longitude=station.longitude,
            latitude=station.latitude,
            elevation=station.elevation,
            crs=station.crs,
            name=station.name,
        )
        for station in stations
    ]

@app.post("/api/stations", response_model = StationRead, status_code=status.HTTP_201_CREATED)
def add_station_to_database(request: Request, station: StationCreate):

    db = request.app.state.runtime.db

    try:
        new_station = db.add_station(**station.model_dump())
    except StationAlreadyExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A station with this ID already exists",
        ) from exc
    except ValueError as exc:
        # add_station uses ValueError for coordinate validation.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return StationRead(
        id=new_station.id,
        latitude=new_station.latitude,
        longitude=new_station.longitude,
        elevation=new_station.elevation,
        crs=station.crs,
        name=new_station.name,
    )

@app.post("/api/refresh_database", response_model=DatabaseRefreshSubmission, status_code=status.HTTP_202_ACCEPTED)
async def submit_database_refresh(request: Request):

    now_dt = datetime.datetime.now(datetime.UTC)
    request.app.state.scheduler.modify_job(
        "database_refresh",
        next_run_time=datetime.datetime.now(datetime.UTC) + Timedelta(seconds = 3),
    )

    return DatabaseRefreshSubmission(
        submitted_at=now_dt
    )

@app.get("/api/interpolate", response_model=InterpolationSubmission)
async def interpolate(
    request: Request,
    param: str,
    start: str,
    end: str,
    timezone: str = "Europe/Rome",
):
    coordinator = request.app.state.coordinator

    interpolation_request = InterpolationRequest(param=param, start=start, end=end, timezone=timezone)
    logger.info("Submitting request %s", request)
    
    return await coordinator.submit(interpolation_request)
