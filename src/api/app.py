import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from ..coordinator import InterpolationCoordinator
from ..database.db import StationAlreadyExistsError
from ..runtime import RuntimeContext
from .schemas import (
    InterpolationRequest,
    InterpolationSubmission,
    StationCreate,
    StationRead,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.example.yaml"

@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime = await RuntimeContext.from_config_file(CONFIG_PATH)

    try:
        if runtime.db is None:
            raise RuntimeError(
                "The station API requires database configuration"
            )

        coordinator = InterpolationCoordinator(runtime)
        app.state.runtime = runtime
        app.state.coordinator = coordinator

        try:
            yield
        finally:
            await coordinator.shutdown()
    finally:
        if runtime.db is not None:
            runtime.db.close()
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
        name=new_station.name,
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
