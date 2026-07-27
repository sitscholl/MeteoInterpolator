import asyncio
import logging

import pandas as pd
from fastapi import FastAPI

from ..coordinator import InterpolationCoordinator
from ..runtime import RuntimeContext
from .schemas import InterpolationRequest, InterpolationSubmission

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Meteorological Interpolation API",
    description="Meteorological Interpolation service that calculates meteorological parameter for unknown points based on observed station data",
    version="1.0.0"
)

## Initialize runtime context & Coordinator
async def initialize_runtime(file):
    runtime = await RuntimeContext.from_config_file(file)
    return runtime

runtime = asyncio.run(initialize_runtime('config.example.yaml'))
coordinator = InterpolationCoordinator(runtime)

@app.get("/api/health")
async def health_check():
    """Lightweight liveness check for container health probes."""
    return {
        "status": "online"
    }

@app.get("/api/stations")
async def display_interpolation_stations():
    return {
        runtime.station_catalog.to_json()
    }

@app.get("/api/interpolate")
async def interpolate(param: str, start: str, end: str, response_model=InterpolationSubmission):
    
    request = InterpolationRequest(param = param, start = pd.Timestamp(start), end = pd.Timestamp(end))
    logger.info("Submitting request %s", request)
    return await coordinator.submit(request)
