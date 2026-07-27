import logging
import os

import uvicorn

logger = logging.getLogger(__name__)

def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

def main():

    ##Set up application logging
    verbose = _env_bool("VERBOSE_LOGGING", True)

    log_level = logging.DEBUG if verbose else logging.INFO
    #log_formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S")
    logging.basicConfig(level = log_level, force = True, format='%(asctime)s %(name)s - %(levelname)s - %(message)s')
    
    #Avoid noisy loggers spam
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('rasterio').setLevel(logging.WARNING)

    logger.info("="*50)
    logger.info("Starting MeteoInterpolator")
    logger.info("="*50)

    uvicorn.run(
        "src.api.app:app",
        host=os.getenv("UVICORN_HOST", "0.0.0.0"),
        port=int(os.getenv("UVICORN_PORT", "8000")),
        workers=int(os.getenv("UVICORN_WORKERS", "1")),
        log_level=os.getenv("UVICORN_LOG_LEVEL", (log_level or 1)),
        log_config=None,
        access_log=_env_bool("UVICORN_ACCESS_LOG", True),
    )


if __name__ == "__main__":
    main()