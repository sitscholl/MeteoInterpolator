import logging
import os
import shutil
from pathlib import Path
from time import sleep
from uuid import uuid4

from pandas import DateOffset, Timestamp

from .array.crs import attach_crs_metadata
from .utils import to_jsonable, to_utc_timestamp
from .worker import InterpolationJobResult

logger = logging.getLogger(__name__)

def _publish_store(partial_filename: Path, out_filename: Path) -> None:
    for attempt in range(5):
        if out_filename.exists():
            raise FileExistsError(f"Output store already exists: {out_filename}")
        try:
            partial_filename.replace(out_filename)
            return
        except PermissionError:
            # Windows can briefly lock newly written directories. Retry only
            # publication, without recomputing or rewriting the completed store.
            if os.name != "nt" or attempt == 4:
                raise
            sleep(.05 * 2 ** attempt)

def attach_request_metadata(ds, timestamp, metadata):

    metadata = dict(metadata)
    aggregation_timezone = metadata.pop('timezone') if 'timezone' in metadata else None
    aggregation_frequency = metadata.pop('freq') if 'freq' in metadata else None
    
    ds.attrs['request_metadata'] = to_jsonable(metadata)
    ds.attrs.update({
        "aggregation_timezone": aggregation_timezone,
        "aggregation_frequency":aggregation_frequency,
    })

    if 'time' in ds.coords:
        ds.time.attrs.update({
            "timezone": "UTC",
            "long_name": "Start of aggregation interval",
        })

    if aggregation_timezone is None:
        logger.warning("Aggregation timezone is None; output zarr files will have limited metadata.")

    if aggregation_timezone is not None and aggregation_frequency == 'D':
        local_timestamp = Timestamp(timestamp).tz_convert(aggregation_timezone)
        next_timestamp = local_timestamp + DateOffset(days=1)

        ds = ds.assign_coords(local_date=("time", [local_timestamp.strftime("%Y-%m-%d")]))
        ds.local_date.attrs.update({
            "long_name": "Local calendar date of the daily aggregation",
            "timezone": aggregation_timezone,
        })

        time_bounds = [
            (local_timestamp.tz_convert('UTC').tz_localize(None), next_timestamp.tz_convert('UTC').tz_localize(None))
            ]
        ds = ds.assign_coords(time_bounds=(('time', 'bounds'), time_bounds))
        ds.time.attrs["bounds"] = "time_bounds"

    return ds

def write_job_result(
    result: InterpolationJobResult,
    root: str | Path,
    request_id: str,
    metadata: dict,
    ) -> Path:

    if result.status in ['failed', 'skipped']:
        raise ValueError(f"Writing job results only works on completed interpolation jobs. Got job with status {result.status}")

    root = Path(root) / request_id
    root.mkdir(parents=True, exist_ok=True)

    job_timestamp = to_utc_timestamp(result.job.timestamp)
    zarr_name = f"{result.job.parameter}_{metadata['freq']}_{job_timestamp:%Y%m%dT%H%M%SZ}.zarr"
    out_filename = Path(root / zarr_name)

    logger.debug("Writing job results to %s", out_filename)

    # Zarr adds its own temporary filenames inside this directory. Keep the
    # directory name short enough for normal Windows paths with a request UUID.
    partial_filename = root / f".partial_{uuid4().hex}.zarr"

    if out_filename.exists():
        raise ValueError(f"Output file {out_filename} already exists! Cannot write job result.")

    try:
        ds = result.to_dataset()
        if ds.rio.crs is None:
            raise ValueError("Output dataset must define a CRS.")

        ds = attach_crs_metadata(ds, ds.rio.crs)
        ds = attach_request_metadata(ds, result.job.timestamp, metadata)

        ds.attrs = to_jsonable(ds.attrs)
        for name in ds.variables:
            ds[name].attrs = to_jsonable(ds[name].attrs)

        store = ds.to_zarr(partial_filename, mode='w-', zarr_format=3, consolidated=False)
        store.close()
        _publish_store(partial_filename, out_filename)
    except Exception as e:
        if partial_filename.exists():
            shutil.rmtree(partial_filename)
        raise

    return out_filename
