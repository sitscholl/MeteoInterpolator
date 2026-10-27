import logging
import os
import shutil
from pathlib import Path
from time import sleep
from uuid import uuid4

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

    partial_id = str(uuid4())
    partial_filename = out_filename.with_name(
        f".{out_filename.stem}.partial_{partial_id}.zarr"
    )

    if out_filename.exists():
        raise ValueError(f"Output file {out_filename} already exists! Cannot write job result.")

    try:
        ds = result.to_dataset()
        ds.attrs['request_metadata'] = to_jsonable(metadata)

        store = ds.to_zarr(partial_filename, mode='w-', zarr_format=3, consolidated=False)
        store.close()
        _publish_store(partial_filename, out_filename)
    except Exception as e:
        if partial_filename.exists():
            shutil.rmtree(partial_filename)
        raise

    return out_filename
