from pathlib import Path
import json
import hashlib
import numpy as np
import xarray as xr
import rioxarray

import logging

from .crs import load_crs_metadata, attach_crs_metadata

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(
        self,
        cache_dir: str | Path,
        zarr_format: int = 2
    ):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir
        self.zarr_format = zarr_format
    
    @staticmethod
    def _to_jsonable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): CacheManager._to_jsonable(val) for key, val in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [CacheManager._to_jsonable(item) for item in value]
        return value

    def _build_cache_path(self, key, cache_params):
        serialized = json.dumps(self._to_jsonable(cache_params), sort_keys=True, separators=(",", ":"))
        cache_id = hashlib.blake2b(serialized.encode("utf-8"), digest_size=16).hexdigest()
        return self.cache_dir / f"{key}_{cache_id}.zarr"

    def initialize_cache(self, data: xr.DataArray | xr.Dataset, key: str, cache_params: dict):

        if isinstance(data, xr.DataArray):
            data_name = data.name or key
            data = data.to_dataset(name=data_name)

        if not isinstance(data, xr.Dataset):
            raise ValueError(f"Data to cache must either be an xarray DataArray or Dataset. Got {type(data)}")

        if data.rio.crs is None:
            raise ValueError("Cannot cache a dataset without a crs specified.")

        cache_path = self._build_cache_path(key, cache_params)
        if cache_path.exists():
            raise ValueError(f"Found already existing cache at {cache_path}")

        try:
            logger.debug(f"Initializing new cache at {cache_path}") 
            data_cache = attach_crs_metadata(data, crs = data.rio.crs)           
            data_cache.to_zarr(cache_path, zarr_format=self.zarr_format)
            return cache_path
        except Exception as e:
            logger.warning(f"Failed to initialize cache at {cache_path} with error: {e}")
            return None

    def load_cache(self, key: str, cache_params: dict):
        cache_path = self._build_cache_path(key, cache_params)

        if cache_path.exists():
            try:
                logger.info(f"Loading existing cache at {cache_path}")
                dataset = xr.open_zarr(cache_path)
                dataset = load_crs_metadata(dataset)

                if dataset.rio.crs is None:
                    logger.warning(f"Unable to load crs information from cache at {cache_path}")

                return dataset
            except Exception as e:
                logger.warning(f"Failed to load cache with error: {e}")
        
        return None