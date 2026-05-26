from typing import Sequence
import logging
from pathlib import Path
from shutil import rmtree

import xarray as xr

from .base import BaseResidualModel
from ..distance import calculate_non_euclidean_distance

logger = logging.getLogger(__name__)

class InverseDistanceWeighting(BaseResidualModel):
    def __init__(
        self,
        connectivity_type: int = 4,
        lam_values: Sequence[float] | None = None,
        max_visibility_distance: float | None = None,
        cache_directory: str | None = None
        ):

        if connectivity_type not in [4, 8]:
            raise ValueError(f"Connectivity type should be one of 4 or 8. Got {connectivity_type}")

        if lam_values is not None and any([i < 0 for i in lam_values]):
            raise ValueError("lam_values must be > 0!")

        if lam_values is not None and max_visibility_distance is None:
            logger.debug("No max_visibility_distance provided; atmospheric movement through free-air not considered in non-euclidean distance metric.")

        if lam_values is None or lam_values == [0]:
            logger.debug("Euclidean distance will be used as distance metric because lam_values is None or [0]. max_visibility_distance will be ignored")
            lam_values = [0]
            max_visibility_distance = None

        self.connectivity_type = connectivity_type
        self.lam_values = lam_values
        self.max_visibility_distance = max_visibility_distance

        if cache_directory is not None:
            cache_directory = Path(cache_directory)
            cache_directory.parent.mkdir(exist_ok = True, parents = True)
            logger.debug(f"Distance fields will be cached at {cache_directory}")
        self.cache_directory = cache_directory

    def key(self):
        return 'idw'

    def _validate_point_in_grid(self, x: float, y:float, grid: xr.DataArray) -> bool:
        pass

    def _build_cache_id(self):
        pass

    def _load_cache(self, cache_path: Path):
        pass

    def calculate_distance_for_point(self, x:float, y:float, id: str, dem: xr.DataArray):
        point_in_grid = self._validate_point_in_grid(x, y, dem)
        
        if point_in_grid:
            return calculate_non_euclidean_distance(
                dem, 
                [x], [y], [id],
                lam_values = self.lam_values, 
                connectivity=self.connectivity_type,
                max_visibility_distance=self.max_visibility_distance
                )
        else:
            logger.warning(f"Point with id {id} is not within provided dem. Cannot calculate distance.")
            return None

    def calculate_distance_fields(
        self, 
        x_coords: list[float], 
        y_coords: list[float], 
        point_ids: list[str], 
        dem: xr.DataArray
        ):
        
        point_ids = [str(i) for i in point_ids]

        cache_path = None
        if self.cache_directory is not None:
            cache_id = self._build_cache_id()
            cache_path = Path(self.cache_directory, f"distance_cache_{cache_id}")
        
            if cache_path.exists():
                try:
                    logger.info(f"Reusing existing distance cache at {cache_path}")
                    return self._load_cache(cache_path)
                except Exception as e:
                    logger.exception("Failed to load distance cache. Deleting cache and calculating new cache")
                    rmtree(cache_path)

        results = []
        for x, y, pid in zip(x_coords, y_coords, point_ids):
            distance = self.calculate_distance_for_point(x, y, pid, dem)
            if distance is not None:
                results.append(distance)

        if not results:
            logger.warning("No distance fields could be calculated")
            return None

        results = xr.merge(results, concat_dim = 'id')

        if cache_path is not None:
            try:
                logger.info(f"Initializing new distance cache at {cache_path}")
                results.to_zarr(cache_path)
            except Exception as e:
                logger.exception(f"Failed to initialize distance cache at {cache_path}")

        return results

