from typing import Sequence
import logging
from pathlib import Path
from shutil import rmtree

import xarray as xr
import numpy as np

from .base import BaseResidualModel
from ..distance import calculate_non_euclidean_distance

logger = logging.getLogger(__name__)

class InverseDistanceWeighting(BaseResidualModel):
    def __init__(
        self,
        connectivity_type: int = 4,
        lam_values: Sequence[float] | None = None,
        max_visibility_distance: float | None = None,
        cache_directory: str | None = None,
        neighbours: int = 5
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

        self.neighbours = neighbours

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

    def _validate_input_array(self, array: xr.DataArray, name: str, spatial: bool = False):
        if not isinstance(array, xr.DataArray):
            raise ValueError(f'{name} input should be an xarray DataArray. Got {type(array)}')
        if 'id' not in array.dims or array.sizes['id'] == 0:
            raise ValueError(f"{name} input must have an id dimension with length > 0. Got {array.dims}")

        if spatial:
            if 'y' not in array.dims:
                raise ValueError(f"Spatial grid {name} requires a y dimension. Got {array.dims}")
            if 'x' not in array.dims:
                raise ValueError(f"Spatial grid requireds an x dimension. Got {array.dims}")

    def interpolate(self, y: xr.DataArray, distance_fields: xr.DataArray):
        
        self._validate_input_array(y, name = 'y')
        self._validate_input_array(distance_fields, name = 'distance_fields', spatial = True)

        common_ids = set(y.id.values).intersection(distance_fields.id.values)
        if len(common_ids) == 0:
            logger.warning("No common ids between y input and distance_fields array. y-values cannot be interpolated")
            return None

        missing_ids = set(y.id.values) - distance_fields.id.values
        if len(missing_ids) > 0:
            logger.warning(f"No distance fields for the following ids were provided. They will not be considered in the interpolation: {missing_ids}")
        
        y = y.sel(id = common_ids)
        distance_fields_start = distance_fields.sel(id = common_ids).copy()
        distance_fields_start = distance_fields_start.where(distance_fields_start > 0) #set source grid pixels to np.nan
        
        residual_factor = (y/distance_fields_start**2)
        
        coords_dict = dict(x=distance_fields.x.values, y=distance_fields.y.values)
        w_tot = xr.DataArray(0.0, dims = ('y', 'x'), coords=coords_dict)
        R = xr.DataArray(0.0, dims = ('y', 'x'), coords=coords_dict)
        
        for i in range(self.neighbours):
            #Get index of minimum value for each pixel
            arr_idx = distance_fields_start.fillna(np.inf).idxmin('st_id')
            ##Add step that clips to aoi, because pixels with nan values in arr_start are assigned the id of the first station in arr_start

            #For each pixel extract minimum distance over all stations
            arr_min = distance_fields_start.sel(st_id = arr_idx).drop('st_id')

            #For each pixel extract the residual factor that corresponds to the minimum distance
            arr_res_sel = residual_factor.sel(st_id = arr_idx).drop('st_id')

            w_tot += (1/(arr_min**2))
            R += arr_res_sel

            distance_fields_start = distance_fields_start.where(distance_fields_start > arr_min)

        R = (1/w_tot) * R
        return(R)