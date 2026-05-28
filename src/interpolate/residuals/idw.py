from typing import Sequence
import logging
from pathlib import Path
from shutil import rmtree

import xarray as xr
import numpy as np

from .base import BaseResidualModel
from ..distance import FieldDistance

logger = logging.getLogger(__name__)

class InverseDistanceWeighting(BaseResidualModel):
    def __init__(
        self,
        neighbours: int = 5
        ):
        self.neighbours = neighbours

    def key(self):
        return 'idw'

    def _validate_input_array(self, array: xr.DataArray):
        if not isinstance(array, xr.DataArray):
            raise ValueError(f'Input should be an xarray DataArray. Got {type(array)}')
        if 'id' not in array.dims or array.sizes['id'] == 0:
            raise ValueError(f"Input must have an id dimension with length > 0. Got {array.dims}")

    def interpolate(self, y: xr.DataArray, distance_fields: FieldDistance):
        
        self._validate_input_array(y)
        distance_fields = distance_fields.data

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