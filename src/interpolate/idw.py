import logging

import xarray as xr

from .distance import DistanceField

logger = logging.getLogger(__name__)

class InverseDistanceWeighting:
    def __init__(
        self,
        neighbours: int = 5
        ):
        if neighbours < 1:
            raise ValueError(f"neighbours must be >= 1. Got {neighbours}")
        self.neighbours = neighbours

    def _validate_input_array(self, array: xr.DataArray):
        if not isinstance(array, xr.DataArray):
            raise ValueError(f'Input should be an xarray DataArray. Got {type(array)}')
        if 'id' not in array.dims or array.sizes['id'] == 0:
            raise ValueError(f"Input must have an id dimension with length > 0. Got {array.dims}")

    def interpolate(self, y: xr.DataArray, distance_fields: DistanceField):
        
        self._validate_input_array(y)
        if not isinstance(distance_fields, DistanceField):
            raise ValueError(f"distance_fields must be a DistanceField. Got {type(distance_fields)}")
        distance_fields = distance_fields.data
        if distance_fields is None:
            raise ValueError("distance_fields.data must not be None")

        distance_ids = set(distance_fields.id.values)
        common_ids = [pid for pid in y.id.values if pid in distance_ids]
        if len(common_ids) == 0:
            logger.warning("No common ids between y input and distance_fields array. y-values cannot be interpolated")
            return None

        missing_ids = set(y.id.values) - set(distance_fields.id.values)
        if len(missing_ids) > 0:
            logger.warning(f"No distance fields for the following ids were provided. They will not be considered in the interpolation: {missing_ids}")
        
        y = y.sel(id = common_ids)
        distance_fields_start = distance_fields.sel(id = common_ids).copy()
        distance_fields_start = distance_fields_start.where(distance_fields_start > 0) #set source grid pixels to np.nan
        
        residual_factor = (y/distance_fields_start**2)
        
        accumulator_template = distance_fields_start.isel(id=0, drop=True)
        w_tot = xr.zeros_like(accumulator_template)
        R = xr.zeros_like(accumulator_template)
        
        for i in range(min(self.neighbours, len(common_ids))):
            #Get index of minimum value for each pixel
            arr_idx = distance_fields_start.fillna(float("inf")).idxmin('id')
            ##Add step that clips to aoi, because pixels with nan values in arr_start are assigned the id of the first station in arr_start

            #For each pixel extract minimum distance over all stations
            arr_min = distance_fields_start.sel(id = arr_idx).drop_vars('id')
            valid_selection = arr_min.notnull()

            #For each pixel extract the residual factor that corresponds to the minimum distance
            arr_res_sel = residual_factor.sel(id = arr_idx).drop_vars('id')
            arr_res_sel = arr_res_sel.where(valid_selection, 0)

            w_tot += (1/(arr_min**2)).where(valid_selection, 0)
            R += arr_res_sel

            distance_fields_start = distance_fields_start.where(distance_fields_start > arr_min)

        R = (R / w_tot).where(w_tot > 0)
        return(R)
