from pathlib import Path
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
import math

import logging

from .aoi import AOI

logger = logging.getLogger(__name__)

_POSSIBLE_X_DIM_NAMES = ["lon", "longitude", "x"]
_POSSIBLE_Y_DIM_NAMES = ["lat", "latitude", "y"]

class BaseGrid:

    def __init__(
        self, 
        path: str | Path, 
        crs: int | None = None, 
        res: int | float | tuple[float, float] | None = None, 
        x_dim: str | None = None,
        y_dim: str | None = None,
        x_dim_target: str = 'x',
        y_dim_target: str = 'y',
        aoi: AOI | None = None,
        resampling_method: str | Resampling = 'bilinear',
        **kwargs
        ):
        
        data = self.open(path, **kwargs)

        data = self._normalize_spatial_dims(data, x_dim=x_dim, y_dim=y_dim, x_dim_target = x_dim_target, y_dim_target = y_dim_target)

        needs_reprojection = False
        if crs is not None:
            if data.rio.crs is None:
                logger.warning(f"Original crs could not be loaded when opening {path}. Setting crs to {crs}")
                data = data.rio.write_crs(crs)
            elif data.rio.crs.to_epsg() != crs:
                needs_reprojection = True
        else:
            if data.rio.crs is None:
                raise ValueError(f"Original crs could not be loaded when opening {path} and crs has not been specified.")
            crs = data.rio.crs.to_epsg()
        self.crs = crs

        original_res_x, original_res_y = data.rio.resolution()
        original_res = (abs(original_res_x), abs(original_res_y))
        if res is not None:
            if isinstance(res, (int, float)):
                target_res = (float(res), float(res))
            else:
                if len(res) != 2:
                    raise ValueError("res must be a scalar or a 2-tuple (x_res, y_res).")
                target_res = (float(res[0]), float(res[1]))
            if not (math.isclose(original_res[0], abs(target_res[0])) and math.isclose(original_res[1], abs(target_res[1]))):
                needs_reprojection = True
        else:
            target_res = original_res

        if needs_reprojection:
            if data.rio.crs.to_epsg() != crs:
                logger.debug(f"Reprojecting base grid from crs {data.rio.crs} to {crs}")
            if original_res != target_res:
                logger.debug(f"Reprojecting base grid from resolution of {original_res} to {target_res}")
            if isinstance(resampling_method, str):
                try:
                    resampling_method = Resampling[resampling_method.upper()]
                except KeyError as exc:
                    valid_methods = ", ".join(r.name.lower() for r in Resampling)
                    raise ValueError(f"Unknown resampling_method '{resampling_method}'. Valid options: {valid_methods}") from exc
            data = data.rio.reproject(dst_crs = crs, resolution = target_res, resampling = resampling_method)

        if aoi is not None:
            data = aoi.filter_bbox(data)

        self.data = data
        self.x_dim = x_dim_target
        self.y_dim = y_dim_target

    def open(self, path, var: str | None = None, squeeze: bool = True):
        data = xr.open_dataset(path)
        
        available_vars = list(data.data_vars.keys())
        if var is not None:
            if var not in available_vars:
                raise ValueError(f"Data variable {var} not found in dataset. Available variables: {available_vars}")
            data = data[var]
        else:
            if len(available_vars) > 1:
                logger.warning(f"Found multiple variables in base grid: {available_vars}. Picking first one.")
            data = data[next(iter(available_vars))]

        if squeeze:
            data = data.squeeze(drop = True)

        return data

    def _find_dim_name(self, data: xr.DataArray | xr.Dataset, lookup_names: list[str]) -> str:
        
        nams_found = []
        for nam in lookup_names:
            if nam in data.dims:
                nams_found.append(nam)

        if len(nams_found) == 0:
            raise ValueError(f"None of the potential dimension names {lookup_names} were found in the dataset.")
        elif len(nams_found) == 1:
            return nams_found[0]
        else:
            logger.warning(f"Found multiple matching dimension names: {nams_found}. Using first one")
            return nams_found[0]

    def _normalize_spatial_dims(
        self,
        data: xr.DataArray | xr.Dataset,
        x_dim_target: str, 
        y_dim_target: str,
        x_dim: str | None = None,
        y_dim: str | None = None,
    ) -> xr.DataArray | xr.Dataset:
        if not x_dim_target or not y_dim_target:
            raise ValueError("x_dim_target and y_dim_target must be non-empty strings.")
        if x_dim_target == y_dim_target:
            raise ValueError("x_dim_target and y_dim_target must be different.")

        if x_dim is None:
            x_dim_name = self._find_dim_name(data, _POSSIBLE_X_DIM_NAMES + [n.upper() for n in _POSSIBLE_X_DIM_NAMES])
        else:
            if x_dim not in data.dims:
                raise ValueError(f"Provided x_dim '{x_dim}' not found in dataset dimensions: {list(data.dims)}")
            x_dim_name = x_dim

        if y_dim is None:
            y_dim_name = self._find_dim_name(data, _POSSIBLE_Y_DIM_NAMES + [n.upper() for n in _POSSIBLE_Y_DIM_NAMES])
        else:
            if y_dim not in data.dims:
                raise ValueError(f"Provided y_dim '{y_dim}' not found in dataset dimensions: {list(data.dims)}")
            y_dim_name = y_dim

        rename_map = {}
        if x_dim_name != x_dim_target:
            rename_map[x_dim_name] = x_dim_target
        if y_dim_name != y_dim_target:
            rename_map[y_dim_name] = y_dim_target
        if len(set(rename_map.values())) != len(rename_map.values()):
            raise ValueError(
                f"Cannot rename dims {rename_map} because multiple source dims map to the same target."
            )
        for src, tgt in rename_map.items():
            if tgt in data.dims and tgt not in rename_map:
                raise ValueError(
                    f"Cannot rename dim '{src}' to '{tgt}' because '{tgt}' already exists in the dataset."
                )
        if rename_map:
            data = data.rename(rename_map)

        # Ensure rioxarray knows which dims are spatial after renaming.
        data = data.rio.set_spatial_dims(x_dim=x_dim_target, y_dim=y_dim_target, inplace=False)
        return data
