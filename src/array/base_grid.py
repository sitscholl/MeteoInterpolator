from pathlib import Path
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
import math

import logging

from ..aoi import AOI

logger = logging.getLogger(__name__)

_POSSIBLE_X_DIM_NAMES = ["lon", "longitude", "x"]
_POSSIBLE_Y_DIM_NAMES = ["lat", "latitude", "y"]

class BaseGrid:

    def __init__(
        self, 
        path: str | Path, 
        target_crs: int | None = None,
        target_res: int | float | tuple[float, float] | None = None, 
        target_x_dim: str = 'x',
        target_y_dim: str = 'y',
        crs: int | None = None, 
        x_dim: str | None = None,
        y_dim: str | None = None,
        aoi: AOI | None = None,
        resampling_method: str | Resampling = 'bilinear',
        **kwargs
        ):
        
        data = self.open(path, **kwargs)

        data = self._normalize_spatial_dims(data, x_dim=x_dim, y_dim=y_dim, x_dim_target = target_x_dim, y_dim_target = target_y_dim)

        needs_reprojection = False
        original_crs_obj = data.rio.crs
        if crs is None and original_crs_obj is None:
            raise ValueError(
                "Dataset crs could not be loaded when opening file. Please provide crs manually in config via crs key."
            )
        if original_crs_obj is None:
            original_crs = crs
            data = data.rio.write_crs(original_crs)
        else:
            original_crs = original_crs_obj.to_epsg()
            if crs is not None and original_crs != crs:
                logger.warning(
                    f"Provided crs does not correspond to dataset crs and will be ignored. {crs} vs {original_crs}"
                )

        if target_crs is None:
            target_crs = original_crs
        elif original_crs != target_crs:
            needs_reprojection = True

        original_res_x, original_res_y = data.rio.resolution()
        original_res = (abs(original_res_x), abs(original_res_y))
        if target_res is not None:
            if isinstance(target_res, (int, float)):
                target_res = (float(target_res), float(target_res))
            else:
                if len(target_res) != 2:
                    raise ValueError("target_res must be a scalar or a 2-tuple (x_res, y_res).")
                target_res = (float(target_res[0]), float(target_res[1]))
            if not (math.isclose(original_res[0], abs(target_res[0])) and math.isclose(original_res[1], abs(target_res[1]))):
                needs_reprojection = True

        # Filter before reprojecting
        if aoi is not None:
            data = aoi.filter_bbox(data)

        if needs_reprojection:
            if data.rio.crs.to_epsg() != target_crs:
                logger.debug(f"Reprojecting base grid from crs {data.rio.crs} to {target_crs}")
            if target_res is not None and original_res != target_res:
                logger.debug(f"Reprojecting base grid from resolution of {original_res} to {target_res}")
            if isinstance(resampling_method, str):
                try:
                    resampling_method = Resampling[resampling_method.lower()]
                except KeyError as exc:
                    valid_methods = ", ".join(r.name.lower() for r in Resampling)
                    raise ValueError(f"Unknown resampling_method '{resampling_method}'. Valid options: {valid_methods}") from exc
            data = data.rio.reproject(dst_crs = target_crs, resolution = target_res, resampling = resampling_method)

        self.data = data
        self.crs = target_crs
        self.res = target_res if target_res is not None else original_res
        self.x_dim = target_x_dim
        self.y_dim = target_y_dim

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

    def __repr__(self):
        return f"BaseGrid (crs: {self.crs}, resolution: {self.res}, shape: {self.data.shape}, dims = {self.data.dims})"
