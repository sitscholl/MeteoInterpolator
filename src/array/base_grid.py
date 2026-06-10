from pathlib import Path
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
from pyproj import CRS
import math
import hashlib
import json
from dataclasses import dataclass

import logging

from ..aoi import AOI
from .cache import CacheManager
from .crs import load_crs_metadata, attach_crs_metadata

logger = logging.getLogger(__name__)

_POSSIBLE_X_DIM_NAMES = ["lon", "longitude", "x"]
_POSSIBLE_Y_DIM_NAMES = ["lat", "latitude", "y"]
_TARGET_X_DIM = 'x'
_TARGET_Y_DIM = 'y'
_CACHE_KEY = 'base_grid'

@dataclass(frozen = True)
class BaseGrid:
    path: Path
    data: xr.DataArray
    aoi: AOI
    resampling_method: str
    from_cache: bool = False

    def __post_init__(self):
        if self.data.rio.crs is None:
            raise ValueError("Base grid must define a CRS")
        for req_dim in ['x', 'y']:
            if req_dim not in self.data.dims:
                raise ValueError(f"Missing dimension {req_dim} in base grid")

def _find_dim_name(data: xr.DataArray | xr.Dataset, lookup_names: list[str]) -> str:
    
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

def _prepare_spatial_dims(
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
        x_dim_name = _find_dim_name(data, _POSSIBLE_X_DIM_NAMES + [n.upper() for n in _POSSIBLE_X_DIM_NAMES])
    else:
        if x_dim not in data.dims:
            raise ValueError(f"Provided x_dim '{x_dim}' not found in dataset dimensions: {list(data.dims)}")
        x_dim_name = x_dim

    if y_dim is None:
        y_dim_name = _find_dim_name(data, _POSSIBLE_Y_DIM_NAMES + [n.upper() for n in _POSSIBLE_Y_DIM_NAMES])
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

def _open_uncached_grid(path: str, var: str | None = None, squeeze: bool = True):

    try:
        if Path(path).suffix == '.zarr':
            data = xr.open_zarr(path)
        else:
            data = xr.open_dataset(path)
    except Exception as e:
        logger.exception(f"Error opending base grid at {path}: {e}")
    
    available_vars = [var_name for var_name in data.data_vars.keys() if var_name != "spatial_ref"]
    if var is not None:
        if var not in data.data_vars:
            raise ValueError(f"Data variable {var} not found in dataset. Available variables: {available_vars}")
        selected = data[var]
    else:
        if not available_vars:
            raise ValueError(f"No data variables found in base grid at {path}.")
        if len(available_vars) > 1:
            logger.warning(f"Found multiple variables in base grid: {available_vars}. Picking first one.")
        selected = data[next(iter(available_vars))]

    if "spatial_ref" in data:
        selected = selected.assign_coords(spatial_ref=data["spatial_ref"])
    data = selected

    data = load_crs_metadata(data) #try to get crs info

    if squeeze:
        data = data.squeeze(drop = True)

    return data

def _check_reprojection(data, target_crs, target_res):
    needs_reprojection = False
    if CRS.from_user_input(data.rio.crs) != CRS.from_user_input(target_crs):
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
    
    return needs_reprojection

def generate_cache_payload(
    path: Path,
    target_crs: int | None,
    target_res: int | float | tuple[float, float] | None,
    crs: int | None,
    x_dim: str | None,
    y_dim: str | None,
    aoi: AOI | None,
    aoi_buffer_m: int | float | None,
    resampling_method: str | Resampling,
    **kwargs
) -> Path:
    if isinstance(target_res, (int, float)):
        target_res = (float(target_res), float(target_res))
    elif target_res is not None:
        target_res = (float(target_res[0]), float(target_res[1]))

    if isinstance(resampling_method, Resampling):
        resampling_method = resampling_method.name.lower()
    else:
        resampling_method = str(resampling_method).lower()

    var = kwargs.get("var")
    squeeze = kwargs.get("squeeze", True)

    aoi_info = None
    if aoi is not None:
        aoi_bounds = tuple(float(v) for v in aoi.bounds)
        aoi_crs = getattr(aoi, "crs", None)
        aoi_info = {"bounds": aoi_bounds, "crs": aoi_crs}

    return {
        "source_path": str(Path(path).resolve()),
        "var": var,
        "squeeze": squeeze,
        "target_crs": target_crs,
        "target_res": target_res,
        "target_x_dim": _TARGET_X_DIM,
        "target_y_dim": _TARGET_Y_DIM,
        "crs": crs,
        "x_dim": x_dim,
        "y_dim": y_dim,
        "aoi": aoi_info,
        "aoi_buffer_m": aoi_buffer_m,
        "resampling_method": resampling_method,
    }

def _prepare_loaded_array(data, aoi, aoi_buffer_m, target_crs, x_dim, y_dim, target_res, needs_reprojection, resampling_method):
    data = _prepare_spatial_dims(
        data,
        x_dim=x_dim,
        y_dim=y_dim,
        x_dim_target=_TARGET_X_DIM,
        y_dim_target=_TARGET_Y_DIM,
    )

    # Filter before reprojecting
    if aoi is not None:
        if needs_reprojection:
            data = aoi.filter_bbox(data, buffer_m=aoi_buffer_m)
        else:
            data = aoi.filter_bbox(data)

    if needs_reprojection:
        if CRS.from_user_input(data.rio.crs) != CRS.from_user_input(target_crs):
            logger.debug(f"Reprojecting base grid from crs {data.rio.crs} to {target_crs}")
        
        original_res_x, original_res_y = data.rio.resolution()
        original_res = (abs(original_res_x), abs(original_res_y))
        if target_res is not None and original_res != target_res:
            logger.debug(f"Reprojecting base grid from resolution of {original_res} to {target_res}")
        
        if isinstance(resampling_method, str):
            try:
                resampling_method = Resampling[resampling_method.lower()]
            except KeyError as exc:
                valid_methods = ", ".join(r.name.lower() for r in Resampling)
                raise ValueError(f"Unknown resampling_method '{resampling_method}'. Valid options: {valid_methods}") from exc
        
        data = data.rio.reproject(dst_crs = target_crs, resolution = target_res, resampling = resampling_method)
        if aoi is not None:
            data = aoi.filter_bbox(data)

    return data
    
def load_base_grid(
    path: str | Path,
    target_crs: int, 
    target_res: int | float | tuple[float, float], 
    original_crs: int | None = None,
    aoi: AOI | None = None, 
    aoi_buffer_m: int | float | None = None,
    x_dim: str | None = None,
    y_dim: str | None = None,
    resampling_method: str | Resampling = 'bilinear',
    cache_manager: CacheManager | None = None,
    **kwargs
    ):
    
    data = None
    from_cache = False
    cache_payload = generate_cache_payload()
    if cache_manager is not None:
        data = cache_manager.load_cache(_CACHE_KEY, cache_payload)

    if data is not None:
        path = cache_manager._build_cache_path(_CACHE_KEY, cache_payload)
        logger.info(f"Using cached base grid from {path}")
        
        if data.rio.crs is None:
            raise ValueError(
                f"Base grid loaded from {path} does not define a CRS. "
                "Rebuild the cache with CRS metadata."
            )
        if target_crs is not None and CRS.from_user_input(data.rio.crs) != CRS.from_user_input(target_crs):
            raise ValueError(
                f"Cached base grid CRS {data.rio.crs} does not match requested target_crs EPSG:{target_crs}. "
                f"Delete cache at {path} and rebuild it."
            )

        data = _prepare_spatial_dims(
            data,
            x_dim=_TARGET_X_DIM,
            y_dim=_TARGET_Y_DIM,
            x_dim_target=_TARGET_X_DIM,
            y_dim_target=_TARGET_Y_DIM,
        )
        needs_reprojection = False
        from_cache = True

    else:
        data = _open_uncached_grid(path)
        data_crs = data.rio.crs

        if original_crs is None and data_crs is None:
            raise ValueError(
                "Dataset crs could not be loaded when opening file. Please provide crs manually in config via crs key."
            )

        if data_crs is None:
            data = data.rio.write_crs(original_crs, inplace = False)

        if original_crs is not None and CRS.from_user_input(data_crs) != CRS.from_user_input(original_crs):
                logger.warning(
                    f"Provided crs does not correspond to dataset crs and will be ignored. {original_crs} vs {data_crs}"
                )
        if target_crs is None:
            target_crs = data_crs

        needs_reprojection = _check_reprojection(data, target_crs, target_res)
    
        data = _prepare_loaded_array(
            data, 
            aoi = aoi, 
            aoi_buffer_m = aoi_buffer_m, 
            x_dim = x_dim,
            y_dim = y_dim,
            target_crs = target_crs, 
            target_res = target_res, 
            needs_reprojection = needs_reprojection,
            resampling_method=resampling_method
            )

    if data.isnull().any().compute().item():
        raise ValueError('BaseGrid cannot contain NaN values. Check reprojection and aoi settings. Set higher value for aoi_buffer_m?')

    if cache_manager is not None and not from_cache:
        cache_manager.initialize_cache(data, key = _CACHE_KEY, cache_params=cache_payload)
        logger.info(f"Base grid cache written to {cache_manager._build_cache_path(key = _CACHE_KEY, cache_params=cache_payload)}")

    return BaseGrid(
        path = path,
        data = data,
        aoi = aoi,
        from_cache = from_cache
    )
