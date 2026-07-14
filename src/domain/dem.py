from pathlib import Path
from urllib.parse import urlparse
import xarray as xr
import rioxarray  # noqa: F401
from dataclasses import dataclass

import logging

from ..array.cache import CacheManager
from ..array.crs import load_crs_metadata

logger = logging.getLogger(__name__)

_POSSIBLE_X_DIM_NAMES = ["lon", "longitude", "x"]
_POSSIBLE_Y_DIM_NAMES = ["lat", "latitude", "y"]
_TARGET_X_DIM = 'x'
_TARGET_Y_DIM = 'y'
def _is_remote_uri(path: str | Path) -> bool:
    parsed = urlparse(str(path))
    return parsed.scheme in {"http", "https", "s3", "gs", "az"}

def _source_identity(path: str | Path) -> str:
    if _is_remote_uri(path):
        return str(path)
    return str(Path(path).resolve())

def _source_suffix(path: str | Path) -> str:
    text = str(path)
    if _is_remote_uri(text):
        return Path(urlparse(text).path).suffix
    return Path(text).suffix

@dataclass(frozen = True)
class DEM:
    path: str | Path
    data: xr.DataArray
    fingerprint: str

    def __post_init__(self):
        if not isinstance(self.data, xr.DataArray):
            raise TypeError(f"DEM must be a DataArray. Got {type(self.data)}")
        if self.data.rio.crs is None:
            raise ValueError("DEM must define a CRS")
        for req_dim in ['x', 'y']:
            if req_dim not in self.data.dims:
                raise ValueError(f"Missing dimension {req_dim} in DEM")
        if self.data.isnull().any().compute().item():
            raise ValueError('DEM cannot contain NaN values.')

    @property
    def crs(self):
        return self.data.rio.crs

    @property
    def bounds(self):
        return self.data.rio.bounds()

    def __repr__(self):
        return f"DEM (shape = {self.data.shape}, dims = {self.data.dims}, crs = {self.crs})"

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

def _select_data_var(data: xr.DataArray | xr.Dataset, path: str | Path, var: str | None = None) -> xr.DataArray:
    if isinstance(data, xr.DataArray):
        if var is not None and data.name not in (None, var):
            raise ValueError(f"Data variable {var} not found in cached DataArray named {data.name}.")
        return data

    available_vars = [var_name for var_name in data.data_vars.keys() if var_name != "spatial_ref"]
    if var is not None:
        if var not in data.data_vars:
            raise ValueError(f"Data variable {var} not found in dataset. Available variables: {available_vars}")
        selected = data[var]
    else:
        if not available_vars:
            raise ValueError(f"No data variables found in DEM at {path}.")
        if len(available_vars) > 1:
            logger.warning(f"Found multiple variables in DEM: {available_vars}. Picking first one.")
        selected = data[next(iter(available_vars))]

    if "spatial_ref" in data:
        selected = selected.assign_coords(spatial_ref=data["spatial_ref"])
    return selected

def load_dem(
    path: str | Path,
    var: str | None = None,
    crs: str | None = None,
    squeeze: bool = True,
    engine: str | None = None,
    x_dim: str | None = None,
    y_dim: str | None = None,
    **kwargs
    ) -> DEM:
    
    path = str(path) if _is_remote_uri(path) else Path(path)

    if _source_suffix(path) == '.zarr':
        data = xr.open_zarr(path)
    else:
        open_kwargs = {}
        if engine is not None:
            open_kwargs["engine"] = engine
        data = xr.open_dataset(path, **open_kwargs)

    data = _select_data_var(data, path=path, var=var)
    data = load_crs_metadata(data) #try to get crs info

    if squeeze:
        data = data.squeeze(drop = True)

    data = _prepare_spatial_dims(
        data,
        x_dim=x_dim,
        y_dim=y_dim,
        x_dim_target=_TARGET_X_DIM,
        y_dim_target=_TARGET_Y_DIM,
    )
    data_crs = data.rio.crs

    if crs is None and data_crs is None:
        raise ValueError(
            "Dataset CRS could not be loaded when opening file. Please provide crs manually in config."
        )
    if data_crs is None:
        data = data.rio.write_crs(crs, inplace = False)
        data_crs = data.rio.crs

    fingerprint = CacheManager.array_fingerprint(data)

    data = data.dropna(dim = 'y', how = 'all').dropna(dim = 'x', how = 'all')

    return DEM(
        path = path,
        data = data,
        fingerprint = fingerprint,
    )
