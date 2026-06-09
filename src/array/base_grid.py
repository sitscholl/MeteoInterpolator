from pathlib import Path
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
from pyproj import CRS
import math
import hashlib
import json

import logging

from ..aoi import AOI

logger = logging.getLogger(__name__)

_POSSIBLE_X_DIM_NAMES = ["lon", "longitude", "x"]
_POSSIBLE_Y_DIM_NAMES = ["lat", "latitude", "y"]
_CRS_ATTR = "crs"
_CRS_EPSG_ATTR = "crs_epsg"

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
        aoi_buffer_m: int | float | None = None,
        resampling_method: str | Resampling = 'bilinear',
        cache: bool = True,
        **kwargs
        ):

        cache_path = self._cache_path(
            path = path,
            target_crs = target_crs,
            target_res = target_res,
            target_x_dim = target_x_dim,
            target_y_dim = target_y_dim,
            crs = crs,
            x_dim = x_dim,
            y_dim = y_dim,
            aoi = aoi,
            aoi_buffer_m = aoi_buffer_m,
            resampling_method = resampling_method,
            **kwargs
        ) if cache else None

        if cache_path is not None and cache_path.exists():
            logger.info(f"Base grid cache hit at {cache_path}")
            data = self.open(cache_path, **kwargs)
            data = self._normalize_spatial_dims(
                data,
                x_dim=target_x_dim,
                y_dim=target_y_dim,
                x_dim_target=target_x_dim,
                y_dim_target=target_y_dim,
            )
            data = self._ensure_loaded_crs(data, source=cache_path)
            if target_crs is not None and CRS.from_user_input(data.rio.crs) != CRS.from_user_input(target_crs):
                raise ValueError(
                    f"Cached base grid CRS {data.rio.crs} does not match requested target_crs EPSG:{target_crs}. "
                    f"Delete cache at {cache_path} and rebuild it."
                )
            self.data = data
            self.crs = self._crs_identifier(data.rio.crs)
            res_x, res_y = data.rio.resolution()
            self.res = (abs(res_x), abs(res_y))
            self.x_dim = target_x_dim
            self.y_dim = target_y_dim
            return

        data = self.open(path, **kwargs)

        data = self._normalize_spatial_dims(data, x_dim=x_dim, y_dim=y_dim, x_dim_target = target_x_dim, y_dim_target = target_y_dim)

        needs_reprojection = False
        data = self._restore_crs_from_metadata(data)
        original_crs_obj = data.rio.crs
        if crs is None and original_crs_obj is None:
            raise ValueError(
                "Dataset crs could not be loaded when opening file. Please provide crs manually in config via crs key."
            )
        if original_crs_obj is None:
            original_crs = self._crs_identifier(crs)
            data = self._attach_crs_metadata(data, crs)
        else:
            original_crs = self._crs_identifier(original_crs_obj)
            if crs is not None and CRS.from_user_input(original_crs_obj) != CRS.from_user_input(crs):
                logger.warning(
                    f"Provided crs does not correspond to dataset crs and will be ignored. {crs} vs {original_crs}"
                )

        if target_crs is None:
            target_crs = original_crs
        elif CRS.from_user_input(original_crs) != CRS.from_user_input(target_crs):
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
            if needs_reprojection:
                data = aoi.filter_bbox(data, buffer_m=aoi_buffer_m)
            else:
                data = aoi.filter_bbox(data)

        if needs_reprojection:
            if CRS.from_user_input(data.rio.crs) != CRS.from_user_input(target_crs):
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
            if aoi is not None:
                data = aoi.filter_bbox(data)

        if data.isnull().any().compute().item():
            raise ValueError('BaseGrid cannot contain NaN values. Check reprojection and aoi settings. Set higher value for aoi_buffer_m?')

        if cache_path is not None:
            cache_path.parent.mkdir(parents = True, exist_ok = True)
            data = self._attach_crs_metadata(data, target_crs)
            data.to_zarr(cache_path, zarr_format=2)
            logger.info(f"Base grid cache written to {cache_path}")

        self.data = data
        self.crs = self._crs_identifier(target_crs)
        self.res = target_res if target_res is not None else original_res
        self.x_dim = target_x_dim
        self.y_dim = target_y_dim

    @staticmethod
    def _crs_identifier(crs):
        crs = CRS.from_user_input(crs)
        return crs.to_epsg() or crs.to_string()

    @staticmethod
    def _metadata_crs(data: xr.DataArray | xr.Dataset) -> CRS | None:
        for attr_name in (_CRS_ATTR, _CRS_EPSG_ATTR):
            value = data.attrs.get(attr_name)
            if value is None:
                continue
            try:
                return CRS.from_user_input(value)
            except Exception:
                logger.warning("Ignoring invalid CRS metadata %s=%s", attr_name, value)

        spatial_ref = data.coords.get("spatial_ref")
        if spatial_ref is not None:
            for attr_name in ("crs_wkt", "spatial_ref"):
                value = spatial_ref.attrs.get(attr_name)
                if value is None:
                    continue
                try:
                    return CRS.from_user_input(value)
                except Exception:
                    logger.warning("Ignoring invalid spatial_ref metadata %s", attr_name)

        return None

    @classmethod
    def _restore_crs_from_metadata(cls, data: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
        if data.rio.crs is not None:
            return data

        crs = cls._metadata_crs(data)
        if crs is None:
            return data

        return data.rio.write_crs(crs, inplace=False)

    @classmethod
    def _ensure_loaded_crs(
        cls,
        data: xr.DataArray | xr.Dataset,
        source: str | Path,
    ) -> xr.DataArray | xr.Dataset:
        data = cls._restore_crs_from_metadata(data)
        if data.rio.crs is None:
            raise ValueError(
                f"Base grid loaded from {source} does not define a CRS. "
                "Provide a CRS in the source file or rebuild the cache with CRS metadata."
            )
        return data

    @staticmethod
    def _attach_crs_metadata(data: xr.DataArray | xr.Dataset, crs) -> xr.DataArray | xr.Dataset:
        crs = CRS.from_user_input(crs)
        data = data.rio.write_crs(crs, inplace=False)
        data.attrs[_CRS_ATTR] = crs.to_string()
        epsg = crs.to_epsg()
        if epsg is not None:
            data.attrs[_CRS_EPSG_ATTR] = f"EPSG:{epsg}"
        return data

    def _cache_path(
        self,
        path: Path,
        target_crs: int | None,
        target_res: int | float | tuple[float, float] | None,
        target_x_dim: str,
        target_y_dim: str,
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

        key_payload = {
            "source_path": str(Path(path).resolve()),
            "var": var,
            "squeeze": squeeze,
            "target_crs": target_crs,
            "target_res": target_res,
            "target_x_dim": target_x_dim,
            "target_y_dim": target_y_dim,
            "crs": crs,
            "x_dim": x_dim,
            "y_dim": y_dim,
            "aoi": aoi_info,
            "aoi_buffer_m": aoi_buffer_m,
            "resampling_method": resampling_method,
        }

        key_json = json.dumps(key_payload, sort_keys = True, default = str)
        key_hash = hashlib.sha256(key_json.encode("utf-8")).hexdigest()[:16]
        return Path("data/cache") / f"{Path(path).stem}.{key_hash}.zarr"

    def open(self, path: str, var: str | None = None, squeeze: bool = True):

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
