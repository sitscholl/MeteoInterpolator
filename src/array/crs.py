import xarray as xr
from pyproj import CRS

import logging

logger = logging.getLogger(__name__)

_CRS_ATTR = "crs"
_CRS_EPSG_ATTR = "crs_epsg"

def attach_crs_metadata(self, data: xr.DataArray | xr.Dataset, crs) -> xr.DataArray | xr.Dataset:
    crs = CRS.from_user_input(crs)
    data = data.rio.write_crs(crs, inplace=False)
    data.attrs[_CRS_ATTR] = crs.to_string()
    epsg = crs.to_epsg()
    if epsg is not None:
        data.attrs[_CRS_EPSG_ATTR] = f"EPSG:{epsg}"
    return data

def load_crs_metadata(self, data: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
    if data.rio.crs is not None:
        return data

    crs = None
    for attr_name in (_CRS_ATTR, _CRS_EPSG_ATTR):
        value = data.attrs.get(attr_name)
        if value is None:
            continue
        try:
            crs = CRS.from_user_input(value)
        except Exception:
            logger.warning("Ignoring invalid CRS metadata %s=%s", attr_name, value)

    spatial_ref = data.coords.get("spatial_ref")
    if spatial_ref is not None:
        for attr_name in ("crs_wkt", "spatial_ref"):
            value = spatial_ref.attrs.get(attr_name)
            if value is None:
                continue
            try:
                crs = CRS.from_user_input(value)
            except Exception:
                logger.warning("Ignoring invalid spatial_ref metadata %s", attr_name)

    if crs is None:
        return data

    return data.rio.write_crs(crs, inplace=False)