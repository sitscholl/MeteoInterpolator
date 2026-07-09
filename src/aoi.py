import rioxarray
import xarray as xr
import geopandas as gpd

from dataclasses import dataclass
from pyproj import CRS, Transformer

@dataclass(frozen=True)
class AOI:
    minx: float
    miny: float
    maxx: float
    maxy: float
    crs: CRS | int | str = 4326

    @classmethod
    def from_array(cls, array: xr.DataArray | xr.Dataset):
        minx, miny, maxx, maxy = array.rio.bounds()
        crs = array.rio.crs

        if crs is None:
            raise ValueError("Cannot create AOI instance from array without crs.")
        
        crs = CRS.from_user_input(crs)

        if crs is None:
            raise ValueError(f"Unable to get CRS object form array crs. Array crs: {array.rio.crs}")

        return cls(
            minx, miny, maxx, maxy, crs
        )


    def _to_crs(self, dst_crs):
        src_crs = CRS.from_user_input(self.crs)
        dst_crs = CRS.from_user_input(dst_crs)
        if dst_crs == src_crs:
            return self.minx, self.miny, self.maxx, self.maxy
        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        x1, y1 = transformer.transform(self.minx, self.miny)
        x2, y2 = transformer.transform(self.maxx, self.maxy)
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    def _buffer_bbox_in_meters(self, dst_crs, buffer_m: int | float):
        minx, miny, maxx, maxy = self._to_crs(dst_crs)
        crs = CRS.from_user_input(dst_crs)
        if crs.is_projected:
            return minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m

        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2
        local_aeqd = CRS.from_proj4(
            f"+proj=aeqd +lat_0={center_y} +lon_0={center_x} +datum=WGS84 +units=m +no_defs"
        )
        to_local = Transformer.from_crs(crs, local_aeqd, always_xy=True)
        from_local = Transformer.from_crs(local_aeqd, crs, always_xy=True)

        x1, y1 = to_local.transform(minx, miny)
        x2, y2 = to_local.transform(maxx, maxy)
        minx_l = min(x1, x2) - buffer_m
        maxx_l = max(x1, x2) + buffer_m
        miny_l = min(y1, y2) - buffer_m
        maxy_l = max(y1, y2) + buffer_m

        bx1, by1 = from_local.transform(minx_l, miny_l)
        bx2, by2 = from_local.transform(maxx_l, maxy_l)
        return min(bx1, bx2), min(by1, by2), max(bx1, bx2), max(by1, by2)

    def filter_bbox(self, data: xr.DataArray | xr.Dataset | gpd.GeoDataFrame, buffer_m: int | float | None = None):
        if isinstance(data, (xr.DataArray, xr.Dataset)):
            return self._filter_bbox_xarray(data, buffer_m)
        elif isinstance(data, gpd.GeoDataFrame):
            return self._filter_bbox_gdf(data, buffer_m)
        else:
            raise ValueError(f"Unsupported data type {type(data)} for filter_bbox.")

    def _filter_bbox_xarray(self, data: xr.DataArray | xr.Dataset, buffer_m: int | float | None = None):
        if data.rio.crs is None:
            raise ValueError("Xarray data has no CRS; cannot apply AOI")

        dst_crs = CRS.from_user_input(data.rio.crs)

        if buffer_m is None or buffer_m == 0:
            minx, miny, maxx, maxy = self._to_crs(dst_crs)
        else:
            minx, miny, maxx, maxy = self._buffer_bbox_in_meters(dst_crs, buffer_m)
        
        return data.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

    def _filter_bbox_gdf(self, data: gpd.GeoDataFrame, buffer_m: int | float | None = None):
        if data.crs is None:
            raise ValueError("Geodataframe has no crs; cannot apply AOI")

        dst_crs = CRS.from_user_input(data.crs)

        if buffer_m is None or buffer_m == 0:
            minx, miny, maxx, maxy = self._to_crs(dst_crs)
        else:
            minx, miny, maxx, maxy = self._buffer_bbox_in_meters(dst_crs, buffer_m)
        
        return data.cx[minx:maxx, miny:maxy]

    @property
    def bounds(self):
        return (self.minx, self.miny, self.maxx, self.maxy)
