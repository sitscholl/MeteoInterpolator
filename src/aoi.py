from dataclasses import dataclass
from pyproj import CRS, Transformer
import rioxarray
import xarray as xr

@dataclass(frozen=True)
class AOI:
    minx: float
    miny: float
    maxx: float
    maxy: float
    crs: int = 4326  # canonical CRS for config bbox

    def _to_crs(self, dst_crs: int):
        if dst_crs == self.crs:
            return self.minx, self.miny, self.maxx, self.maxy
        transformer = Transformer.from_crs(self.crs, dst_crs, always_xy=True)
        x1, y1 = transformer.transform(self.minx, self.miny)
        x2, y2 = transformer.transform(self.maxx, self.maxy)
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    def _buffer_bbox_in_meters(self, dst_crs: int, buffer_m: int | float):
        minx, miny, maxx, maxy = self._to_crs(dst_crs)
        crs = CRS.from_epsg(dst_crs)
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

    def filter_bbox(self, data: xr.DataArray | xr.Dataset, buffer_m: int | float | None = None):
        if data.rio.crs is None:
            raise ValueError("Data has no CRS; cannot apply AOI")

        dst_epsg = data.rio.crs.to_epsg()

        if dst_epsg is None:
            raise ValueError("Data CRS has no EPSG code; cannot apply AOI")

        if buffer_m is None or buffer_m == 0:
            minx, miny, maxx, maxy = self._to_crs(dst_epsg)
        else:
            minx, miny, maxx, maxy = self._buffer_bbox_in_meters(dst_epsg, buffer_m)
        
        return data.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

    @property
    def bounds(self):
        return f"{self.minx}, {self.miny}, {self.maxx}, {self.maxy}"
