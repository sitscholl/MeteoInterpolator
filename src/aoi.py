from dataclasses import dataclass
from pyproj import Transformer
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

    def filter_bbox(self, data: xr.DataArray | xr.Dataset):
        if data.rio.crs is None:
            raise ValueError("Data has no CRS; cannot apply AOI")
        dst_epsg = data.rio.crs.to_epsg()
        if dst_epsg is None:
            raise ValueError("Data CRS has no EPSG code; cannot apply AOI")
        minx, miny, maxx, maxy = self._to_crs(dst_epsg)
        return data.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
