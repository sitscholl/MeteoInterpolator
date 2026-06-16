import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from pyproj import CRS, Transformer

from src.aoi import AOI


def test_aoi_filter_accepts_xarray_crs_without_epsg_code():
    aoi = AOI(minx=10, miny=46, maxx=12, maxy=47)
    crs = CRS.from_proj4("+proj=aeqd +lat_0=46.5 +lon_0=11 +datum=WGS84 +units=m +no_defs")
    assert crs.to_epsg() is None

    transformer = Transformer.from_crs(aoi.crs, crs, always_xy=True)
    x1, y1 = transformer.transform(aoi.minx, aoi.miny)
    x2, y2 = transformer.transform(aoi.maxx, aoi.maxy)
    minx, maxx = sorted((x1, x2))
    miny, maxy = sorted((y1, y2))

    data = xr.DataArray(
        np.ones((5, 5), dtype=float),
        dims=("y", "x"),
        coords={
            "y": np.linspace(miny - 1000, maxy + 1000, 5),
            "x": np.linspace(minx - 1000, maxx + 1000, 5),
        },
        name="orog",
    ).rio.write_crs(crs)

    clipped = aoi.filter_bbox(data)

    assert clipped.sizes["x"] > 0
    assert clipped.sizes["y"] > 0
