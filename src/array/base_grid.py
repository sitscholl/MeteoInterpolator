from pathlib import Path
import xarray as xr
import rioxarray

import logging

from .aoi import AOI

logger = logging.getLogger(__name__)

class BaseGrid:

    def __init__(
        self, 
        path: str | Path, 
        crs: int | None = None, 
        res: int | None = None, 
        aoi: AOI | None = None,
        resampling_method: str = 'bilinear',
        **kwargs
        ):
        
        data = self.open(path, **kwargs)

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

        original_res = data.rio.resolution()[0]
        if res is not None:
            if original_res != res:
                needs_reprojection = True
        else:
            res = original_res

        if needs_reprojection:
            if data.rio.crs.to_epsg() != crs:
                logger.debug(f"Reprojecting base grid from crs {data.rio.crs} to {crs}")
            if original_res != res:
                logger.debug(f"Reprojecting base grid from resolution of {original_res} to {res}")
            data = data.rio.reproject(dst_crs = crs, resolution = res, resampling = resampling_method)

        if aoi is not None:
            data = aoi.filter_bbox(data)

        self.data = data

    def open(self, path, var: str | None = None, squeeze: bool = True):
        data = xr.open_dataset(path)
        
        available_vars = list(data.keys())
        if var is not None:
            if var not in available_vars:
                raise ValueError(f"Data variable {var} not found in dataset. Available variables: {available_vars}")
            data = data[var]
        else:
            if len(available_vars) > 1:
                logger.warning(f"Found multiple variables in base grid: {available_vars}. Picking fist one. ")
            data = data[available_vars[0]]

        if squeeze:
            data = data.squeeze(drop = True)

        return data