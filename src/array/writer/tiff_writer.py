import xarray as xr

from pathlib import Path

from .base import GridWriter

class TiffWriter(GridWriter):
    
    def __init__(
        self,
        path: str | Path
    ):
        self.path = Path(path)

    def _to_tiff(self, arr: xr.DataArray, out_path: str, overwrite: bool = False):
        
        if Path(out_path).exists() and not overwrite:
            raise ValueError(f"File {out_path} already exists and overwrite set to false")
        if Path(out_path).exists() and overwrite:
            Path(out_path).unlink()

        arr.rio.to_raster(str(out_path))

    def write(self, data: xr.Dataset | xr.DataArray, overwrite: bool = False):

        if isinstance(data, xr.DataArray):
            if data.name is None:
                data = data.rename("var")
            data = data.to_dataset()

        self.path.mkdir(parents=True, exist_ok=True)

        for var, da in data.data_vars.items():

            if 'time' in da.dims:
                for ts_coord, arr in da.groupby('time'):
                    out_path = f"{self.path}/{var}_{ts_coord:%Y_%m_%d_%H%M%S}.tif"
                    self._to_tiff(arr, out_path, overwrite = overwrite)
            else:
                out_path = f"{self.path}/{var}.tif"
                self._to_tiff(da, out_path, overwrite = overwrite)
