import xarray as xr

from pathlib import Path
import re
import pandas as pd

from .base import GridWriter
from ...utils import get_date_format_from_freq

class TiffWriter(GridWriter):
    
    def __init__(
        self,
        root: str | Path,
        filename_pattern: str = "",
        date_format: str | None = None,
        _date_format_explicit: bool | None = None,
    ):
        self.root = Path(root)
        self.filename_pattern = filename_pattern or "{var}_{date}"
        self.date_format = date_format or "%Y_%m_%d_%H%M%S"
        self._date_format_explicit = date_format is not None if _date_format_explicit is None else _date_format_explicit

    @classmethod
    def key(cls):
        return 'tiff'

    def initialize(
        self,
        param: str | None = None,
        start=None,
        end=None,
        freq: str | None = None,
        **kwargs,
    ):
        date_format = self.date_format
        date_format_explicit = self._date_format_explicit
        if kwargs.get("date_format") is not None:
            date_format = kwargs["date_format"]
            date_format_explicit = True
        elif freq is not None and not self._date_format_explicit:
            date_format = get_date_format_from_freq(freq, filename_safe=True)
        return type(self)(
            root=self.root,
            filename_pattern=self.filename_pattern,
            date_format=date_format,
            _date_format_explicit=date_format_explicit,
        )

    @staticmethod
    def _clean_filename_stem(value: str) -> str:
        value = re.sub(r"[\s:]+", "_", value.strip())
        value = re.sub(r"_+", "_", value)
        return value.strip("_-.")

    def _filename_stem(
        self,
        var: str,
        timestamp=None,
        param: str | None = None,
        suffix: str | None = None,
    ) -> str:
        date = ""
        if timestamp is not None:
            date = pd.Timestamp(timestamp).strftime(self.date_format)
        try:
            raw = self.filename_pattern.format(
                var=var,
                variable=var,
                date=date,
                param=param or var,
                suffix=suffix or "",
            )
        except KeyError as exc:
            raise ValueError(
                "Unknown filename_pattern field. Supported fields are "
                "{var}, {variable}, {date}, {param}, and {suffix}."
            ) from exc
        stem = self._clean_filename_stem(raw)
        if not stem:
            raise ValueError("Filename pattern produced an empty filename.")
        return stem

    def _output_path(
        self,
        var: str,
        timestamp=None,
        param: str | None = None,
        suffix: str | None = None,
    ) -> Path:
        return self.root / f"{self._filename_stem(var, timestamp, param, suffix)}.tif"

    def _to_tiff(self, arr: xr.DataArray, out_path: str, overwrite: bool = False):
        
        if Path(out_path).exists() and not overwrite:
            raise ValueError(f"File {out_path} already exists and overwrite set to false")
        if Path(out_path).exists() and overwrite:
            Path(out_path).unlink()

        arr.rio.to_raster(str(out_path))

    def write(
        self,
        data: xr.Dataset | xr.DataArray,
        overwrite: bool = False,
        *,
        param: str | None = None,
        timestamp=None,
        suffix: str | None = None,
    ):

        if param is not None or timestamp is not None or suffix is not None:
            if param is None or timestamp is None:
                raise ValueError("param and timestamp are required when preparing TIFF output in write().")
            data = self.prepare_grid_for_output(data, param, timestamp, suffix)

        if isinstance(data, xr.DataArray):
            if data.name is None:
                data = data.rename("var")
            data = data.to_dataset()

        self.root.mkdir(parents=True, exist_ok=True)

        for var, da in data.data_vars.items():

            if 'time' in da.dims:
                for ts_coord, arr in da.groupby('time'):
                    out_path = self._output_path(var, ts_coord, param=param, suffix=suffix)
                    self._to_tiff(arr, out_path, overwrite = overwrite)
            else:
                out_path = self._output_path(var, param=param, suffix=suffix)
                self._to_tiff(da, out_path, overwrite = overwrite)
