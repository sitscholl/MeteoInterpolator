from abc import ABC, abstractmethod
import pandas as pd
import xarray as xr

class GridWriter(ABC):

    registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if cls is GridWriter:
            return

        class_key = cls.key()
        if class_key in GridWriter.registry and GridWriter.registry[class_key] is not cls:
            raise ValueError(f"Duplicate class name on intialization: {class_key}")
        
        GridWriter.registry[class_key] = cls

    @classmethod
    @abstractmethod
    def key(cls):
        return 'GridWriter'

    @classmethod
    def create(cls, key: str, **kwargs):
        writer_cls = cls.registry.get(key)
        if writer_cls is None:
            raise ValueError(f"Unknown key {key} for output format. Choose one of {cls.registry.keys()}")
        return writer_cls(**kwargs)

    def initialize(self, **kwargs):
        return self

    @staticmethod
    def output_var_name(param: str, suffix: str | None = None) -> str:
        parts = [str(param).strip()]
        if suffix is not None:
            parts.append(str(suffix).strip().strip("_"))
        return "_".join(part for part in parts if part)

    @classmethod
    def prepare_grid_for_output(
        cls,
        interpolated_grid: xr.DataArray | xr.Dataset,
        param: str,
        interp_date,
        suffix: str | None = None,
    ) -> xr.Dataset | xr.DataArray:
        new_name = cls.output_var_name(param, suffix)
        if isinstance(interpolated_grid, xr.Dataset):
            data = interpolated_grid
            data_vars = list(data.data_vars)
            if len(data_vars) == 1 and data_vars[0] != new_name:
                old_name = data_vars[0]
                data = data.rename({old_name: new_name})
            elif param in data.data_vars and param != new_name:
                if new_name in data.data_vars:
                    raise ValueError(f"Cannot rename output variable {param!r} to existing variable {new_name!r}.")
                data = data.rename({param: new_name})
        elif isinstance(interpolated_grid, xr.DataArray):
            data = interpolated_grid.rename(new_name)
        else:
            raise TypeError(f"Interpolated grid must be an xarray DataArray or Dataset. Got {type(interpolated_grid)}")

        if "time" not in data.dims:
            data = data.expand_dims(time=[pd.Timestamp(interp_date)])
        else:
            data = data.assign_coords(time=[pd.Timestamp(interp_date)])
        return data

    @abstractmethod
    def write(
        self,
        data: xr.DataArray | xr.Dataset,
        overwrite: bool = False,
        *,
        param: str | None = None,
        timestamp=None,
        suffix: str | None = None,
    ) -> None:
        pass
