from abc import ABC, abstractmethod
import xarray as xr

class GridWriter(ABC):

    @abstractmethod
    def write(self, data: xr.DataArray | xr.Dataset, overwrite: bool = False) -> None:
        pass