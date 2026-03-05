from abc import ABC, abstractmethod
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

    @abstractmethod
    def write(self, data: xr.DataArray | xr.Dataset, overwrite: bool = False) -> None:
        pass