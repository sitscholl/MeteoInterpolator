from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import ClassVar, Sequence, Hashable
from pathlib import Path
import logging
from shutil import rmtree

import xarray as xr

logger = logging.getLogger(__name__)

@dataclass
class DistanceField:
    distance_type: str
    data: xr.DataArray | None
    _required_dims: ClassVar[tuple[str, ...]] = ("id", "y", "x")

    @classmethod
    def from_list(cls, lst: list["DistanceField"]):
        fields = [i for i in lst if i.data is not None]
        if not fields:
            raise ValueError("Cannot create a DistanceField from an empty list.")
        distance_types = {i.distance_type for i in fields}
        if len(distance_types) != 1:
            raise ValueError(f"Cannot combine distance fields with different types: {distance_types}")

        data = xr.concat([i.data for i in fields], dim = 'id')
        return cls(
            distance_type = fields[0].distance_type,
            data = data
        )

    def __post_init__(self):
        ##make sure the structure of the dataArray corresponds to a fixed schema
        if self.data is not None and not all([i in self.data.dims for i in self._required_dims]):
            raise ValueError(f"Not all required dims {self._required_dims} found in data. Got {self.data.dims}")
        if self.data is not None:
            self.data.attrs.setdefault("distance_type", self.distance_type)

class BaseDistanceCalculator(ABC):
    registry: dict[str, type["BaseDistanceCalculator"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls is not BaseDistanceCalculator:
            BaseDistanceCalculator.registry[cls.key()] = cls
    
    @classmethod
    @abstractmethod
    def key(cls) -> str:
        pass

    @classmethod
    def create(cls, key: str, **kwargs):
        model_cls = cls.registry.get(key)
        if model_cls is None:
            available = ", ".join(sorted(cls.registry)) or "none"
            raise ValueError(f"Unknown residual model '{key}'. Available: {available}")
        return model_cls(**kwargs)

    def __init__(self, cache_directory: str | Path | None = None):
        if cache_directory is not None:
            cache_directory = Path(cache_directory)
            cache_directory.parent.mkdir(exist_ok = True, parents = True)
            logger.debug(f"Distance fields will be cached at {cache_directory}")
        self.cache_directory = cache_directory

    def _build_cache_path(self, distance_params: dict | None = None):
        if self.cache_directory is None:
            return None
        return None

    def _initialize_cache(self, distance_fields: DistanceField, cache_path: Path | None = None):
        if cache_path is not None:

            if cache_path.exists():
                raise ValueError(f"Found already existing cache at {cache_path}")
            if distance_fields.data is None:
                logger.warning("Cannot initialize distance cache from an empty DistanceField.")
                return False

            try:
                logger.debug(f"Initializing new distance cache at {cache_path}")
                data_name = distance_fields.data.name or "distance"
                dataset = distance_fields.data.to_dataset(name=data_name)
                dataset.attrs["distance_type"] = distance_fields.distance_type
                dataset.to_zarr(cache_path)
                return True
            except Exception as e:
                logger.warning(f"Failed to initialize distance cache at {cache_path} with error: {e}")
                return False

    def _load_cache(self, cache_path: Path | None = None):
        if cache_path is not None and cache_path.exists():
            try:
                logger.info(f"Loading existing distance cache at {cache_path}")
                dataset = xr.open_zarr(cache_path)
                data_vars = list(dataset.data_vars)
                if len(data_vars) != 1:
                    raise ValueError(f"Expected exactly one cached distance variable. Got {data_vars}")
                data = dataset[data_vars[0]]
                distance_type = dataset.attrs.get("distance_type", data.attrs.get("distance_type", data.name))
                return DistanceField(distance_type = distance_type, data = data)
            except Exception as e:
                logger.warning(f"Failed to load distance cache with error: {e}. Deleting cache and calculating new distance fields")
                rmtree(cache_path)
        return None

    def _validate_point_in_grid(self, x: float, y:float, grid: xr.DataArray) -> bool:
        grid = self._validate_dem(grid)
        x_values = grid.coords["x"].values
        y_values = grid.coords["y"].values
        return (
            min(x_values) <= x <= max(x_values)
            and min(y_values) <= y <= max(y_values)
        )

    @staticmethod
    def _validate_dem(dem: xr.DataArray) -> xr.DataArray:
        if not isinstance(dem, xr.DataArray):
            raise TypeError(
                f"dem should be a DataArray. Got {type(dem)}"
            )
        if "y" not in dem.dims:
            raise ValueError(
                f"Missing y dimension 'y'. Make sure the dem has the vertical dimension named y. Got {dem.dims}"
            )
        if "x" not in dem.dims:
            raise ValueError(
                f"Missing x dimension 'x'. Make sure the dem has the horizontal dimension named x. Got {dem.dims}"
            )
        if dem.ndim != 2:
            raise ValueError(f"Expected a 2D DEM with dimensions ('y', 'x'). Got shape {dem.shape}")

        dem = dem.transpose("y", "x")

        x = dem.coords["x"].values
        y = dem.coords["y"].values
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("Only rectilinear DEM grids with 1D x and y coordinates are supported.")
        if len(x) != dem.sizes["x"] or len(y) != dem.sizes["y"]:
            raise ValueError("DEM x/y coordinate lengths do not match the DEM dimensions.")

        return dem

    @abstractmethod
    def calculate_distance(
        self,
        dem: xr.DataArray,
        x_coords: Sequence[float],
        y_coords: Sequence[float],
        point_ids: Sequence[Hashable] | None = None,
    ) -> DistanceField:
        ...

    def calculate_fields(
        self,
        dem: xr.DataArray,
        x_coords: Sequence[float],
        y_coords: Sequence[float],
        point_ids: Sequence[Hashable],
    ) -> DistanceField | None:

        point_ids = [str(i) for i in point_ids]

        cache_path = self._build_cache_path()
        distance_fields = self._load_cache(cache_path)

        if distance_fields is None:
            for x, y, pid in zip(x_coords, y_coords, point_ids):
                point_in_grid = self._validate_point_in_grid(x, y, dem)
                if not point_in_grid:
                    logger.warning(f"Point with id {pid} is not within supplied dem. Check crs and dem extent.")

            distance_fields = self.calculate_distance(dem, x_coords, y_coords, point_ids)
            self._initialize_cache(distance_fields, cache_path)

        return distance_fields
