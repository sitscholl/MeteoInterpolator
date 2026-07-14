from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import ClassVar, Sequence, Hashable
import logging

import numpy as np
import xarray as xr

from ...domain.dem import DEM
from ...array.cache import CacheManager

logger = logging.getLogger(__name__)
_CACHE_KEY = "distance_field"

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

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset) -> "DistanceField":
        data_vars = list(dataset.data_vars)
        if len(data_vars) != 1:
            raise ValueError(f"Expected exactly one cached distance variable. Got {data_vars}")
        data = dataset[data_vars[0]]
        distance_type = dataset.attrs.get("distance_type", data.attrs.get("distance_type", data.name))
        return cls(distance_type=distance_type, data=data)

    def to_dataset(self) -> xr.Dataset:
        if self.data is None:
            raise ValueError("Cannot convert an empty DistanceField to a Dataset.")
        data_name = self.data.name or self.distance_type or "distance"
        dataset = self.data.to_dataset(name=data_name)
        dataset.attrs["distance_type"] = self.distance_type
        return dataset

    def to_points(
        self,
        target_ids: Sequence[Hashable],
        x_coords: Sequence[float],
        y_coords: Sequence[float],
        method: str | None = "nearest",
    ) -> xr.DataArray:
        if self.data is None:
            raise ValueError("Cannot sample an empty DistanceField.")

        target_ids = [str(target_id) for target_id in target_ids]
        x_coords = list(x_coords)
        y_coords = list(y_coords)
        if len(target_ids) != len(x_coords) or len(target_ids) != len(y_coords):
            raise ValueError(
                "target_ids, x_coords, and y_coords must have the same length. "
                f"Got {len(target_ids)}, {len(x_coords)}, and {len(y_coords)}."
            )
        if len(target_ids) == 0:
            raise ValueError("At least one target point is required.")
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("Target point ids must be unique.")

        x_values = self.data.coords["x"].values
        y_values = self.data.coords["y"].values
        invalid_ids = [
            target_id
            for target_id, x, y in zip(target_ids, x_coords, y_coords)
            if not (min(x_values) <= x <= max(x_values) and min(y_values) <= y <= max(y_values))
        ]
        if invalid_ids:
            raise ValueError(
                "Target points are outside the distance field extent. "
                f"Check CRS and field extent for ids: {invalid_ids}"
            )

        target_coord = xr.DataArray(
            target_ids,
            dims=("target_id",),
            coords={"target_id": target_ids},
        )
        x_indexer = xr.DataArray(
            np.asarray(x_coords, dtype=float),
            dims=("target_id",),
            coords={"target_id": target_coord},
        )
        y_indexer = xr.DataArray(
            np.asarray(y_coords, dtype=float),
            dims=("target_id",),
            coords={"target_id": target_coord},
        )

        point_distances = self.data.sel(x=x_indexer, y=y_indexer, method=method)
        return point_distances.assign_coords(
            {
                "target_id": target_ids,
                "target_x": ("target_id", np.asarray(x_coords, dtype=float)),
                "target_y": ("target_id", np.asarray(y_coords, dtype=float)),
            }
        )

    def __post_init__(self):
        ##make sure the structure of the dataArray corresponds to a fixed schema
        if self.data is None:
            return
        missing_dims = [dim for dim in self._required_dims if dim not in self.data.dims]
        if missing_dims:
            raise ValueError(f"Missing required dims {missing_dims}. Got {self.data.dims}")
        missing_coords = [dim for dim in self._required_dims if dim not in self.data.coords]
        if missing_coords:
            raise ValueError(f"Missing required coords {missing_coords}. Got {list(self.data.coords)}")
        if self.data.sizes["id"] == 0:
            raise ValueError("Distance fields require at least one source id.")
        ids = list(self.data.coords["id"].values)
        if len(ids) != len(set(ids)):
            raise ValueError("Distance field source ids must be unique.")
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
        ...

    @classmethod
    def create(cls, key: str, **kwargs):
        model_cls = cls.registry.get(key)
        if model_cls is None:
            available = ", ".join(sorted(cls.registry)) or "none"
            raise ValueError(f"Unknown distance calculator '{key}'. Available: {available}")
        return model_cls(**kwargs)

    def __init__(self, cache_manager: CacheManager | None = None):
        self.cache_manager = cache_manager

    def cache_parameters(self) -> dict:
        return {}

    def _distance_cache_parameters(
        self,
        dem: DEM,
        x_coords: Sequence[float],
        y_coords: Sequence[float],
        point_ids: Sequence[Hashable],
    ) -> dict:
        return {
            "cache_schema_version": 1,
            "calculator": self.key(),
            "calculator_parameters": self.cache_parameters(),
            "dem_fingerprint": dem.fingerprint,
            "sources": [
                {"id": str(pid), "x": float(x), "y": float(y)}
                for x, y, pid in zip(x_coords, y_coords, point_ids)
            ],
        }

    def _load_cache(self, cache_params: dict) -> DistanceField | None:
        if self.cache_manager is None:
            return None

        dataset = self.cache_manager.load_cache(_CACHE_KEY, cache_params)
        if dataset is not None:
            try:
                return DistanceField.from_dataset(dataset)
            except Exception as e:
                logger.warning(
                    "Failed to load distance cache with error: %s. Deleting cache and calculating new distance fields",
                    e,
                )
                self.cache_manager.delete_cache(_CACHE_KEY, cache_params)
        return None

    def _initialize_cache(self, distance_fields: DistanceField, cache_params: dict):
        if self.cache_manager is None:
            return False
        try:
            self.cache_manager.initialize_cache(
                distance_fields.to_dataset(),
                key=_CACHE_KEY,
                cache_params=cache_params,
            )
            return True
        except Exception as e:
            logger.warning("Failed to initialize distance cache with error: %s", e)
            return False

    def _validate_point_in_grid(self, x: float, y:float, grid: xr.DataArray) -> bool:
        x_values = grid.coords["x"].values
        y_values = grid.coords["y"].values
        return (
            min(x_values) <= x <= max(x_values)
            and min(y_values) <= y <= max(y_values)
        )

    @staticmethod
    def _validate_dem_data(dem: xr.DataArray) -> xr.DataArray:
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
        if len(x) == 0 or len(y) == 0:
            raise ValueError("DEM x/y coordinates must not be empty.")

        return dem

    @classmethod
    def _validate_dem(cls, dem: DEM) -> xr.DataArray:
        if not isinstance(dem, DEM):
            raise TypeError(
                f"dem should be a DEM. Got {type(dem)}"
            )
        return cls._validate_dem_data(dem.data)

    @staticmethod
    def _validate_source_points(
        x_coords: Sequence[float],
        y_coords: Sequence[float],
        point_ids: Sequence[Hashable] | None = None,
    ) -> tuple[list[float], list[float], list[Hashable] | None]:
        x_coords = list(x_coords)
        y_coords = list(y_coords)
        if len(x_coords) != len(y_coords):
            raise ValueError(
                f"The number of x and y coordinates must match. Got {len(x_coords)} vs {len(y_coords)}"
            )
        if len(x_coords) == 0:
            raise ValueError("At least one source coordinate is required.")
        if point_ids is None:
            return x_coords, y_coords, None

        point_ids = list(point_ids)
        if len(point_ids) != len(x_coords):
            raise ValueError(
                f"If supplied, the number of ids must match the number of points. Got {len(point_ids)} vs {len(x_coords)}"
            )
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("Source point ids must be unique.")
        return x_coords, y_coords, point_ids

    @abstractmethod
    def calculate_distance(
        self,
        dem: DEM,
        x_coords: Sequence[float],
        y_coords: Sequence[float],
        point_ids: Sequence[Hashable] | None = None,
    ) -> DistanceField:
        ...

    def calculate_fields(
        self,
        dem: DEM,
        x_coords: Sequence[float],
        y_coords: Sequence[float],
        point_ids: Sequence[Hashable],
    ) -> DistanceField:
        dem_data = self._validate_dem(dem)
        x_coords, y_coords, point_ids = self._validate_source_points(x_coords, y_coords, point_ids)
        point_ids = [str(i) for i in point_ids]

        distance_params = self._distance_cache_parameters(dem, x_coords, y_coords, point_ids)
        distance_fields = self._load_cache(distance_params)

        if distance_fields is None:
            invalid_points = []
            for x, y, pid in zip(x_coords, y_coords, point_ids):
                point_in_grid = self._validate_point_in_grid(x, y, dem_data)
                if not point_in_grid:
                    invalid_points.append(pid)
            if invalid_points:
                raise ValueError(
                    "Source points are outside the supplied DEM extent. "
                    f"Check CRS and DEM extent for ids: {invalid_points}"
                )

            distance_fields = self.calculate_distance(dem, x_coords, y_coords, point_ids)
            self._initialize_cache(distance_fields, distance_params)

        return distance_fields
