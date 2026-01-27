import math
from typing import Iterable, Tuple

import numpy as np
import rioxarray
import xarray as xr

class GridValidator:

    def __init__(
        self,
        crs: str,
        res: Tuple[float, float],
        x_dim: str,
        y_dim: str,
        x_coords: Iterable[float],
        y_coords: Iterable[float],
    ):
        self.crs = crs
        self.res = (float(res[0]), float(res[1]))
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.x_coords = np.asarray(x_coords)
        self.y_coords = np.asarray(y_coords)

    @classmethod
    def from_grid(cls, grid: xr.DataArray | xr.Dataset, x_dim: str, y_dim: str):

        if grid.rio.crs is None:
            raise ValueError("Provided grid has no crs to read from")
        crs = grid.rio.crs
        res = grid.rio.resolution()
        if x_dim not in grid.dims or y_dim not in grid.dims:
            raise ValueError(f"Provided grid missing required dims '{x_dim}' and/or '{y_dim}'. Dims: {list(grid.dims)}")
        x_coords = grid[x_dim].values
        y_coords = grid[y_dim].values

        return GridValidator(
            crs = crs,
            res = res,
            x_dim = x_dim,
            y_dim = y_dim,
            x_coords = x_coords,
            y_coords = y_coords
        )

    def validate(self, other: xr.DataArray | xr.Dataset):
        """Make sure has the same grid specifications as self"""
        if other.rio.crs is None:
            raise ValueError("Other grid has no CRS.")

        other_crs = other.rio.crs
        if other_crs != self.crs:
            raise ValueError(f"CRS mismatch. Expected {self.crs}, got {other_crs}.")

        if self.x_dim not in other.dims or self.y_dim not in other.dims:
            raise ValueError(
                f"Dimension mismatch. Expected dims '{self.x_dim}' and '{self.y_dim}', got {list(other.dims)}."
            )

        other_res = other.rio.resolution()
        if not (
            math.isclose(self.res[0], other_res[0], rel_tol=0, abs_tol=1e-9)
            and math.isclose(self.res[1], other_res[1], rel_tol=0, abs_tol=1e-9)
        ):
            raise ValueError(f"Resolution mismatch. Expected {self.res}, got {other_res}.")

        other_x = np.asarray(other[self.x_dim].values)
        other_y = np.asarray(other[self.y_dim].values)

        if self.x_coords.shape != other_x.shape or self.y_coords.shape != other_y.shape:
            raise ValueError(
                f"Coordinate shape mismatch. Expected x/y shapes {self.x_coords.shape}/{self.y_coords.shape}, "
                f"got {other_x.shape}/{other_y.shape}."
            )

        if not np.allclose(self.x_coords, other_x, rtol=0, atol=1e-9) or not np.allclose(
            self.y_coords, other_y, rtol=0, atol=1e-9
        ):
            raise ValueError("Coordinate mismatch. Grid extents or spacing differ.")
