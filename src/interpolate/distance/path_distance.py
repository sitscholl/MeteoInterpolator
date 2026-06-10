from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.sparse import coo_array, csr_array
from scipy.sparse.csgraph import dijkstra

import logging

from ...array.base_grid import BaseGrid
from ...array.cache import CacheManager
from .base import BaseDistanceCalculator, DistanceField

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class _TerrainGraphEdges:
    n_cells: int
    rows: np.ndarray
    cols: np.ndarray
    horizontal: np.ndarray
    vertical: np.ndarray

class PathDistanceCalculator(BaseDistanceCalculator):
    def __init__(
        self,
        connectivity_type: int = 4,
        lam_values: Sequence[float] | None = None,
        max_visibility_distance: float | None = None,
        cache_manager: CacheManager | None = None,
        ):
        super().__init__(cache_manager=cache_manager)

        if connectivity_type not in [4, 8]:
            raise ValueError(f"Connectivity type should be one of 4 or 8. Got {connectivity_type}")

        if lam_values is not None:
            lam_values = [float(i) for i in lam_values]

        if lam_values is not None and any([i < 0 for i in lam_values]):
            raise ValueError("lam_values must be > 0!")

        if max_visibility_distance is not None:
            max_visibility_distance = float(max_visibility_distance)
            if max_visibility_distance <= 0:
                raise ValueError(f"max_visibility_distance must be positive. Got {max_visibility_distance}")

        if lam_values is not None and max_visibility_distance is None:
            logger.debug("No max_visibility_distance provided; atmospheric movement through free-air not considered in path distance metric.")

        if lam_values is None or lam_values == [0.0]:
            logger.debug("Simple horizontal distance will be used as distance metric because lam_values is None or [0]. max_visibility_distance will be ignored")
            lam_values = [0.0]
            max_visibility_distance = None

        self.connectivity_type = connectivity_type
        self.lam_values = lam_values
        self.max_visibility_distance = max_visibility_distance

    @classmethod
    def key(cls):
        return "path_distance"

    def cache_parameters(self) -> dict:
        return {
            "connectivity_type": self.connectivity_type,
            "lam_values": list(self.lam_values),
            "max_visibility_distance": self.max_visibility_distance,
        }

    @property
    def _neighbor_offsets(self) -> tuple[tuple[int, int], ...]:
        if self.connectivity_type == 4:
            return ((0, 1), (1, 0))
        return ((0, 1), (1, 0), (1, 1), (1, -1))

    @staticmethod
    def _bresenham_offsets(row_offset: int, col_offset: int) -> tuple[tuple[int, int], ...]:
        x0 = y0 = 0
        x1 = col_offset
        y1 = row_offset

        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy

        offsets = []
        x = x0
        y = y0
        while True:
            offsets.append((y, x))
            if x == x1 and y == y1:
                break
            doubled_error = 2 * error
            if doubled_error >= dy:
                error += dy
                x += sx
            if doubled_error <= dx:
                error += dx
                y += sy

        return tuple(offsets[1:-1])

    @staticmethod
    def _nearest_cell_indices(
        dem: xr.DataArray,
        x_coords: Sequence[float],
        y_coords: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(x_coords) != len(y_coords):
            raise ValueError(
                f"The number of x and y coordinates must match. Got {len(x_coords)} vs {len(y_coords)}"
            )
        if len(x_coords) == 0:
            raise ValueError("At least one source coordinate is required.")

        x = dem.coords["x"].values
        y = dem.coords["y"].values
        x_coords = np.asarray(x_coords, dtype=float)
        y_coords = np.asarray(y_coords, dtype=float)

        x_idx = np.abs(x[None, :] - x_coords[:, None]).argmin(axis=1)
        y_idx = np.abs(y[None, :] - y_coords[:, None]).argmin(axis=1)

        dem_values = np.asarray(dem.values, dtype=float)
        invalid_sources = ~np.isfinite(dem_values[y_idx, x_idx])
        if invalid_sources.any():
            invalid = np.flatnonzero(invalid_sources).tolist()
            raise ValueError(f"Source coordinates snap to invalid DEM cells at point indices: {invalid}")

        return y_idx, x_idx

    @staticmethod
    def _point_ids(
        n_points: int,
        point_ids: Sequence[Hashable] | None,
    ) -> list[Hashable]:
        if point_ids is None:
            return list(range(n_points))
        if len(point_ids) != n_points:
            raise ValueError(
                f"If supplied, the number of ids must correspond to the number of points. Got {len(point_ids)} vs {n_points}"
            )
        return list(point_ids)

    def _build_visibility_edges(self, dem: xr.DataArray) -> _TerrainGraphEdges:
        z = np.asarray(dem.values, dtype=float)
        y = np.asarray(dem.coords["y"].values, dtype=float)
        x = np.asarray(dem.coords["x"].values, dtype=float)

        n_y, n_x = z.shape
        n_cells = n_y * n_x
        valid = np.isfinite(z)

        x_resolution = np.min(np.abs(np.diff(x))) if n_x > 1 else np.inf
        y_resolution = np.min(np.abs(np.diff(y))) if n_y > 1 else np.inf
        max_col_offset = int(np.floor(self.max_visibility_distance / x_resolution)) if np.isfinite(x_resolution) else 0
        max_row_offset = int(np.floor(self.max_visibility_distance / y_resolution)) if np.isfinite(y_resolution) else 0

        logger.debug(
            "Building visibility edges within %s coordinate units",
            self.max_visibility_distance,
        )

        row_parts = []
        col_parts = []
        horizontal_parts = []
        vertical_parts = []

        for row_offset in range(max_row_offset + 1):
            for col_offset in range(-max_col_offset, max_col_offset + 1):
                if row_offset == 0 and col_offset <= 0:
                    continue
                if max(abs(row_offset), abs(col_offset)) <= 1:
                    continue

                line_offsets = self._bresenham_offsets(row_offset, col_offset)
                if not line_offsets:
                    continue

                row_start = max(0, -row_offset)
                row_stop = n_y - max(0, row_offset)
                col_start = max(0, -col_offset)
                col_stop = n_x - max(0, col_offset)
                if row_start >= row_stop or col_start >= col_stop:
                    continue

                rows, cols = np.mgrid[row_start:row_stop, col_start:col_stop]
                next_rows = rows + row_offset
                next_cols = cols + col_offset

                source = rows * n_x + cols
                target = next_rows * n_x + next_cols
                x0 = x[cols]
                y0 = y[rows]
                x1 = x[next_cols]
                y1 = y[next_rows]
                z0 = z[rows, cols]
                z1 = z[next_rows, next_cols]

                horizontal = np.hypot(x1 - x0, y1 - y0)
                vertical = z1 - z0
                horizontal_squared = horizontal**2
                edge_mask = valid[rows, cols] & valid[next_rows, next_cols] & (horizontal <= self.max_visibility_distance)

                for line_row_offset, line_col_offset in line_offsets:
                    line_rows = rows + line_row_offset
                    line_cols = cols + line_col_offset
                    line_x = x[line_cols]
                    line_y = y[line_rows]
                    line_z = z[line_rows, line_cols]

                    line_fraction = ((line_x - x0) * (x1 - x0) + (line_y - y0) * (y1 - y0)) / horizontal_squared
                    free_air_z = z0 + line_fraction * vertical
                    edge_mask &= np.isfinite(line_z) & (line_z < free_air_z)

                if not edge_mask.any():
                    continue

                row_parts.append(source[edge_mask].ravel())
                col_parts.append(target[edge_mask].ravel())
                horizontal_parts.append(horizontal[edge_mask].ravel())
                vertical_parts.append(vertical[edge_mask].ravel())

        if not row_parts:
            logger.debug("Built 0 visibility edges")
            return _TerrainGraphEdges(
                n_cells=n_cells,
                rows=np.array([], dtype=np.int64),
                cols=np.array([], dtype=np.int64),
                horizontal=np.array([], dtype=float),
                vertical=np.array([], dtype=float),
            )

        n_edges = sum(len(rows) for rows in row_parts)
        logger.debug("Built %s visibility edges", n_edges)

        return _TerrainGraphEdges(
            n_cells=n_cells,
            rows=np.concatenate(row_parts),
            cols=np.concatenate(col_parts),
            horizontal=np.concatenate(horizontal_parts),
            vertical=np.concatenate(vertical_parts),
        )

    def _build_terrain_graph_edges(
        self,
        dem: xr.DataArray,
    ) -> _TerrainGraphEdges:
        z = np.asarray(dem.values, dtype=float)
        y = np.asarray(dem.coords["y"].values, dtype=float)
        x = np.asarray(dem.coords["x"].values, dtype=float)

        n_y, n_x = z.shape
        n_cells = n_y * n_x
        valid = np.isfinite(z)

        row_parts = []
        col_parts = []
        horizontal_parts = []
        vertical_parts = []

        for row_offset, col_offset in self._neighbor_offsets:
            row_start = max(0, -row_offset)
            row_stop = n_y - max(0, row_offset)
            col_start = max(0, -col_offset)
            col_stop = n_x - max(0, col_offset)

            rows, cols = np.mgrid[row_start:row_stop, col_start:col_stop]
            next_rows = rows + row_offset
            next_cols = cols + col_offset

            edge_mask = valid[rows, cols] & valid[next_rows, next_cols]
            if not edge_mask.any():
                continue

            source = (rows[edge_mask] * n_x + cols[edge_mask]).ravel()
            target = (next_rows[edge_mask] * n_x + next_cols[edge_mask]).ravel()

            horizontal = np.hypot(
                x[next_cols[edge_mask]] - x[cols[edge_mask]],
                y[next_rows[edge_mask]] - y[rows[edge_mask]],
            )
            vertical = z[next_rows[edge_mask], next_cols[edge_mask]] - z[rows[edge_mask], cols[edge_mask]]

            row_parts.append(source)
            col_parts.append(target)
            horizontal_parts.append(horizontal)
            vertical_parts.append(vertical)

        if not row_parts:
            surface_edges = _TerrainGraphEdges(
                n_cells=n_cells,
                rows=np.array([], dtype=np.int64),
                cols=np.array([], dtype=np.int64),
                horizontal=np.array([], dtype=float),
                vertical=np.array([], dtype=float),
            )
        else:
            surface_edges = _TerrainGraphEdges(
                n_cells=n_cells,
                rows=np.concatenate(row_parts),
                cols=np.concatenate(col_parts),
                horizontal=np.concatenate(horizontal_parts),
                vertical=np.concatenate(vertical_parts),
            )

        if self.max_visibility_distance is None:
            logger.debug("Built %s surface graph edges", len(surface_edges.rows))
            return surface_edges

        visibility_edges = self._build_visibility_edges(dem)
        logger.debug(
            "Built %s total graph edges: %s surface, %s visibility",
            len(surface_edges.rows) + len(visibility_edges.rows),
            len(surface_edges.rows),
            len(visibility_edges.rows),
        )
        return _TerrainGraphEdges(
            n_cells=n_cells,
            rows=np.concatenate((surface_edges.rows, visibility_edges.rows)),
            cols=np.concatenate((surface_edges.cols, visibility_edges.cols)),
            horizontal=np.concatenate((surface_edges.horizontal, visibility_edges.horizontal)),
            vertical=np.concatenate((surface_edges.vertical, visibility_edges.vertical)),
        )

    @staticmethod
    def _build_terrain_graph(edges: _TerrainGraphEdges, lam: float) -> csr_array:
        cost = np.sqrt(edges.horizontal**2 + (lam * edges.vertical) ** 2)
        return coo_array((cost, (edges.rows, edges.cols)), shape=(edges.n_cells, edges.n_cells)).tocsr()

    def calculate_distance(
        self,
        dem: BaseGrid,
        x_coords: Sequence[float],
        y_coords: Sequence[float],
        point_ids: Sequence[Hashable] | None = None,
    ) -> DistanceField:
        dem_data = self._validate_dem(dem)
        x_coords, y_coords, point_ids = self._validate_source_points(x_coords, y_coords, point_ids)
        y_idx, x_idx = self._nearest_cell_indices(dem_data, x_coords, y_coords)
        ids = self._point_ids(len(x_idx), point_ids)
        lam_values = list(self.lam_values)

        if len(lam_values) == 0:
            raise ValueError("At least one lambda value is required.")
        if any(lam < 0 for lam in lam_values):
            raise ValueError("Lambda values must be non-negative.")

        n_y = dem_data.sizes["y"]
        n_x = dem_data.sizes["x"]
        source_indices = y_idx * n_x + x_idx

        logger.info(
            "Calculating generalized distance fields for %s source(s), %s lambda value(s)",
            len(ids),
            len(lam_values),
        )
        logger.debug(
            "Building %s-neighbor terrain graph edges with max_visibility_distance=%s",
            self.connectivity_type,
            self.max_visibility_distance,
        )
        edges = self._build_terrain_graph_edges(dem_data)

        distance_fields = []
        for lam in lam_values:
            logger.debug("Building terrain graph for lam value %s", lam)
            graph = self._build_terrain_graph(edges, lam=float(lam))

            logger.debug("Calculating non-euclidean distance for lam value %s", lam)
            distances = dijkstra(
                csgraph=graph,
                directed=False,
                indices=source_indices,
                return_predecessors=False,
            )
            distances = np.asarray(distances, dtype=float).reshape((len(ids), n_y, n_x))
            distances[~np.isfinite(distances)] = np.nan
            distance_fields.append(distances)

        data = np.stack(distance_fields, axis=0)
        edge_description = f"{self.connectivity_type}-neighbor terrain graph"
        if self.max_visibility_distance is not None:
            edge_description += f" plus visible free-air edges up to {self.max_visibility_distance} coordinate units"

        distance_array = xr.DataArray(
            data,
            dims=("lam_value", "id", "y", "x"),
            coords={
                "lam_value": lam_values,
                "id": ids,
                "y": dem_data.coords["y"],
                "x": dem_data.coords["x"],
            },
            name="path_distance",
            attrs={
                "description": (
                    "Frei-style generalized distance computed as shortest paths over a "
                    f"{edge_description} with edge costs sqrt(horizontal_distance^2 + "
                    "(lambda * elevation_difference)^2)."
                ),
                "connectivity": self.connectivity_type,
                "max_visibility_distance": self.max_visibility_distance,
            },
        )
        return DistanceField(distance_type = 'path_distance', data = distance_array)
