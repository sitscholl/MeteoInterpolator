from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _select_distance_field(
    distances: xr.DataArray,
    lam_value: float,
    point_id,
) -> xr.DataArray:
    field = distances.sel(lam_value=lam_value, id=point_id)
    if "lam_value" in field.dims or "id" in field.dims:
        raise ValueError("Distance selection must result in one 2D field with dimensions ('y', 'x').")
    return field.transpose("y", "x")


def _plot_distance_field_on_axis(
    ax,
    field: xr.DataArray,
    dem: xr.DataArray,
    station_x: float,
    station_y: float,
    *,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    distance_levels: int,
):
    x = field.coords["x"].values
    y = field.coords["y"].values

    mesh = ax.pcolormesh(x, y, field.values, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    if distance_levels > 0:
        values = np.asarray(field.values, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size > 0 and np.nanmin(finite) < np.nanmax(finite):
            ax.contour(
                x,
                y,
                values,
                levels=distance_levels,
                colors="white",
                linewidths=0.45,
                alpha=0.65,
            )

    ax.scatter(station_x, station_y, s=36, c="black", edgecolors="white", linewidths=0.8, zorder=5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    return mesh


def plot_distance_field(
    distances: xr.DataArray,
    dem: xr.DataArray,
    station_x: float,
    station_y: float,
    lam_value: float,
    point_id: str,
    *,
    ax=None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    distance_levels: int = 12,
    dem_levels: int = 14,
    colorbar: bool = True,
):

    field = _select_distance_field(distances, lam_value=lam_value, point_id=point_id)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    else:
        fig = ax.figure

    mesh = _plot_distance_field_on_axis(
        ax,
        field,
        dem,
        station_x,
        station_y,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        distance_levels=distance_levels,
    )
    ax.set_title(f"Station {point_id}, lambda={lam_value}")

    if colorbar:
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.85)
        cbar.set_label("Generalized distance")

    return fig, ax


def plot_distance_panel(
    distances: xr.DataArray,
    dem: xr.DataArray,
    station_x: float,
    station_y: float,
    point_id: str,
    lam_values: Sequence[float] | None = None,
    *,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    cmap: str = "viridis",
    distance_levels: int = 12,
    dem_levels: int = 14,
):

    if lam_values is None:
        lam_values = list(distances.coords["lam_value"].values)
    else:
        lam_values = list(lam_values)
    if not lam_values:
        raise ValueError("At least one lambda value is required for a distance panel.")

    fields = [_select_distance_field(distances, lam_value=lam, point_id=point_id) for lam in lam_values]
    combined = np.concatenate([np.asarray(field.values, dtype=float).ravel() for field in fields])
    finite = combined[np.isfinite(combined)]
    vmin = float(np.nanmin(finite)) if finite.size else None
    vmax = float(np.nanmax(finite)) if finite.size else None

    ncols = min(max(1, ncols), len(lam_values))
    nrows = int(np.ceil(len(lam_values) / ncols))
    if figsize is None:
        figsize = (5.0 * ncols, 4.6 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, constrained_layout=True)
    mesh = None

    for ax, lam, field in zip(axes.ravel(), lam_values, fields):
        mesh = _plot_distance_field_on_axis(
            ax,
            field,
            dem,
            station_x,
            station_y,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            distance_levels=distance_levels,
        )
        ax.set_title(f"lambda={lam}")

    for ax in axes.ravel()[len(lam_values):]:
        ax.set_visible(False)
    
    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.6, location = 'bottom')
        cbar.set_label("Generalized distance")

    fig.suptitle(f"Distance fields for station {point_id}")
    return fig, axes
