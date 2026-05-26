from pathlib import Path
import logging

import matplotlib.pyplot as plt
import rioxarray  # noqa: F401
import xarray as xr

from src.interpolate.distance import calculate_non_euclidean_distance
from src.visualization.distance import plot_distance_panel


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, force=True)

    dem_file = Path("data/dem_envelope_1000m.tif")
    output_dir = Path("data/thrash")
    output_dir.mkdir(parents=True, exist_ok=True)

    x_coords = [638312, 629287]
    y_coords = [5164307, 5167218]
    point_ids = ["station_0", "station_1"]
    lam_values = [0, 25, 50, 75, 100, 150, 200]
    max_visibility_distance = 1000

    dem = xr.open_dataset(dem_file).band_data.squeeze(drop=True)
    distances = calculate_non_euclidean_distance(
        dem,
        x_coords=x_coords,
        y_coords=y_coords,
        point_ids=point_ids,
        lam_values=lam_values,
        max_visibility_distance=max_visibility_distance,
    )

    for (lam_value, point_id), field in distances.groupby(["lam_value", "id"]):
        output_file = output_dir / f"non_euc_distance_{lam_value}_point_{point_id}.tif"
        field.transpose("stacked_lam_value_id", "y", "x").rio.to_raster(output_file)
        logger.info("Wrote %s", output_file)

    plot_point_index = 0
    fig, _ = plot_distance_panel(
        distances,
        dem=dem,
        station_x=x_coords[plot_point_index],
        station_y=y_coords[plot_point_index],
        point_id=point_ids[plot_point_index],
        lam_values=lam_values,
        ncols=3,
    )
    figure_file = output_dir / f"distance_panel_{point_ids[plot_point_index]}.png"
    fig.savefig(figure_file, dpi=200)
    plt.close(fig)
    logger.info("Wrote %s", figure_file)


if __name__ == "__main__":
    main()
