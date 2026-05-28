from pathlib import Path
import logging
import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import rioxarray  # noqa: F401
import xarray as xr

from src.interpolate.distance import PathDistanceCalculator
from src.visualization.distance import plot_distance_panel


logger = logging.getLogger(__name__)


def main():
    ##For a mountainside point use:
    #uv run python -m scripts.distance -x 629287 -y 5167218
    ##
    logging.basicConfig(level=logging.INFO, force=True)

    parser = argparse.ArgumentParser(description="Generate distance plots.")
    parser.add_argument("-x", "--xcoord", type=float, default = 638312, help = 'x-coordinate in the crs of the dem')
    parser.add_argument("-y", "--ycoord", type=float, default = 5164307, help = 'y-coordinate in the crs of the dem')
    parser.add_argument("-p", "--pointid", default = "Source Point", help = 'Id of the source point.')
    parser.add_argument("-r", "--res", default = 1000, help = 'Resolution of dem')
    parser.add_argument('-d', "--maxd", type=float, default = None, help = 'Maximum distance for the visibility line.')
    parser.add_argument('-s', action = 'store_true', help = 'Activate to save distance grids as geotiff files.')
    args = parser.parse_args()

    start_time = datetime.now()
    logger.info(f"Launching script for point ({args.xcoord, args.ycoord})")

    dem_file = Path(f"data/dem_envelope_{args.res}m.tif")

    if not dem_file.exists():
        raise FileNotFoundError(f"File {dem_file} does not exist.")

    output_dir = Path(f"data/thrash/distance_{start_time:%d-%m-%Y_%H%M%S}")
    output_dir.mkdir(parents=True, exist_ok=True)

    lam_values = [0, 25, 50, 75, 100, 150, 200]

    dem = xr.open_dataset(dem_file).band_data.squeeze(drop=True)
    distance_calculator = PathDistanceCalculator(
        lam_values=lam_values,
        max_visibility_distance=args.maxd,
    )
    distances = distance_calculator.calculate_fields(
        dem=dem,
        x_coords=[args.xcoord],
        y_coords=[args.ycoord],
        point_ids=[args.pointid],
    ).data

    if args.s:
        for (lam_value, pid), field in distances.groupby(["lam_value", "id"]):
            output_file = output_dir / f"non_euc_distance_{lam_value}_point_{pid}.tif"
            field.transpose("stacked_lam_value_id", "y", "x").rio.to_raster(output_file)
            logger.info("Wrote %s", output_file)

    fig, _ = plot_distance_panel(
        distances,
        dem=dem,
        station_x=args.xcoord,
        station_y=args.ycoord,
        point_id=args.pointid,
        lam_values=lam_values,
        ncols=3,
    )
    figure_file = output_dir / f"_distance_panel_{args.pointid}.png"
    fig.savefig(figure_file, dpi=300, bbox_inches = 'tight')
    plt.close(fig)
    logger.info("Wrote %s", figure_file)


if __name__ == "__main__":
    main()
