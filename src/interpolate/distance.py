import xarray as xr
from xrspatial import proximity
import numpy as np

class DistanceCalculator:
    def __init__(
        self
        ):
        pass

    def _get_grid_coords(self, x_coords: list[float], y_coords: list[float], grid: xr.DataArray):
        grid_coords = grid.sel(dict(x = x_coords, y = y_coords), method = 'nearest')
        if grid_coords.empty:
            raise ValueError("No source coordinates fall within the grid.")
        return grid_coords.x.values, grid_coords.y.values

    def _prepare_source_grid(self, grid_x, grid_y, grid):
        ##transform base grid to grid where source pixels have a value of 1 and others 0
        source_grid = grid.copy()
        return xr.where((source_grid.coords["y"] == grid_y) & (source_grid.coords["x"] == grid_x), 1, 0)

    def calculate_euclidean_distance(
        self,
        dem: xr.DataArray, 
        x_coords: list[float], 
        y_coords: list[float], 
        ):

        if 'y' not in dem.coords:
            raise ValueError(f"Missing y dimension 'y'. Make sure the dem has the vertical coordinate named y. Got {dem.coords}")
        if 'x' not in dem.coords:
            raise ValueError(f"Missing x dimension 'x'. Make sure the dem has the horizontal coordinate named x. Got {dem.coords}")
        
        grid_x, grid_y = self._get_grid_coords(x_coords, y_coords, dem)
        source_grid = self._prepare_source_grid(grid_x, grid_y, dem)
        euc_dist = proximity(source_grid)

        return euc_dist

    def calculate_non_euclidean_distance(
        self,
        dem: xr.DataArray, 
        x_coords: list[float], 
        y_coords: list[float], 
        point_ids: list[int] | None = None,
        lam_values: list[float] = [0, 25, 50, 75, 100, 150, 200]
    ):
        
        euc_dist = self.calculate_euclidean_distance(dem, x_coords, y_coords)
        grid_x, grid_y = self._get_grid_coords(x_coords, y_coords, dem)

        if point_ids is not None:

            if len(point_ids) != len(x_coords):
                raise ValueError(f"If supplied, the number of ids must correspond to the number of points. Got {len(point_ids)} vs {len(x_coords)}")

            grid_x = xr.DataArray(data = grid_x, coords = {'id': point_ids})
            grid_y = xr.DataArray(data = grid_y, coords = {'id': point_ids})

        results = []
        for lam in lam_values:
            points_elev = dem.sel(y = grid_y, x = grid_x).values

            neuc_dist = euc_dist + np.sqrt((lam * (dem - points_elev))**2)
            neuc_dist = neuc_dist.where(neuc_dist > 0)

            #todo: preserve ids of x and y coords somehow
            neuc_dist = neuc_dist.assign_coords(
                lam_value = lam
            )

            results.append(neuc_dist)
        
        return xr.merge(results)



    # #lam_values = np.arange(0, 210, 20)
    # lam_values = [0, 25, 50, 75, 100, 150, 200]
    # ids_distance = gpd.overlay(st_info, aoi_square, how = 'intersection')['st_id'].unique()
    # ids_distance = [i for i in ids_distance if i in st_data['st_id'].unique()]
    
    
    # ds = xr.Dataset(
    #     data_vars = {f'l{l}': xr.DataArray(np.nan, dims = ('y', 'x', 'st_id'), coords=dict(x=dem_clip.x.values, y=dem_clip.y.values, st_id = ids_distance)) for l in lam_values},
    #     coords=dict(x=dem_clip.x.values, y=dem_clip.y.values, st_id = ids_distance)
    # )

    # i = 0
    # for sid in ids_distance:

    #     clear_output(wait = True)

    #     st_coords = st_info.loc[st_info['st_id'] == sid, 'geometry'].drop_duplicates()
    #     st_x, st_y = st_coords.x.values, st_coords.y.values

    #     euc_source = dem_square.copy()
    #     pcoords = euc_source.sel(dict(y = st_y, x = st_x), method = 'nearest')
    #     px, py = pcoords.x.values, pcoords.y.values
    #     euc_source = xr.where((euc_source.coords["y"] == py) & (euc_source.coords["x"] == px), 1, 0)

    #     st_elev = dem_square.sel(y = st_y, x = st_x, method = 'nearest').values

    #     euc_dist = proximity(euc_source)
    #     euc_dist = euc_dist.rio.clip(aoi.geometry, aoi.crs)

    #     for lam in lam_values:
    #         neuc_dist = euc_dist + np.sqrt((lam * (dem_clip - st_elev))**2)
    #         neuc_dist = neuc_dist.where(neuc_dist > 0)
    #         ds[f'l{lam}'].loc[dict(st_id = sid)] = neuc_dist

    #     i += 1
    #     perc = np.round((i/len(ids_distance)) * 100, 2)
    #     print(f"Current progress ({sid}): {perc}%")
        
    # ds.to_netcdf(lambda_arrays_path)