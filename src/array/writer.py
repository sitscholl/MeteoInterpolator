import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import numpy as np
import xarray as xr
from zarr.codecs import Blosc
from pyproj import CRS

logger = logging.getLogger(__name__)

DEFAULT_COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=1)

class GridWriter:

    def __init__(
            self, 
            path: Path | str,
            start: str,
            end: str,
            freq: str,
            variables: Sequence[str],
            shape: Sequence[int] | None = None,
            coords: Mapping[str, Any] | None = None,
            crs: Any | None = None,
            chunks: Mapping[str, Any] | None = None,
            fill_value: int | float = -999,
            scale_factor: int | float = 1,
            dtype: str = 'int32',
            drop_attrs: bool = True,
            append_dims: Sequence[str] = ('time',),
            align: bool = True,
            method: str = 'nearest'
        ):
        
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(append_dims, str):
            self.append_dims = [append_dims]
        else:
            self.append_dims = list(append_dims)

        self.time_coords = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq = freq)

        self.compressor = DEFAULT_COMPRESSOR
        self.variables = list(variables)

        self.coords = None
        if isinstance(coords, Mapping):
            self.coords = dict(coords)
        elif coords is not None:
            raise TypeError("coords must be a mapping of dimension names to sizes or None.")

        self.shape = None
        if shape is not None:
            self.shape = tuple(shape)

        if self.shape is not None and self.coords is not None:
            if len(self.shape) != len(self.coords):
                raise ValueError(
                    f"Dimensions in shape do not match number of coordinates. "
                    f"Got {len(self.shape)} vs {len(self.coords)}"
                )

            # Validate coordinate lengths against provided shape where possible
            for dim, expected in zip(self.coords, self.shape):
                coord = self.coords[dim]
                coord_len = getattr(coord, "size", None) or len(coord)
                if coord_len != expected:
                    raise ValueError(
                        f"Coordinate '{dim}' length ({coord_len}) does not match expected shape entry ({expected})."
                    )

        self.chunks = None
        if isinstance(chunks, Mapping):
            self.chunks = dict(chunks)
        elif chunks is not None:
            raise TypeError("chunks must be a mapping of dimension names to chunk sizes or None.")

        self.crs = crs
        self.fill_value = fill_value
        self.scale_factor = scale_factor
        self.dtype = dtype
        self.drop_attrs = drop_attrs
        self.align = align
        self.method = method

    def get_encoding(self) -> dict:
        if not self.variables:
            raise ValueError("Cannot build encoding because no variables are defined.")
        return {
            var: {
                "compressors": (self.compressor,),
                "_FillValue": self.fill_value,
                "scale_factor": self.scale_factor,
                "dtype": self.dtype,
            }
            for var in self.variables
        }

    @staticmethod
    def _values_equal(a: Any, b: Any) -> bool:
        if a is b:
            return True
        if a is None or b is None:
            return a is b
        if isinstance(a, xr.DataArray) and isinstance(b, xr.DataArray):
            return a.equals(b)
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return np.array_equal(np.asarray(a), np.asarray(b))
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            return a.keys() == b.keys() and all(
                GridWriter._values_equal(a[k], b[k]) for k in a
            )
        if isinstance(a, Sequence) and isinstance(b, Sequence) and not isinstance(a, (str, bytes)):
            return len(a) == len(b) and all(
                GridWriter._values_equal(x, y) for x, y in zip(a, b)
            )
        return a == b

    def _initialize_spec_from_data(self, data: xr.Dataset) -> None:
        if not isinstance(data, xr.Dataset):
            raise TypeError("Data must be an xarray Dataset.")
        if not data.data_vars:
            raise ValueError("Dataset has no data variables to infer spec from.")

        ref_var = next(iter(data.data_vars))
        ref = data[ref_var]

        # Shape
        if self.shape is None:
            self.shape = tuple(ref.shape)
        elif tuple(self.shape) != tuple(ref.shape):
            raise ValueError(
                f"Provided shape {self.shape} does not match data shape {ref.shape}."
            )

        # Coords
        if self.coords is None:
            self.coords = {dim: ref[dim].values for dim in ref.dims}
        else:
            for dim, coord in self.coords.items():
                if dim not in ref.dims:
                    raise ValueError(
                        f"Provided coord dim '{dim}' is not present in data dims {ref.dims}."
                    )
                ref_coord = ref[dim].values
                if not np.array_equal(np.asarray(coord), np.asarray(ref_coord)):
                    raise ValueError(
                        f"Provided coordinate values for '{dim}' do not match data."
                    )

        # CRS
        if self.crs is None:
            self.crs = data.rio.crs or data.attrs.get("crs")

        # Chunks
        if self.chunks is None and ref.chunks is not None:
            self.chunks = {
                dim: tuple(chunks)
                for dim, chunks in zip(ref.dims, ref.chunks)
            }
                
    def _check_dims(self, arr_existing, arr_new, append_dims):
        existing_dims = {
            name: size
            for name, size in arr_existing.sizes.items()
            if name not in append_dims
        }
        new_dims = {
            name: size
            for name, size in arr_new.sizes.items()
            if name not in append_dims
        }

        if existing_dims != new_dims:
            raise ValueError(
                "Existing dataset dimensions and shapes "
                f"{existing_dims} do not match expected {new_dims}."
            )

        dims_unequal = {
            dim: not arr_new[dim].equals(arr_existing[dim])
            for dim in existing_dims
        }

        if any(dims_unequal.values()):
            if not self.align:
                raise ValueError(
                    "Cannot insert in zarr store if dimensions are not equal. "
                    f"Unequal values found for dimensions {dims_unequal}."
                )
            needs_alignment = True
        else:
            needs_alignment = False

        return needs_alignment

    def _check_chunks(self, arr_existing, arr_new, append_dims):
        if arr_new.chunks is not None:
            
            if arr_existing.chunks is not None:
                existing_chunks = {
                    dim: chunk
                    for chunk, dim in zip(arr_existing.chunks, arr_existing.dims)
                    if dim not in append_dims
                }
            else:
                existing_chunks = {}

            new_chunks = {
                dim: chunk
                for chunk, dim in zip(arr_new.chunks, arr_new.dims)
                if dim not in append_dims
            }

            if existing_chunks != new_chunks:
                raise ValueError(
                    "Existing dataset chunks "
                    f"{arr_existing.chunks} do not match expected {arr_new.chunks}."
                )

    def _check_coordinate_values(self, arr_existing, arr_new, append_dims):
        for dim in append_dims:
            if dim in arr_new.dims:
                missing_dim_values = arr_new[dim][~arr_new[dim].isin(arr_existing[dim])].values
                if len(missing_dim_values) > 0:
                    raise ValueError(
                        f"Not all values of coordinate {dim} are present in existing dataset. "
                        "Writing to a zarr region slice requires that no dimensions or metadata "
                        f"are changed by the write. Missing values: {missing_dim_values}"
                    )

    def validate_store(self, ds_new: xr.Dataset) -> xr.Dataset:
        """
        Ensures that an existing Zarr store aligns with ds_new for appending data.
        If align is true, ds_new will be reprojected to match the coordinate values of the existing zarr store.

        Parameters
        ----------
        ds_new : xr.Dataset
            The new data to be appended.

        Raises
        ------
        ValueError
            If the existing dataset dimensions or chunks do not match the expected dimensions or chunks,
            or if not all values of a coordinate in append_dims are present in the existing dataset.
        NotImplementedError
            If attempting to add new variables to an existing Zarr store.
        """

        if not self.path.exists():
            raise ValueError(
                f"zarr store at {self.path} does not exist. Create one before validating."
            )

        ds_existing = xr.open_zarr(self.path)
        existing_crs = ds_existing.rio.crs or ds_existing.attrs.get('crs')

        if existing_crs is None:
            raise ValueError(f"Cannot read crs info from existing zarr store at {self.path}")

        new_crs = ds_new.rio.crs or ds_new.attrs.get("crs")
        if new_crs is None:
            raise ValueError("Cannot read crs info from new dataset.")

        existing_crs_wkt = CRS.from_user_input(existing_crs).to_wkt()
        new_crs_wkt = CRS.from_user_input(new_crs).to_wkt()
        if existing_crs_wkt != new_crs_wkt:
            raise ValueError(
                "Coordinate systems of the existing store and new data differ: "
                f"{existing_crs} vs {new_crs}"
            )

        append_dims = set(self.append_dims)
        needs_alignment = False
        align_base = None

        for var_name in ds_new.data_vars:
            if var_name not in ds_existing:
                raise NotImplementedError(
                    f"Found existing zarr store but variable '{var_name}' is not present. "
                    "Adding new variables is not supported."
                )

            arr_existing = ds_existing[var_name].drop_vars('spatial_ref', errors='ignore')
            arr_new = ds_new[var_name].drop_vars('spatial_ref', errors='ignore')

            needs_alignment_var = self._check_dims(arr_existing, arr_new, append_dims)
            if needs_alignment_var:
                needs_alignment = True
                if align_base is None:
                    align_base = arr_existing

            self._check_chunks(arr_existing, arr_new, append_dims)
            self._check_coordinate_values(arr_existing, arr_new, append_dims)

        if needs_alignment:
            # logger.debug(
            #     "Unequal coordinate values detected. Aligning incoming dataset to existing store."
            # )
            # ds_new = align_arrays(
            #     ds_new,
            #     base=align_base,
            #     method=self.method,
            # )[0]
            raise NotImplementedError("Unequal coordinate values between new array and existing zarr store detected. Dynamic aligning currently not implemented.")

        logger.debug('New data succesfully validated against existing zarr store.')

        return ds_new

    def _write_crs_info(self, ds, crs):
        ds = ds.rio.write_crs(crs)
        return ds.assign_attrs(crs = CRS.from_user_input(crs).to_wkt())

    def create(self, overwrite: bool = False):
        """
        Creates a zarr store according to the parameters in self
        """
        if self.path.exists() and not overwrite:
            logger.warning("Zarr store at %s already exists", self.path)
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Create dummy dataset to preallocate a zarr store with necessary metadata but no data
        if self.shape is None:
            raise ValueError('Shape information is None. Cannot create base zarr store.')

        if self.crs is None:
            raise ValueError('No crs information available. Cannot create zarr store without crs.')

        dims = list(self.coords.keys()) if self.coords is not None else None
        dummy = xr.DataArray(np.empty(self.shape), coords=self.coords, dims=dims)

        if self.chunks:
            dummy = dummy.chunk(self.chunks)

        if "time" not in dummy.dims:
            dummy = dummy.expand_dims({"time": self.time_coords})
        else:
            dummy = dummy.assign_coords(time=self.time_coords)

        dummy = dummy.expand_dims({'var': self.variables}).to_dataset(dim='var')
        
        dummy = self._write_crs_info(dummy, self.crs)

        dummy.to_zarr(
            self.path,
            mode="w",
            compute=False,
            encoding=self.get_encoding(),
        )

        logger.debug(f"New zarr store created at {self.path}")

    def write(self, data: xr.Dataset | xr.DataArray, overwrite: bool = False):
        """
        Writes data into an existing Zarr store.
        Inserting of new variables to an existing store is currently not supported and will raise an error.
        If the zarr store already exists, the function will check if the new data is compatible with the
        existing store.

        Parameters
        ----------
        data : xr.Dataset or xr.DataArray
            The data to be inserted into the Zarr store.

        Raises
        ------
        NotImplementedError
            If attempting to add new variables to an existing Zarr store.
        """

        if self.variables is None:
            raise ValueError('Variables in ZarrSpec not specified. Cannot write to zarr store.')

        if isinstance(data, xr.DataArray):
            if data.name is None:
                if len(self.variables) == 1:
                    data = data.rename(self.variables[0])
                else:
                    data = data.rename("var")
            data = data.to_dataset()

        if not isinstance(data, xr.Dataset):
            raise TypeError("Data must be an xarray Dataset or DataArray.")

        self._initialize_spec_from_data(data)

        unknown_vars = set(data.data_vars) - set(self.variables)
        if unknown_vars:
            raise ValueError(
                f"Variables {unknown_vars} are not defined in the Zarr specification."
            )

        if not self.path.exists() or overwrite:
            self.create(overwrite=overwrite)

        aligned = self.validate_store(ds_new=data)

        if self.drop_attrs:
            aligned = aligned.drop_attrs()

        aligned = aligned.drop_vars(
            ['spatial_ref', 'rotated_latitude_longitude'],
            errors='ignore'
        )

        #Tranform to float to avoid error with encoding
        for var in aligned.keys():
            if np.issubdtype(aligned[var].dtype, np.integer):
                aligned[var] = aligned[var].astype(np.float32)

        aligned.to_zarr(
            self.path,
            mode="a",
            region="auto",
        )

        logger.debug(f'New data with shape {aligned.sizes} inserted into zarr store at {self.path}')
