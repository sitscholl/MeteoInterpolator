from dataclasses import dataclass

import pandas as pd

from ..domain.schemas import _STATION_DATA_SCHEMA


@dataclass
class Station:
    id: str
    x: float
    y: float
    crs: int
    elevation: float | None = None
    name: str | None = None
    data: pd.DataFrame | None = None

    def __post_init__(self):

        if self.id is None:
            raise ValueError("Station id cannot be None.")
        if self.x is None:
            raise ValueError("Station x-coordinate cannot be None.")
        if self.y is None:
            raise ValueError("Station y-coordinate cannot be None.")
        if self.crs is None:
            raise ValueError("Station crs cannot be None.")

        if self.crs != 4326:
            raise NotImplementedError(f"Station crs is {self.crs}. Only 4326 is implemented for now. Make sure the MeteoHandler returns coordinates in this crs.")

        if -90 > self.y or self.y > 90:
            raise ValueError("Latitude must be between -90 and 90")
        if -180 > self.x or self.x > 180:
            raise ValueError("Longitude must be between -180 and 180")

        if self.data is not None:
            self.data = _STATION_DATA_SCHEMA.validate(self.data)