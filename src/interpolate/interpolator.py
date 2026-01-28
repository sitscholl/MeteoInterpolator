import numpy as np
import xarray as xr

from dataclasses import dataclass

from .vertical import VerticalModel
from .residuals import ResidualModel
from .regions import InterpolationRegions
from .cv import CrossValidator

@dataclass
class Interpolator:
    vertical_model: VerticalModel
    residual_model: ResidualModel | None = None
    regions: InterpolationRegions | None = None
    cross_validator: CrossValidator | None = None

    def interpolate(self, X: np.ndarray, y: np.ndarray, target_grid: np.ndarray | xr.DataArray):
        vertical_fit = self.vertical_model.fit(X, y)
        vertical_preds = vertical_fit.predict(target_grid)

        if self.residual_model is not None:
            raise NotImplementedError("Residual interpolation has not been implemented yet")
            # residuals = self.calculate_residuals(X, vertical_preds, coords = coords)
            # residual_fit = self.residual_model.fit(residuals, coords)
            # residual_preds = residual_fit.predict(target_grid)

        return vertical_preds