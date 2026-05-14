import numpy as np
import xarray as xr

from dataclasses import dataclass
import logging

from .vertical import BaseVerticalModel
from .residuals import BaseResidualModel
from .regions import InterpolationRegions
from .cv import CrossValidator

logger = logging.getLogger(__name__)

@dataclass
class Interpolator:
    vertical_model: BaseVerticalModel
    residual_model: BaseResidualModel | None = None
    regions: InterpolationRegions | None = None
    cross_validator: CrossValidator | None = None

    @classmethod
    def from_config(cls, config: dict):
        vertical_config = dict(config["vertical_model"])
        vertical_handler = vertical_config.pop("type")
        vertical_model = BaseVerticalModel.create(vertical_handler, **vertical_config)

        residual_config = config.get('residual_model')
        if residual_config is None:
            logger.info("No residual model configuration provided. Residuals will not be interpolated")
            residual_model = None
        else:
            residual_config = dict(residual_config)
            residual_handler = residual_config.pop("type")
            residual_model = BaseResidualModel.create(residual_handler, **residual_config)

        ## Interpolation Regions
        region_config = config.get('interpolation_regions')
        interpolation_regions = InterpolationRegions(**region_config) if region_config is not None else None
        if region_config is None:
            logger.info("No interpolation regions specified.")

        ## Cross validation
        cv_config = config.get('cross_validation')
        cross_validator = CrossValidator(**cv_config) if cv_config is not None else None
        if cv_config is None:
            logger.info('No cross validation configuration provided. Cross validation will be skipped')

        return cls(vertical_model = vertical_model, residual_model = residual_model, regions = interpolation_regions, cross_validator = cross_validator)

    def interpolate(self, X: np.ndarray, y: np.ndarray, target_grid: np.ndarray | xr.DataArray):
        if self.cross_validator is not None:
            raise NotImplementedError("Cross Validation has not been implemented yet")
        else:
            cv_results = None     
        
        vertical_fit = self.vertical_model.fit(X, y)
        vertical_preds = vertical_fit.predict(target_grid)

        if self.residual_model is not None:
            raise NotImplementedError("Residual interpolation has not been implemented yet")
            # residuals = self.calculate_residuals(X, vertical_preds, coords = coords)
            # residual_fit = self.residual_model.fit(residuals, coords)
            # residual_preds = residual_fit.predict(target_grid)

        return vertical_preds, cv_results
