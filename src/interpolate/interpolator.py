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