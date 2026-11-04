import logging
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from zoneinfo import ZoneInfo

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from .database.db import InterpolationDB
    from .meteo import BaseMeteoHandler, MeteoResampler

logger = logging.getLogger(__name__)

class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

class LoaderConfig(ConfigModel):
    stations: list[str] | None = None

    @field_validator('stations', mode='before')
    @classmethod
    def ensure_list(cls, value: Any) -> list:  
        return [value] if isinstance(value, str) else value

class ProvinceLoaderConfig(LoaderConfig):
    handler: Literal["province_api"]
    chunk_size_days: int = Field(default=365, gt=10)
    timeout: int = Field(default=20, gt=0)
    max_concurrent_requests: int = Field(default=5, ge=1)
    sleep_time: int = Field(default=1, ge=0)

class LocalFileLoaderConfig(LoaderConfig):
    handler: Literal["local_file"]
    observations_path: list[str]
    station_metadata_path: str
    timezone: ZoneInfo

    @field_validator('observations_path', mode='before')
    @classmethod
    def normalize_observations_path(cls, value: Any):
        return [value] if isinstance(value, str) else value

    @field_validator('station_metadata_path', mode='after')
    @classmethod
    def validate_metadata_path(cls, value: str):
        if not Path(value).is_file():
            raise ValueError(f"Could not find metadata file at {value}. Check station_metadata_path configuration.")
        return value

    @field_validator('timezone', mode = 'before')
    @classmethod
    def validate_timezone(cls, value: Any):
        if isinstance(value, ZoneInfo):
            return value
        try:
            tz = ZoneInfo(value)
            return tz
        except Exception as e:
            raise ValueError(f"Invalid timezone '{value}' for local file loader.")

class SchedulerConfig(ConfigModel):
    hour: int = Field(ge=0, le=23)

class IngestorConfig(ConfigModel):
    overlap_days: int = Field(gt=0, default = 3)

class ResamplerConfig(ConfigModel):
    min_coverage: float = Field(gt=0, le=1, default = .7)

class DatabaseConfig(ConfigModel):
    engine: str = Field("sqlite:///db/database.db")

class MeteoDataConfig(ConfigModel):
    station_catalog: str
    loader: ProvinceLoaderConfig | LocalFileLoaderConfig = Field(discriminator="handler")
    scheduler: SchedulerConfig
    database_ingestion: IngestorConfig
    resampling: ResamplerConfig
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)

    @property
    def configured_stations(self):
        return self.loader.stations

    @property
    def scheduler_options(self) -> dict:
        return self.scheduler.model_dump()

    @property
    def ingestor_options(self) -> dict:
        return self.database_ingestion.model_dump()

    def create_meteo_loader(self) -> "BaseMeteoHandler":
        from .meteo import BaseMeteoHandler
        options = self.loader.model_dump(exclude={"handler", "stations"})
        return BaseMeteoHandler.create(self.loader.handler, **options)

    def create_database_handler(self) -> "InterpolationDB":
        from .database.db import InterpolationDB
        return InterpolationDB(**self.database.model_dump())

    def create_meteo_resampler(self) -> "MeteoResampler":
        from .meteo import MeteoResampler
        return MeteoResampler(**self.resampling.model_dump())

class CrossValidationConfig(ConfigModel):
    enabled: bool = False
    param_grid: dict[str, list[float]] | None = Field(default=None)
    n_splits: int | None = Field(default=None, ge=2, strict=True)
    min_relative_score_improvement: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    improvement_latency_steps: int = Field(default=3, ge=1, strict=True)
    select_by: Literal['rmse', 'mae'] = 'rmse'

    @field_validator("select_by", mode='before')
    @classmethod
    def to_lower(cls, value: str):
        return value.lower() if isinstance(value, str) else value

    @model_validator(mode='after')
    def validate_param_grid(self):
        if not self.enabled:
            return self
        
        param_grid = self.param_grid

        if param_grid is None:
            raise ValueError("If cross validation is enabled, please set a param_grid.")

        if set(param_grid) != {"lam_value"}:
            keys = param_grid.keys()
            raise ValueError(
                "Cross-validation parameter grid must contain exactly one parameter named "
                f"'lam_value'. Got {keys}"
            )
        
        lam_values = param_grid['lam_value']
        if any(not isinstance(value, Real) or not np.isfinite(value) for value in lam_values):
            raise ValueError("Parameter grid for lam_value must contain only finite numeric values")

        return self