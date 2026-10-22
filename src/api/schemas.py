from datetime import datetime
from typing import Annotated, Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
from pydantic import AfterValidator, BaseModel, BeforeValidator, model_validator

from ..utils import localize_datetime_string

RunStatus = Literal["queued", "running", "completed", "failed"]
InterpolationFrequency = Literal["h", "D"]


def parse_datetime(value) -> datetime:
    try:
        parsed = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"Datetime could not be parsed with error: {exc}") from exc

    if pd.isna(parsed):
        raise ValueError(f"Datetime value {value!r} could not be parsed.")

    return parsed.to_pydatetime()


def validate_timezone(tz: str) -> str:
    if not isinstance(tz, str) or not tz.strip():
        raise ValueError("Timezone must be a non-empty string.")

    try:
        ZoneInfo(tz)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Invalid timezone: {tz!r}") from exc

    return tz

class StationRead(BaseModel):
    id: str
    longitude: float
    latitude: float
    elevation: float
    crs: int
    name: str | None = None

class StationCreate(StationRead):
    pass

class DatabaseRefreshSubmission(BaseModel):
    submitted_at: datetime

class JobOverview(BaseModel):
    id: str
    name: str
    trigger: str
    next_run_time: datetime | None
    func: str | None

class InterpolationRequest(BaseModel):
    param: str
    start: Annotated[datetime, BeforeValidator(parse_datetime)]
    end: Annotated[datetime, BeforeValidator(parse_datetime)]
    timezone: Annotated[str, AfterValidator(validate_timezone)] = 'Europe/Rome'
    freq: InterpolationFrequency = 'D'
    target_points: list[tuple[float, float]] | None = None ##TODO: Decide final datatype for target_points support

    @model_validator(mode='after')
    def attach_timezone(self) -> Self:
        self.start = localize_datetime_string(self.start, self.timezone).to_pydatetime()
        self.end = localize_datetime_string(self.end, self.timezone).to_pydatetime()
        if self.start >= self.end:
            raise ValueError("start date must be before end date")
        return self

    @property
    def tzinfo(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


class InterpolationSubmission(BaseModel):
    request_id: str
    status: RunStatus
