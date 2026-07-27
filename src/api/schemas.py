from datetime import datetime
from typing import Annotated, Literal, Self
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import AfterValidator, BaseModel, BeforeValidator, model_validator
from pytz.exceptions import UnknownTimeZoneError

from ..utils import localize_datetime_string

RunStatus = Literal["queued", "running", "completed", "failed"]


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
    except UnknownTimeZoneError as exc:
        raise ValueError(f"Invalid timezone: {tz!r}") from exc

    return tz


class InterpolationRequest(BaseModel):
    param: str
    start: Annotated[datetime, BeforeValidator(parse_datetime)]
    end: Annotated[datetime, BeforeValidator(parse_datetime)]
    timezone: Annotated[str, AfterValidator(validate_timezone)] = 'Europe/Rome'
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
