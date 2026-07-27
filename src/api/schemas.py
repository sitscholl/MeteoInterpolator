from datetime import datetime
from typing import Annotated, Literal, Self

import pandas as pd
from pydantic import AfterValidator, BaseModel, model_validator
from pytz import timezone

RunStatus = Literal["queued", "running", "completed", "failed"]

def parse_timezone(tz: str) -> timezone:
    try:
        tz_parsed = timezone(tz)
        return tz_parsed
    except Exception as e:
        raise ValueError(f"Timezone could not be parsed with error: {e}")

class InterpolationRequest(BaseModel):
    param: str
    start: datetime
    end: datetime
    timezone: Annotated[str, AfterValidator(parse_timezone)] = 'Europe/Rome'
    target_points: list[tuple[float, float]] | None = None ##TODO: Decide final datatype for target_points support

    @model_validator(mode='after')
    def attach_timezone(self) -> Self:
        self.start = pd.Timestamp(self.start, tzinfo = self.timezone)
        self.end = pd.Timestamp(self.end, tzinfo = self.timezone)
        return self


class InterpolationSubmission(BaseModel):
    request_id: str
    status: RunStatus