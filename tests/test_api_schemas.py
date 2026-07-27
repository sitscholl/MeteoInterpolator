from datetime import datetime

import pytest
from pydantic import ValidationError

from src.api.schemas import InterpolationRequest


def test_interpolation_request_localizes_datetime_strings_to_timezone():
    request = InterpolationRequest(
        param="tair_2m",
        start="2026-05-13 12:30",
        end="2026-05-13 13:30",
        timezone="Europe/Rome",
    )

    assert isinstance(request.start, datetime)
    assert isinstance(request.end, datetime)
    assert str(request.start.tzinfo) == "Europe/Rome"
    assert str(request.end.tzinfo) == "Europe/Rome"
    assert request.start.hour == 12
    assert request.end.hour == 13


def test_interpolation_request_converts_aware_datetimes_to_requested_timezone():
    request = InterpolationRequest(
        param="tair_2m",
        start="2026-05-13T10:30:00Z",
        end="2026-05-13T11:30:00Z",
        timezone="Europe/Rome",
    )

    assert str(request.start.tzinfo) == "Europe/Rome"
    assert str(request.end.tzinfo) == "Europe/Rome"
    assert request.start.hour == 12
    assert request.end.hour == 13


def test_interpolation_request_rejects_invalid_timezone():
    with pytest.raises(ValidationError, match="Invalid timezone"):
        InterpolationRequest(
            param="tair_2m",
            start="2026-05-13",
            end="2026-05-14",
            timezone="Not/A_Timezone",
        )


def test_interpolation_request_requires_start_before_end():
    with pytest.raises(ValidationError, match="start date must be before end date"):
        InterpolationRequest(
            param="tair_2m",
            start="2026-05-14",
            end="2026-05-13",
            timezone="Europe/Rome",
        )
