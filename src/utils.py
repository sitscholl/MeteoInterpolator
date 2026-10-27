import datetime
import logging
import math
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pyproj import CRS

logger = logging.getLogger(__name__)

def split_dates(start_date, end_date, freq, n_days=7, split_on_year=False):
    """
    freq: string (e.g., '1h', '15min') or pd.Timedelta
    """
    if end_date < start_date:
        raise ValueError(f"Start date cannot be smaller than end date. Got {start_date} and {end_date}")

    # Convert freq to a Timedelta for easy math
    freq_delta = pd.Timedelta(freq)
    date_pairs = []
    current_start = start_date
    
    while current_start <= end_date:
        # 1. Calculate the normal step (e.g., 7 days)
        # Note: We subtract one frequency unit from the potential end 
        # so that the chunk spans n_days TOTAL including the start and end.
        potential_end = current_start + datetime.timedelta(days=n_days) - freq_delta
        
        if split_on_year:
            # 2. Calculate the very last possible timestamp of the current year
            # (December 31st, 23:59:59... or whatever the last 'freq' step is)
            next_year_start = datetime.datetime(current_start.year + 1, 1, 1, tzinfo=current_start.tzinfo)
            last_of_year = next_year_start - freq_delta
            
            # 3. Pick the earliest of the three boundaries
            current_end = min(potential_end, end_date, last_of_year)
        else:
            current_end = min(potential_end, end_date)

        # Safety: check we didn't go backwards
        # This can happen if start_date is already the last timestamp of the year
        # Force the end to be the start so we at least get one record
        current_end = max(current_end, current_start)

        date_pairs.append((current_start, current_end))
        
        # 4. MOVE TO THE NEXT TIMESTAMP
        # The next start is exactly one frequency step after the current end
        current_start = current_end + freq_delta
    
    return date_pairs

def localize_datetime_string(value: str, timezone: str):
    """Transform an input datetime string to a timezone-aware pandas Timestamp."""
    if value is None:
        raise ValueError("Datetime value cannot be None.")
    if timezone is None:
        raise ValueError("Timezone cannot be None.")

    ts = pd.to_datetime(value)
    if pd.isna(ts):
        raise ValueError(f"Could not parse datetime value {value!r}.")

    if ts.tzinfo is None:
        try:
            ts = ts.tz_localize(
                timezone,
                ambiguous="raise",
                nonexistent="raise",
            )
        except ValueError as exc:
            raise ValueError(
                f"Cannot interpret {value!r} in timezone {timezone!r}: {exc}. "
                "For a repeated local time, provide an explicit UTC offset."
            ) from exc
    else:
        converted = ts.tz_convert(timezone)
        if converted != ts:
            logger.warning(
                "Input datetime %s was converted to configured timezone %s as %s",
                ts,
                timezone,
                converted,
            )
        ts = converted
    return ts

def get_date_format_from_freq(freq: str, filename_safe: bool = False) -> str:
    if freq is None or str(freq).strip() == "":
        raise ValueError("Frequency must be a non-empty string.")

    try:
        offset = to_offset(freq)
    except ValueError as exc:
        raise ValueError(f"Invalid frequency {freq!r}.") from exc

    try:
        delta = pd.Timedelta(offset)
    except (TypeError, ValueError):
        try:
            delta = pd.Timedelta(offset.nanos, unit="ns")
        except (AttributeError, TypeError, ValueError):
            delta = None

    if delta is None or delta >= pd.Timedelta(days=1):
        return "%Y-%m-%d"
    if delta >= pd.Timedelta(minutes=1):
        return "%Y-%m-%d_%H-%M" if filename_safe else "%Y-%m-%d %H:%M"
    return "%Y-%m-%d_%H-%M-%S" if filename_safe else "%Y-%m-%d %H:%M:%S"

def validate_station_coordinates(
    latitude: float,
    longitude: float,
    elevation: float,
) -> tuple[float, float, float]:
    """Return finite station coordinates after validating their geographic bounds."""
    values = {
        "latitude": latitude,
        "longitude": longitude,
        "elevation": elevation,
    }
    numeric_values: dict[str, float] = {}
    for field, value in values.items():
        if value is None:
            raise ValueError(f"Station {field} cannot be None.")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Station {field} must be a finite number. Got {value!r}"
            ) from exc
        if not math.isfinite(numeric_value):
            raise ValueError(
                f"Station {field} must be a finite number. Got {value!r}"
            )
        numeric_values[field] = numeric_value

    if not -90 <= numeric_values["latitude"] <= 90:
        raise ValueError("Station latitude must be between -90 and 90 degrees.")
    if not -180 <= numeric_values["longitude"] <= 180:
        raise ValueError("Station longitude must be between -180 and 180 degrees.")

    return (
        numeric_values["latitude"],
        numeric_values["longitude"],
        numeric_values["elevation"],
    )

def validate_crs(crs: int):
    if not isinstance(crs, int):
        try:
            crs = int(crs)
        except Exception as e:
            logger.exception(f"crs object should be of type int, got {type(crs)}")
            raise

    if crs != 4326:
        raise ValueError(f"Stations currently can only have a crs of 4326. Got {crs}")

    try:
        _ = CRS.from_epsg(crs)
    except Exception as e:
        logger.exception(f"Invalid crs object: {crs}")
        raise
    return crs

def to_utc_timestamp(input: datetime.datetime | pd.Timestamp | str, raise_missing_timezone: bool = False):
    
    ts = pd.Timestamp(input)
    if ts.tz is None:
        if raise_missing_timezone:
            raise ValueError("Missing timezone in input date")
        ts = ts.tz_localize('UTC')
    else:
        ts = ts.tz_convert('UTC')

    return ts

def to_jsonable(value):
    """Convert metadata values without silently stringifying unknown objects."""
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported output metadata type: {type(value).__name__}")
