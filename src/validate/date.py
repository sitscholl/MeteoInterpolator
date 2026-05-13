import logging

import pandas as pd

logger = logging.getLogger(__name__)

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
            ts = ts.tz_localize(timezone)
        except Exception:
            ts = ts.tz_localize(timezone, ambiguous = False, nonexistent = "shift_forward")
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
