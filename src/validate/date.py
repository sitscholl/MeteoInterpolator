from pandas import to_datetime
import logging

logger = logging.getLogger(__name__)

def localize_datetime_string(value: str, timezone: str):
    "Transforms input datetime strings to datetime objects with correct timezone from runtime context"
    ts = to_datetime(value, dayfirst = True)

    if ts.tzinfo is None:
        try:
            ts = ts.tz_localize(timezone)
        except Exception:
            ts = ts.tz_localize(timezone, ambiguous = False, nonexistent = "shift_forward")
    else:
        if ts.tzinfo != timezone:
            logger.warning(f"Timezone of input datetimes is not the same as configured in the config file. Input dates will be conterted to {timezone} time")
            ts = ts.tz_convert(timezone)
    return ts