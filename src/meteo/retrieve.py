from datetime import datetime
from pandas import Timestamp

from ..database.db import InterpolationDB
from ..database.models import DATABASE_VARIABLES
from .types import Station
from ..utils import to_utc_timestamp


def retrieve_stations(
    db: InterpolationDB,
    station_ids: list[str],
    sensor_codes: list[str],
    start: datetime,
    end: datetime,
    target_timezone: str = 'UTC',
    ) -> list[Station]:

    requested_stations = {str(station_id) for station_id in ([station_ids] if isinstance(station_ids, str) else station_ids)}
    if not requested_stations:
        raise ValueError("At least one station_id is required.")

    requested_sensors = [sensor_codes] if isinstance(sensor_codes, str) else list(sensor_codes)
    invalid_sensors = [i for i in requested_sensors if i not in DATABASE_VARIABLES]
    if invalid_sensors:
        raise ValueError(f"Invalid sensors for database retrieval: {invalid_sensors}")

    valid_sensors = [i for i in DATABASE_VARIABLES if i in requested_sensors]
    if not valid_sensors:
        raise ValueError(f"No valid sensors to retrive. Choose from: {DATABASE_VARIABLES}")

    start = to_utc_timestamp(start)
    end = to_utc_timestamp(end)
    if start >= end:
        raise ValueError(f"Start must be smaller than end. Got {start} vs {end}")

    data = db.get_data(
        station_ids = requested_stations,
        start_time = start,
        end_time = end
        )

    if data.empty:
        raise ValueError(f"No data found for stations {requested_stations} and sensors {requested_sensors} for period {start} - {end}")

    if data["datetime"].dt.tz is None:
        data['datetime'] = data['datetime'].dt.tz_localize("UTC")

    if data['datetime'].dt.tz != target_timezone:
        data['datetime'] = data['datetime'].dt.tz_convert(target_timezone)
    data = data[['station_id', 'datetime'] + valid_sensors].copy()

    all_stations = db.get_stations()

    stations = []
    for station_id, station_data in data.groupby('station_id'):
        station_model = [i for i in all_stations if i.id == station_id]
        if not station_model:
            raise ValueError(f"No station with id {station_id} found in database")
        
        station_model = station_model[0]
        stations.append(
            Station(
                id = station_model.id,
                x = station_model.longitude,
                y = station_model.latitude,
                elevation = station_model.elevation,
                name = station_model.name,
                crs = station_model.crs,
                data = station_data
            )
        )
    return stations