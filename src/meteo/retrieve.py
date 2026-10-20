from datetime import datetime
from pandas import Timestamp

from ..database.db import InterpolationDB
from .types import Station

def _to_utc_timestamp(dt: datetime):
    
    ts = Timestamp(dt)
    if ts.tz is None:
        ts = ts.tz_localize('UTC')
    else:
        ts = ts.tz_convert('UTC')

    return ts

def retrieve_stations(
    db: InterpolationDB,
    station_ids: list[str],
    sensor_codes: list[str],
    start: datetime,
    end: datetime,
    target_timezone: str = 'UTC',
    ) -> list[Station]:

    requested_stations = {str(station_id) for station_id in ([station_ids] if isinstance(station_ids, str) else station_ids)}
    requested_sensors = [sensor_codes] if isinstance(sensor_codes, str) else list(sensor_codes)
    if not requested_stations:
        raise ValueError("At least one station_id is required.")
    if not requested_sensors:
        raise ValueError("At least one sensor code is required.")

    start = _to_utc_timestamp(start)
    end = _to_utc_timestamp(end)
    if start >= end:
        raise ValueError(f"Start must be smaller than end. Got {start} vs {end}")

    data = db.get_data(
        station_ids = requested_stations,
        variables = requested_sensors,
        start_time = start,
        end_time = end
        )

    if data.empty:
        raise ValueError(f"No data found for stations {requested_stations} and sensors {requested_sensors} for period {start} - {end}")

    if data["datetime"].dt.tz is None:
        data['datetime'] = data['datetime'].dt.tz_localize("UTC")

    if data['datetime'].dt.tz != target_timezone:
        data['datetime'] = data['datetime'].dt.tz_convert(target_timezone)

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
                crs = 4326,
                data = station_data
            )
        )
    return stations