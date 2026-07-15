import pandas as pd
import pandera.pandas as pa
import pandera.geopandas as pg

def _is_timezone_aware(s) -> bool:
    if s.empty:
        return True
    if not pd.api.types.is_datetime64_any_dtype(s):
        return False
    return getattr(s.dt, "tz", None) is not None

def station_data_schema(timezone: str | None = None) -> pa.DataFrameSchema:
    checks = pa.Check(
        _is_timezone_aware,
        element_wise=False,
        error="datetime must be timezone-aware",
    )
    dtype = f"datetime64[ns, {timezone}]" if timezone is not None else None
    sensor_columns = [
        "tair_2m",
        "relative_humidity",
        "precipitation",
        "wind_speed",
        "wind_direction",
        "wind_gust",
        "air_pressure",
        "sun_duration",
        "solar_radiation",
        "snow_height",
        "water_level",
        "discharge",
    ]
    return pa.DataFrameSchema(
        {
            "datetime": pa.Column(dtype, checks=checks, coerce=timezone is not None),
            "station_id": pa.Column(str, coerce=True),
            **{
                sensor: pa.Column(float, nullable=True, required=False, coerce=True)
                for sensor in sensor_columns
            },
        },
        index = pa.Index(int),
        unique = ['datetime', 'station_id'],
        strict = 'filter'
    )

_STATION_DATA_SCHEMA = station_data_schema()

_OBSERVATION_POINTS_SCHEMA = pg.GeoDataFrameSchema(
        {
            "geometry": pg.Column("geometry", nullable=False, required=True),
            "elevation": pa.Column(float, nullable=True, required=False, coerce=True)
        },
        index = pa.Index(str, unique=True),
        strict = 'filter'
    )
