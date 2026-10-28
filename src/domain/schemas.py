import pandas as pd
import numpy as np
import pandera.pandas as pa
import pandera.geopandas as pg

def _is_timezone_aware(s) -> bool:
    if s.empty:
        return True
    if not pd.api.types.is_datetime64_any_dtype(s):
        return False
    return getattr(s.dt, "tz", None) is not None

def _clean_humidity(x):
    x = x.where(x.between(0, 101))
    return x.clip(upper=100)

_tair_parser = pa.Parser(lambda x: x.where(x.between(-50, 50) & np.isfinite(x)))
_zero_parser = pa.Parser(lambda x: x.where((x >= 0) & np.isfinite(x)))

def station_data_schema(timezone: str | None = None) -> pa.DataFrameSchema:
    checks = pa.Check(
        _is_timezone_aware,
        element_wise=False,
        error="datetime must be timezone-aware",
    )
    dtype = f"datetime64[ns, {timezone}]" if timezone is not None else None
    return pa.DataFrameSchema(
        columns={
            "datetime": pa.Column(dtype, checks=checks, coerce=timezone is not None),
            "station_id": pa.Column(str, coerce=True),
            "tair_2m": pa.Column(float, nullable=True, required=False, coerce=True, parsers=_tair_parser),
            "tair_2m_min": pa.Column(float, nullable=True, required=False, coerce=True, parsers=_tair_parser),
            "tair_2m_max": pa.Column(float, nullable=True, required=False, coerce=True, parsers=_tair_parser),
            "relative_humidity": pa.Column(float, nullable=True, required=False, coerce=True, parsers=pa.Parser(_clean_humidity)),
            "precipitation": pa.Column(float, nullable=True, required=False, coerce=True, parsers=_zero_parser),
            "wind_speed": pa.Column(float, nullable=True, required=False, coerce=True, parsers=_zero_parser),
            "wind_direction": pa.Column(float, nullable=True, required=False, coerce=True, parsers=pa.Parser(lambda x: x.where(x.between(0, 360) & np.isfinite(x)))),
            "wind_gust": pa.Column(float, nullable=True, required=False, coerce=True, parsers=pa.Parser(lambda x: x.where(x.between(0, 150) & np.isfinite(x)))),
            "air_pressure": pa.Column(float, nullable=True, required=False, coerce=True, parsers=pa.Parser(lambda x: x.where(x.between(700, 1200) & np.isfinite(x)))),
            "sun_duration": pa.Column(float, nullable=True, required=False, coerce=True, parsers=_zero_parser),
            "solar_radiation": pa.Column(float, nullable=True, required=False, coerce=True, parsers=_zero_parser),
            "snow_height": pa.Column(float, nullable=True, required=False, coerce=True, parsers=_zero_parser),
            "water_level": pa.Column(float, nullable=True, required=False, coerce=True, parsers=pa.Parser(lambda x: x.where(np.isfinite(x)))),
            "discharge": pa.Column(float, nullable=True, required=False, coerce=True, parsers=_zero_parser),
        },
        index = pa.Index(int),
        unique = ['datetime', 'station_id'],
        strict = 'filter'
    )

_STATION_DATA_SCHEMA = station_data_schema()

_STATION_CATALOG_SCHEMA = pa.DataFrameSchema(
    columns = {
        'station_id': pa.Column(str, coerce=True, nullable=False, unique = True),
        'name': pa.Column(str, coerce=True, nullable=True),
        'latitude': pa.Column(float, coerce=True, nullable=False, checks=pa.Check(lambda x: np.isfinite(x))),
        'longitude': pa.Column(float, coerce=True, nullable=False, checks=pa.Check(lambda x: np.isfinite(x))),
        'elevation': pa.Column(float, coerce=True, nullable=False, checks=pa.Check(lambda x: np.isfinite(x) & x.between(0, 5000))),
        'crs': pa.Column(int, nullable=False, checks=pa.Check.equal_to(4326)),
        'status': pa.Column(str, nullable=False, coerce=True, parsers=pa.Parser(lambda x: x.str.lower()), checks=pa.Check.isin(allowed_values=['active', 'inactive']))
    },
    strict = 'filter'
)

_OBSERVATION_POINTS_SCHEMA = pg.GeoDataFrameSchema(
        {
            "geometry": pg.Column("geometry", nullable=False, required=True),
            "elevation": pa.Column(float, nullable=True, required=False, coerce=True)
        },
        index = pa.Index(str, unique=True),
        strict = 'filter'
    )
