import pandas as pd
import pandera.pandas as pa
import pandera.geopandas as pg

_STATION_DATA_SCHEMA = pa.DataFrameSchema(
        {
            "datetime": pa.Column(pd.DatetimeTZDtype(tz=timezone), coerce=True),
            "station_id": pa.Column(str, coerce=True),
            "tair_2m": pa.Column(float, nullable=True, required=True, coerce=True),
        },
        index = pa.Index(int),
        unique = ['datetime', 'station_id'],
        strict = 'filter'
    )

_OBSERVATION_POINTS_SCHEMA = pg.GeoDataFrameSchema(
        {
            "geometry": pg.Column("geometry", nullable=False, required=True),
        },
        index = pa.Index(str, unique=True),
        strict = 'filter'
    )