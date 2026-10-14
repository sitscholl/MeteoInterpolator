import math
from datetime import UTC, datetime

import pandas as pd
import pytest
from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError

from src.database import models
from src.database.db import InterpolationDB


@pytest.fixture
def database():
    db = InterpolationDB("sqlite:///:memory:")
    yield db
    db.close()


@pytest.mark.parametrize(
    ("latitude", "longitude", "elevation"),
    [
        (91, 11, 300),
        (-91, 11, 300),
        (46, 181, 300),
        (46, -181, 300),
        (46, 11, None),
        (46, 11, math.nan),
    ],
)
def test_add_station_rejects_invalid_required_coordinates(
    database, latitude, longitude, elevation
):
    with pytest.raises(ValueError):
        database.add_station("station-1", latitude, longitude, elevation)


def test_sqlite_enforces_measurement_foreign_keys(database):
    variable = database.add_variable("tair_2m", "degC")

    with pytest.raises(IntegrityError):
        with database.engine.begin() as connection:
            connection.execute(
                insert(models.Measurement).values(
                    station_id="unknown-station",
                    variable_id=variable.id,
                    datetime=datetime(2026, 1, 1, tzinfo=UTC),
                    value=1.0,
                    retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
                )
            )


def test_add_station_data_stores_retrieval_timestamp(database):
    database.add_station("station-1", 46, 11, 300)
    database.add_variable("tair_2m", "degC")
    database.add_station_data(
        "station-1",
        pd.DataFrame(
            {
                "station_id": ["station-1"],
                "datetime": ["2026-01-01T00:00:00Z"],
                "tair_2m": [2.5],
            }
        ),
    )

    data = database.get_data(
        "station-1",
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-02T00:00:00Z"),
        variables=["tair_2m"],
    )

    assert data["tair_2m"].tolist() == [2.5]
