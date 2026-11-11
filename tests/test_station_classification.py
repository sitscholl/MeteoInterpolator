import pandas as pd
import pytest

from src.database.db import InterpolationDB
from src.domain.schemas import _STATION_CATALOG_SCHEMA


@pytest.fixture
def database():
    db = InterpolationDB("sqlite:///:memory:")
    yield db
    db.close()


@pytest.mark.parametrize("blank", [None, pd.NA, float("nan"), "", "  "])
def test_classification_clear_and_only_missing(database, blank):
    database.add_station("a", 46, 11, 900, 4326, "active", station_type=" COLD-POOL ")
    database.update_station("a", only_missing=False, name="Renamed")
    assert database.get_stations("a")[0].station_type == "cold-pool"
    database.update_station("a", station_type=blank)
    assert database.get_stations("a")[0].station_type == "cold-pool"
    database.update_station("a", only_missing=False, station_type=blank)
    assert database.get_stations("a")[0].station_type is None
    database.update_station("a", station_type="cold-pool")
    assert database.get_stations("a")[0].station_type == "cold-pool"


@pytest.mark.parametrize("blank", [None, pd.NA, float("nan")])
def test_catalog_sync_clears_classification(database, blank):
    catalog = pd.DataFrame([dict(
        station_id="001", name="Station", latitude=46., longitude=11.,
        elevation=900., crs=4326, status="active", station_type="cold-pool",
    )])
    database.sync_catalog(catalog)
    assert [s.id for s in database.get_stations(station_type="cold-pool")] == ["001"]
    catalog["station_type"] = blank
    database.sync_catalog(_STATION_CATALOG_SCHEMA.validate(catalog))
    assert database.get_stations("001")[0].station_type is None
    assert database.get_stations(station_type="cold-pool") == []


@pytest.mark.parametrize("value,error", [("unknown", ValueError), (123, TypeError)])
def test_database_rejects_invalid_classification(database, value, error):
    with pytest.raises(error):
        database.add_station("a", 46, 11, 900, 4326, "active", station_type=value)
    assert not database.station_exists("a")
    database.add_station("a", 46, 11, 900, 4326, "active", station_type="cold-pool")
    with pytest.raises(error):
        database.update_station("a", only_missing=False, station_type=value)
    assert database.get_stations("a")[0].station_type == "cold-pool"
