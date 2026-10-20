import asyncio
from zoneinfo import ZoneInfo

import pandas as pd

from src.meteo.base import BaseMeteoHandler
from src.meteo.local import LocalFileMeteoHandler


def _write_local_fixture(tmp_path):
    stations_path = tmp_path / "stations.csv"
    stations_path.write_text(
        "\n".join(
            [
                "y,x,elevation,name,station_id",
                "46.0,11.0,1000.0,Station A,station-a",
                "46.1,11.1,1100.0,Station B,station-b",
                "46.2,11.2,1200.0,Station Without File,station-no-file",
            ]
        ),
        encoding="utf-8",
    )

    station_files = tmp_path / "station_files"
    station_files.mkdir()
    (station_files / "station-a.csv").write_text(
        "\n".join(
            [
                "datetime,tair,rad",
                "2026-01-01T00:00:00Z,1.5,10.0",
                "2026-01-01T00:10:00Z,2.5,11.0",
                "2026-01-01T00:20:00Z,NA,NA",
                "2026-01-01T00:30:00Z,3.5,NA",
            ]
        ),
        encoding="utf-8",
    )
    (station_files / "station-b.csv").write_text(
        "\n".join(
            [
                "datetime,st_id,tair",
                "2026-01-01T00:00:00Z,station-b,3.5",
                "2026-01-01T00:10:00Z,station-b,4.5",
            ]
        ),
        encoding="utf-8",
    )
    (station_files / "station-not-in-metadata.csv").write_text(
        "\n".join(
            [
                "datetime,tair",
                "2026-01-01T00:00:00Z,5.5",
            ]
        ),
        encoding="utf-8",
    )
    return stations_path, station_files


def test_local_file_handler_is_registered():
    assert BaseMeteoHandler.get_handler("local_file") is LocalFileMeteoHandler


def test_local_file_handler_filters_globbed_station_files_and_infers_station_id(tmp_path):
    stations_path, station_files = _write_local_fixture(tmp_path)
    handler = LocalFileMeteoHandler(
        observations_path=str(station_files / "*.csv"),
        station_metadata_path=str(stations_path),
        timezone="Europe/Rome",
    )

    result = asyncio.run(handler.get_stations_for_sensors(["tair_2m", "solar_radiation"]))

    assert result["tair_2m"] == ["station-a", "station-b", "station-not-in-metadata"]
    assert result["solar_radiation"] == ["station-a"]


def test_local_file_handler_get_data_returns_valid_station_with_renamed_columns(tmp_path):
    stations_path, station_files = _write_local_fixture(tmp_path)
    handler = LocalFileMeteoHandler(
        observations_path=str(station_files / "*.csv"),
        station_metadata_path=str(stations_path),
        timezone="Europe/Rome",
    )

    station = asyncio.run(
        handler.get_data(
            station_id="station-a",
            start=pd.Timestamp("2026-01-01 00:00", tz="UTC"),
            end=pd.Timestamp("2026-01-01 00:20", tz="UTC"),
            sensor_codes=["tair_2m", "solar_radiation"],
            target_timezone=ZoneInfo("Europe/Rome"),
        )
    )

    assert station.id == "station-a"
    assert station.crs == 4326
    assert station.data["datetime"].dt.tz is not None
    assert station.data["datetime"].iloc[0] == pd.Timestamp("2026-01-01 01:00", tz="Europe/Rome")
    assert station.data["tair_2m"].tolist() == [1.5, 2.5]
    assert station.data["solar_radiation"].tolist() == [10.0, 11.0]
    assert "tair" not in station.data.columns
    assert "rad" not in station.data.columns


def test_local_file_handler_drops_rows_with_all_requested_sensors_missing(tmp_path):
    stations_path, station_files = _write_local_fixture(tmp_path)
    handler = LocalFileMeteoHandler(
        observations_path=str(station_files / "*.csv"),
        station_metadata_path=str(stations_path),
        timezone="Europe/Rome",
    )

    station = asyncio.run(
        handler.get_data(
            station_id="station-a",
            start=pd.Timestamp("2026-01-01", tz="Europe/Rome"),
            end=pd.Timestamp("2026-01-05", tz="Europe/Rome"),
            sensor_codes=["tair_2m", "solar_radiation"],
            target_timezone=ZoneInfo("Europe/Rome"),
        )
    )

    assert station.data["datetime"].dt.minute.tolist() == [0, 10, 30]
    assert station.data["tair_2m"].tolist() == [1.5, 2.5, 3.5]
    assert station.data["solar_radiation"].iloc[:2].tolist() == [10.0, 11.0]
    assert pd.isna(station.data["solar_radiation"].iloc[2])


def test_local_file_handler_skips_station_when_requested_sensor_is_missing(tmp_path):
    stations_path, station_files = _write_local_fixture(tmp_path)
    handler = LocalFileMeteoHandler(
        observations_path=str(station_files / "*.csv"),
        station_metadata_path=str(stations_path),
        timezone="Europe/Rome",
    )

    station = asyncio.run(
        handler.get_data(
            station_id="station-b",
            start=pd.Timestamp("2026-01-01"),
            end=pd.Timestamp("2026-01-03"),
            sensor_codes=["solar_radiation"],
            target_timezone=ZoneInfo("Europe/Rome"),
        )
    )

    assert station.id == "station-b"
    assert station.data is None


def test_local_sensor_discovery_returns_provider_codes_without_identifiers(tmp_path):
    stations_path, station_files = _write_local_fixture(tmp_path)
    (station_files / "station-a.csv").write_text(
        "datetime,st_id,tair,snow_height,wind_speed,unrelated\n"
        "2026-01-01T00:00:00Z,station-a,2.0,3.0,4.0,5.0\n",
        encoding="utf-8",
    )
    handler = LocalFileMeteoHandler(str(station_files / "*.csv"), str(stations_path), "UTC")
    assert asyncio.run(handler.get_sensors_for_station("station-a")) == {
        "tair", "snow_height", "wind_speed",
    }
    assert asyncio.run(handler.get_sensors_for_station("station-no-file")) == set()
    station = asyncio.run(handler.get_data(
        station_id="station-a", sensor_codes=["snow_height", "wind_speed"],
        start=pd.Timestamp("2026-01-01", tz="UTC"),
        end=pd.Timestamp("2026-01-02", tz="UTC"), target_timezone=ZoneInfo("UTC"),
    ))
    assert station.data["snow_height"].tolist() == [3.0]
    assert station.data["wind_speed"].tolist() == [4.0]


def test_local_loader_uses_st_id_in_multi_station_file(tmp_path):
    stations_path, station_files = _write_local_fixture(tmp_path)
    shared_path = station_files / "shared.csv"
    shared_path.write_text(
        "datetime,st_id,tair\n"
        "2026-01-01T00:00:00Z,station-a,1.0\n"
        "2026-01-01T00:00:00Z,station-b,2.0\n",
        encoding="utf-8",
    )
    handler = LocalFileMeteoHandler(str(shared_path), str(stations_path), "UTC")
    station = asyncio.run(handler.get_data(
        station_id="station-b", sensor_codes=["tair_2m"],
        start=pd.Timestamp("2026-01-01", tz="UTC"),
        end=pd.Timestamp("2026-01-02", tz="UTC"), target_timezone=ZoneInfo("UTC"),
    ))
    assert station.data.columns.is_unique
    assert station.data["station_id"].tolist() == ["station-b"]
    assert station.data["tair_2m"].tolist() == [2.0]
