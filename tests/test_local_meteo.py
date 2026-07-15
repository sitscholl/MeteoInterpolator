import asyncio

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
                "date,tmean,insol",
                "2026-01-01,1.5,10.0",
                "2026-01-02,2.5,11.0",
                "2026-01-03,NA,NA",
                "2026-01-04,3.5,NA",
            ]
        ),
        encoding="utf-8",
    )
    (station_files / "station-b.csv").write_text(
        "\n".join(
            [
                "date,station_id,tmean",
                "2026-01-01,station-b,3.5",
                "2026-01-02,station-b,4.5",
            ]
        ),
        encoding="utf-8",
    )
    (station_files / "station-not-in-metadata.csv").write_text(
        "\n".join(
            [
                "date,tmean",
                "2026-01-01,5.5",
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
        target_timezone="Europe/Rome",
    )

    result = asyncio.run(handler.get_stations_for_sensors(["tair_2m", "solar_radiation"]))

    assert result["tair_2m"] == ["station-a", "station-b", "station-not-in-metadata"]
    assert result["solar_radiation"] == ["station-a"]


def test_local_file_handler_get_data_returns_valid_station_with_renamed_columns(tmp_path):
    stations_path, station_files = _write_local_fixture(tmp_path)
    handler = LocalFileMeteoHandler(
        observations_path=str(station_files / "*.csv"),
        station_metadata_path=str(stations_path),
        target_timezone="Europe/Rome",
    )

    station = asyncio.run(
        handler.get_data(
            station_id="station-a",
            start=pd.Timestamp("2026-01-01", tz="Europe/Rome"),
            end=pd.Timestamp("2026-01-03", tz="Europe/Rome"),
            sensor_codes=["tair_2m", "solar_radiation"],
        )
    )

    assert station.id == "station-a"
    assert station.crs == 4326
    assert station.data["datetime"].dt.tz is not None
    assert station.data["tair_2m"].tolist() == [1.5, 2.5]
    assert station.data["solar_radiation"].tolist() == [10.0, 11.0]
    assert "tmean" not in station.data.columns
    assert "insol" not in station.data.columns


def test_local_file_handler_drops_rows_with_all_requested_sensors_missing(tmp_path):
    stations_path, station_files = _write_local_fixture(tmp_path)
    handler = LocalFileMeteoHandler(
        observations_path=str(station_files / "*.csv"),
        station_metadata_path=str(stations_path),
        target_timezone="Europe/Rome",
    )

    station = asyncio.run(
        handler.get_data(
            station_id="station-a",
            start=pd.Timestamp("2026-01-01", tz="Europe/Rome"),
            end=pd.Timestamp("2026-01-05", tz="Europe/Rome"),
            sensor_codes=["tair_2m", "solar_radiation"],
        )
    )

    assert station.data["datetime"].dt.date.astype(str).tolist() == [
        "2026-01-01",
        "2026-01-02",
        "2026-01-04",
    ]
    assert station.data["tair_2m"].tolist() == [1.5, 2.5, 3.5]
    assert station.data["solar_radiation"].iloc[:2].tolist() == [10.0, 11.0]
    assert pd.isna(station.data["solar_radiation"].iloc[2])


def test_local_file_handler_skips_station_when_requested_sensor_is_missing(tmp_path):
    stations_path, station_files = _write_local_fixture(tmp_path)
    handler = LocalFileMeteoHandler(
        observations_path=str(station_files / "*.csv"),
        station_metadata_path=str(stations_path),
        target_timezone="Europe/Rome",
    )

    station = asyncio.run(
        handler.get_data(
            station_id="station-b",
            start=pd.Timestamp("2026-01-01"),
            end=pd.Timestamp("2026-01-03"),
            sensor_codes=["solar_radiation"],
        )
    )

    assert station.id == "station-b"
    assert station.data is None
