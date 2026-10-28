import asyncio
import importlib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from zoneinfo import ZoneInfo

import pytest
import xarray as xr

import src.config as config_module
from src.database.db import InterpolationDB


@pytest.fixture
def script_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for directory in ('configured', 'override'):
        folder = tmp_path / directory
        folder.mkdir()
        (folder / 'stations.csv').write_text('station_id\n', encoding='utf-8')
    (tmp_path / 'catalog.csv').write_text(
        'station_id,name,latitude,longitude,elevation,crs,status\n'
        'station-a,Station A,46,11,1000,4326,ACTIVE\n'
        'station-b,Station B,46,11,1000,4326,inactive\n',
        encoding='utf-8',
    )
    config = {
        "meteo_data": {
            "station_catalog": "catalog.csv",
            "loader": {
                "handler": "local_file",
                "observations_path": "configured/*.csv",
                "station_metadata_path": "configured/stations.csv",
                "timezone": "Europe/Rome",
            },
            "database": {"engine": "sqlite:///configured.db"},
            "scheduler": {"hour": 3},
            "resampling": { "min_coverage": 0.6},
            "database_ingestion": {
                "overlap_days": 5,
            },
        },
        "dem": {"path": "configured-dem.tif"},
    }
    database = Mock()
    database.get_last_row.return_value = None
    database_factory = Mock(return_value=database)
    monkeypatch.setattr(config_module, "InterpolationDB", database_factory)
    loader = SimpleNamespace(get_station_info=AsyncMock(return_value={"station-a": {}}))
    loader_factory = Mock(return_value=loader)
    monkeypatch.setattr(config_module.BaseMeteoHandler, "create", loader_factory)
    ingestor = SimpleNamespace(
        upsert_station=AsyncMock(return_value=SimpleNamespace(status="completed")),
        refresh_database=AsyncMock(return_value=SimpleNamespace(n_failed=0)),
    )
    ingestor_factory = Mock(return_value=ingestor)
    data = xr.DataArray(
        [[1000.0, 1000.0], [1000.0, 1000.0]], dims=('y', 'x'),
        coords={'y': [45.9, 46.1], 'x': [10.9, 11.1]},
    ).rio.write_crs(4326)
    close_dem = Mock()
    data.set_close(close_dem)
    dem = SimpleNamespace(data=data, crs=data.rio.crs)

    def prepare(module_name, overrides=()):
        module = importlib.import_module(f"scripts.{module_name}")
        monkeypatch.setattr(module, "load_config_file", lambda path: config)
        monkeypatch.setattr(module, "MeteoIngestor", ingestor_factory)
        if module_name == "add_local_observations_to_db":
            monkeypatch.setattr(module, "load_dem", Mock(return_value=dem))
        monkeypatch.setattr("sys.argv", [module_name, "-c", "nested.yaml", *overrides])
        return module

    return SimpleNamespace(
        prepare=prepare, config=config, database=database,
        database_factory=database_factory, loader_factory=loader_factory,
        ingestor_factory=ingestor_factory, ingestor=ingestor, dem=dem, close_dem=close_dem,
    )


@pytest.mark.parametrize("script", ["add_local_observations_to_db", "refresh_database"])
@pytest.mark.parametrize("override_paths", [False, True])
def test_scripts_use_nested_config_and_explicit_path_overrides(script_environment, script, override_paths):
    env = script_environment
    overrides = ["-f", "override/*.csv", "-m", "override/stations.csv"] if override_paths else []
    module = env.prepare(script, overrides)
    asyncio.run(module._main())

    env.database_factory.assert_called_once_with(engine="sqlite:///configured.db")
    expected_prefix = "override" if override_paths else "configured"
    env.loader_factory.assert_called_once_with(
        "local_file", observations_path=[f"{expected_prefix}/*.csv"],
        station_metadata_path=f"{expected_prefix}/stations.csv", timezone=ZoneInfo("Europe/Rome"),
    )
    # Applying CLI overrides must not mutate the source configuration.
    assert env.config["meteo_data"]["loader"]["observations_path"] == "configured/*.csv"
    options = env.ingestor_factory.call_args.kwargs
    assert options["resampler"].min_coverage == 0.6
    assert options["overlap_days"] == 5
    assert options["default_start_date"] == datetime(2000, 1, 1, tzinfo=UTC)
    env.database.close.assert_called_once()
    if script == "add_local_observations_to_db":
        env.ingestor.upsert_station.assert_awaited_once()
        env.close_dem.assert_called_once()
    else:
        env.ingestor.refresh_database.assert_awaited_once_with(
            persist_requested_end=True,
        )


@pytest.mark.parametrize("script", ["add_local_observations_to_db", "refresh_database"])
def test_scripts_validate_config_before_creating_database(script_environment, script):
    env = script_environment
    env.config["meteo_data"]["resampling"]["min_coverage"] = 0
    module = env.prepare(script)
    with pytest.raises(ValueError, match="min_coverage"):
        asyncio.run(module._main())
    env.database_factory.assert_not_called()


def test_refresh_script_uses_configured_remote_loader(script_environment):
    env = script_environment
    env.config["meteo_data"]["loader"] = {"handler": "province_api"}
    module = env.prepare("refresh_database")
    asyncio.run(module._main())
    env.loader_factory.assert_called_once_with(
        "province_api", chunk_size_days=365, timeout=20,
        max_concurrent_requests=5, sleep_time=1,
    )


def test_local_ingestion_discards_remote_options_before_validation(script_environment):
    env = script_environment
    env.config["meteo_data"]["loader"] = {
        "handler": "province_api", "timeout": 40, "stations": "station-a",
    }
    module = env.prepare(
        "add_local_observations_to_db",
        ["-f", "override/*.csv", "-m", "override/stations.csv"],
    )
    asyncio.run(module._main())
    env.loader_factory.assert_called_once_with(
        "local_file", observations_path=["override/*.csv"],
        station_metadata_path="override/stations.csv", timezone=ZoneInfo("UTC"),
    )
    assert env.config["meteo_data"]["loader"] == {
        "handler": "province_api", "timeout": 40, "stations": "station-a",
    }


@pytest.mark.parametrize("script", ["add_local_observations_to_db", "refresh_database"])
def test_scripts_apply_metadata_override_before_validation(script_environment, script):
    env = script_environment
    env.config["meteo_data"]["loader"]["station_metadata_path"] = "missing/stations.csv"
    module = env.prepare(script, ["-m", "override/stations.csv"])
    asyncio.run(module._main())
    assert env.loader_factory.call_args.kwargs["station_metadata_path"] == "override/stations.csv"
    assert env.config["meteo_data"]["loader"]["station_metadata_path"] == "missing/stations.csv"


def test_refresh_rejects_path_overrides_for_remote_loader(script_environment):
    env = script_environment
    env.config["meteo_data"]["loader"] = {"handler": "province_api"}
    module = env.prepare("refresh_database", ["-m", "override/stations.csv"])
    with pytest.raises(ValueError, match="require a local_file loader"):
        asyncio.run(module._main())
    env.database_factory.assert_not_called()


def test_local_import_syncs_real_catalog_before_ingestion(script_environment, tmp_path):
    env = script_environment
    engine = f"sqlite:///{(tmp_path / 'catalog-sync.db').as_posix()}"
    database = InterpolationDB(engine)
    database.add_station('station-a', 46, 11, 900, 4326, status='inactive')
    database.add_station('removed-station', 46, 11, 900, 4326, status='active')
    env.database_factory.return_value = database
    env.loader_factory.return_value.get_station_info.return_value = {
        'station-a': {}, 'station-b': {}, 'unknown-station': {},
    }

    async def ingest(**kwargs):
        assert kwargs['station_id'] == 'station-a'
        statuses = {
            station.id: station.status
            for station in database.get_stations(include_inactive=True)
        }
        assert statuses == {
            'station-a': 'active', 'station-b': 'inactive',
            'removed-station': 'inactive',
        }
        return SimpleNamespace(status='completed')

    env.ingestor.upsert_station.side_effect = ingest
    module = env.prepare('add_local_observations_to_db')
    asyncio.run(module._main())
    env.ingestor.upsert_station.assert_awaited_once()

    # Verify catalog synchronization committed, independently of ingestion.
    reopened = InterpolationDB(engine)
    try:
        assert [station.id for station in reopened.get_stations()] == ['station-a']
        assert len(reopened.get_stations(include_inactive=True)) == 3
    finally:
        reopened.close()


@pytest.mark.parametrize('has_observations', [False, True])
def test_local_skip_existing_checks_observations(script_environment, has_observations):
    env = script_environment
    env.database.get_last_row.return_value = object() if has_observations else None
    module = env.prepare('add_local_observations_to_db', ['--skip_existing'])
    asyncio.run(module._main())
    env.database.sync_catalog.assert_called_once()
    env.database.get_last_row.assert_called_once_with('station-a')
    assert env.ingestor.upsert_station.await_count == (0 if has_observations else 1)


def test_local_sync_failure_stops_ingestion_and_closes_database(script_environment):
    env = script_environment
    env.database.sync_catalog.side_effect = RuntimeError('sync failed')
    module = env.prepare('add_local_observations_to_db')
    with pytest.raises(RuntimeError, match='sync failed'):
        asyncio.run(module._main())
    env.ingestor_factory.assert_not_called()
    env.ingestor.upsert_station.assert_not_awaited()
    env.database.close.assert_called_once()


def test_local_invalid_dem_catalog_stops_before_sync(script_environment):
    env = script_environment
    path = Path(env.config['meteo_data']['station_catalog'])
    path.write_text(path.read_text(encoding='utf-8').replace('46,11', '50,11'), encoding='utf-8')
    module = env.prepare('add_local_observations_to_db')
    with pytest.raises(ValueError, match='outside.*station-a'):
        asyncio.run(module._main())
    env.database.sync_catalog.assert_not_called()
    env.ingestor.upsert_station.assert_not_awaited()
    env.database.close.assert_called_once()
    env.close_dem.assert_called_once()
