import asyncio
import importlib
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import src.config as config_module


@pytest.fixture
def script_environment(monkeypatch):
    config = {
        "meteo_data": {
            "loader": {
                "handler": "local_file",
                "observations_path": "configured/*.csv",
                "station_metadata_path": "configured/stations.csv",
                "timezone": "Europe/Rome",
            },
            "database": {"engine": "sqlite:///configured.db"},
            "resampling": {"target_freq": "h", "min_coverage": 0.6},
            "database_ingestion": {
                "overlap_days": 5,
                "default_start_date": datetime(2010, 1, 1, tzinfo=UTC),
            },
        },
        "dem": {"path": "configured-dem.tif"},
    }
    database = Mock()
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
    dem = Mock()

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
        ingestor_factory=ingestor_factory, ingestor=ingestor, dem=dem,
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
        "local_file", observations_path=f"{expected_prefix}/*.csv",
        station_metadata_path=f"{expected_prefix}/stations.csv", timezone="Europe/Rome",
    )
    # Applying CLI overrides must not mutate the source configuration.
    assert env.config["meteo_data"]["loader"]["observations_path"] == "configured/*.csv"
    options = env.ingestor_factory.call_args.kwargs
    assert options["resampler"].target_freq == "h"
    assert options["resampler"].min_coverage == 0.6
    assert options["overlap_days"] == 5
    assert options["default_start_date"] == datetime(2010, 1, 1, tzinfo=UTC)
    env.database.close.assert_called_once()
    if script == "add_local_observations_to_db":
        env.ingestor.upsert_station.assert_awaited_once()
        env.dem.data.close.assert_called_once()
    else:
        env.ingestor.refresh_database.assert_awaited_once_with(
            update_station_metadata=False, persist_requested_end=True,
        )


@pytest.mark.parametrize("script", ["add_local_observations_to_db", "refresh_database"])
def test_scripts_close_database_when_resampler_creation_fails(script_environment, script):
    env = script_environment
    env.config["meteo_data"]["resampling"]["min_coverage"] = 0
    module = env.prepare(script)
    with pytest.raises(ValueError, match="min_coverage"):
        asyncio.run(module._main())
    env.database.close.assert_called_once()


def test_refresh_script_uses_configured_remote_loader(script_environment):
    env = script_environment
    env.config["meteo_data"]["loader"] = {"handler": "province_api"}
    module = env.prepare("refresh_database")
    asyncio.run(module._main())
    env.loader_factory.assert_called_once_with("province_api")
