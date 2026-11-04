import asyncio
import importlib
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock


def test_lifespan_builds_sync_runtime_off_loop_and_closes_resources(
    monkeypatch, tmp_path
):
    api = importlib.import_module("src.api.app")
    path = tmp_path / "config.yaml"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CONFIG_FILE", str(path))
    monkeypatch.setattr(api, "load_dotenv", lambda: None)
    runtime = SimpleNamespace(
        meteo_data_config=SimpleNamespace(scheduler_options={"hour": 3}),
        db=SimpleNamespace(close=Mock()),
        dem=SimpleNamespace(data=SimpleNamespace(close=Mock())),
    )
    coordinator = SimpleNamespace(shutdown=AsyncMock())
    scheduler = Mock(running=True)
    builder_threads = []

    def build_runtime(config_path):
        assert config_path == str(path)
        builder_threads.append(threading.get_ident())
        return runtime

    build = Mock(side_effect=build_runtime)
    monkeypatch.setattr(api.RuntimeContext, "from_config_file", build)
    monkeypatch.setattr(api, "InterpolationCoordinator", Mock(return_value=coordinator))
    monkeypatch.setattr(api, "BackgroundScheduler", Mock(return_value=scheduler))
    app = SimpleNamespace(state=SimpleNamespace())

    async def run():
        loop_thread = threading.get_ident()
        async with api.lifespan(app):
            assert builder_threads and builder_threads[0] != loop_thread
            assert app.state.runtime is runtime
            assert app.state.coordinator is coordinator
            assert app.state.scheduler is scheduler
            scheduler.start.assert_called_once()
            runtime.db.close.assert_not_called()
        scheduler.pause.assert_called_once()
        scheduler.shutdown.assert_called_once_with(wait=True)
        coordinator.shutdown.assert_awaited_once()

    asyncio.run(run())
    build.assert_called_once_with(str(path))
    assert scheduler.add_job.call_args.kwargs["kwargs"] == {
        "config": runtime.meteo_data_config
    }
    runtime.db.close.assert_called_once()
    runtime.dem.data.close.assert_called_once()
