import asyncio
from unittest.mock import AsyncMock

import pandas as pd
import pytest
import xarray as xr
from pandera.errors import SchemaError

from src.aoi import AOI
from src.domain.dem import DEM
from src.runtime import RuntimeContext, load_station_catalog


@pytest.fixture
def dem():
    data = xr.DataArray(
        [[1000.0, 1000.0], [1000.0, 1000.0]], dims=('y', 'x'),
        coords={'y': [45.9, 46.1], 'x': [10.9, 11.1]},
    ).rio.write_crs(4326)
    return DEM(path='synthetic-dem', data=data, fingerprint='synthetic')


@pytest.fixture
def catalog_file(tmp_path):
    def write(rows=None):
        catalog = pd.DataFrame([
            dict(station_id='001', name='First', latitude=46.0, longitude=11.0,
                 elevation=1000.0, crs=4326, status='ACTIVE'),
            dict(station_id='002', name='Second', latitude=46.05, longitude=11.05,
                 elevation=1200.0, crs=4326, status='inactive'),
        ])
        if rows is not None:
            catalog = rows(catalog)
        path = tmp_path / 'stations.csv'
        catalog.to_csv(path, index=False)
        return path
    return write


def _load(path, dem, crs=4326):
    return load_station_catalog(path, AOI.from_array(dem.data), crs)


def _config(path):
    return {
        'meteo_data': {
            'station_catalog': str(path), 'loader': {'handler': 'province_api'},
            'scheduler': {'hour': 3}, 'database_ingestion': {'overlap_days': 3},
            'resampling': {'min_coverage': 0.7},
            'database': {'engine': 'sqlite:///:memory:'},
        },
        'dem': {'path': 'synthetic-dem'}, 'cache': {'enabled': False},
        'interpolation': {'vertical_model': {'type': 'linear'}},
    }


def test_catalog_preserves_ids_metadata_and_normalizes_status(catalog_file, dem):
    catalog = _load(catalog_file(), dem, crs=32632)
    assert catalog.station_id.tolist() == ['001', '002']
    assert catalog.status.tolist() == ['active', 'inactive']
    assert catalog.name.tolist() == ['First', 'Second']
    assert catalog.crs.to_epsg() == 32632
    geographic = catalog.to_crs(4326)
    assert geographic.geometry.x.tolist() == pytest.approx(catalog.longitude.tolist())
    assert geographic.geometry.y.tolist() == pytest.approx(catalog.latitude.tolist())


@pytest.mark.parametrize('empty', [False, True])
def test_catalog_requires_at_least_one_active_station(catalog_file, dem, empty):
    path = catalog_file(lambda df: df.iloc[:0] if empty else df.assign(status='inactive'))
    with pytest.raises(ValueError, match='No active stations'):
        _load(path, dem)


@pytest.mark.parametrize('outside_status', ['active', 'inactive'])
def test_catalog_rejects_any_station_outside_dem(catalog_file, dem, outside_status):
    def outside(df):
        df.loc[1, ['longitude', 'status']] = [15.0, outside_status]
        return df
    with pytest.raises(ValueError, match='outside.*002'):
        _load(catalog_file(outside), dem)


@pytest.mark.parametrize('column,value', [
    ('station_id', '001'), ('status', 'unknown'), ('crs', 3857),
    ('longitude', float('inf')), ('latitude', float('nan')),
    ('elevation', float('inf')),
])
def test_catalog_rejects_invalid_entries(catalog_file, dem, column, value):
    def invalid(df):
        df.loc[1, column] = value
        return df
    with pytest.raises(SchemaError):
        _load(catalog_file(invalid), dem)


def test_runtime_retains_catalog_but_selects_only_active_ids(monkeypatch, tmp_path, catalog_file, dem):
    config = _config(catalog_file())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('src.runtime.load_dem', lambda **kwargs: dem)
    discovery = AsyncMock(side_effect=AssertionError('Startup must not contact provider'))
    monkeypatch.setattr('src.meteo.province.ProvinceAPI.get_station_info', discovery)
    runtime = RuntimeContext(config=config)
    try:
        asyncio.run(runtime.initialize_runtime(config))
        assert runtime.initialized
        assert runtime.active_stations == ['001']
        assert runtime.station_catalog.station_id.tolist() == ['001', '002']
        assert [station.id for station in runtime.db.get_stations()] == ['001']
        assert len(runtime.db.get_stations(include_inactive=True)) == 2
        discovery.assert_not_awaited()
    finally:
        if runtime.db is not None:
            runtime.db.close()
        dem.data.close()


def test_runtime_rejects_inactive_catalog_before_database_sync(monkeypatch, catalog_file, dem):
    config = _config(catalog_file(lambda df: df.assign(status='inactive')))
    monkeypatch.setattr('src.runtime.load_dem', lambda **kwargs: dem)
    runtime = RuntimeContext(config=config)
    with pytest.raises(ValueError, match='No active stations'):
        asyncio.run(runtime.initialize_runtime(config))
    assert runtime.db is None
    assert not runtime.initialized
