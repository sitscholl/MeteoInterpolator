import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from src.domain.meteo_data import MeteoData
from src.resample import MeteoResampler


def test_min_samples_for_coverage_uses_source_and_target_frequency():
    resampler = MeteoResampler(target_freq="D", min_coverage=0.4)

    assert resampler.min_samples_for_coverage("10min") == 58


def test_resample_meteo_data_skips_when_source_matches_target_frequency():
    stations = gpd.GeoDataFrame(
        {"station_id": ["a"], "elevation": [1000.0]},
        geometry=[Point(11.0, 46.0)],
        crs=4326,
    ).set_index("station_id")
    observations = pd.DataFrame(
        {
            "station_id": ["a"],
            "datetime": [pd.Timestamp("2026-01-01", tz="Europe/Rome")],
            "tair_2m": [1.5],
        }
    )
    meteo_data = MeteoData(stations=stations, observations=observations)
    resampler = MeteoResampler(target_freq="D", min_coverage=1.0)

    result = resampler.resample_meteo_data(
        meteo_data,
        source_freq="D",
        datetime_col="datetime",
        groupby_cols=["station_id"],
    )

    assert result is meteo_data


def test_resampling_masks_values_below_min_coverage():
    resampler = MeteoResampler(target_freq="D", min_coverage=0.5)
    data = pd.DataFrame(
        {
            "station_id": ["a", "a"],
            "datetime": pd.to_datetime(
                ["2026-01-01 00:00", "2026-01-01 00:10"]
            ).tz_localize("Europe/Rome"),
            "tair_2m": [1.0, 2.0],
        }
    )

    result = resampler.apply_resampling(
        data,
        freq="D",
        datetime_col="datetime",
        groupby_cols=["station_id"],
        min_sample_size=resampler.min_samples_for_coverage("10min"),
    )

    assert pd.isna(result.loc[0, "tair_2m"])
