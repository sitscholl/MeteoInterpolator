import pandas as pd

from src.meteo.station import MeteoData
from src.validate.date import localize_datetime_string


def test_localize_datetime_string_uses_configured_timezone():
    ts = localize_datetime_string("2026-05-13", "Europe/Rome")

    assert ts.tzinfo is not None
    assert str(ts.tzinfo) == "Europe/Rome"


def test_iter_samples_is_single_parameter_and_end_exclusive():
    meteo_data = MeteoData(
        ids=["a", "b", "c"],
        coords=[(11.0, 46.0), (11.1, 46.1), (11.2, 46.2)],
        elevation=[1000.0, 1200.0, 1400.0],
        crs=4326,
        data=[
            pd.DataFrame(
                {
                    "datetime": pd.to_datetime(["2026-05-13", "2026-05-14"]),
                    "tair_2m": [10.0, 11.0],
                }
            ),
            pd.DataFrame(
                {
                    "datetime": pd.to_datetime(["2026-05-13", "2026-05-14"]),
                    "tair_2m": [9.0, 10.0],
                }
            ),
            pd.DataFrame(
                {
                    "datetime": pd.to_datetime(["2026-05-13", "2026-05-14"]),
                    "tair_2m": [8.0, 9.0],
                }
            ),
        ],
    )

    samples = list(
        meteo_data.iter_samples(
            start=pd.Timestamp("2026-05-13"),
            end=pd.Timestamp("2026-05-14"),
            param="tair_2m",
        )
    )

    assert len(samples) == 1
    interp_date, X, y = samples[0]
    assert interp_date == pd.Timestamp("2026-05-13")
    assert X.shape == (3, 1)
    assert y.tolist() == [10.0, 9.0, 8.0]
