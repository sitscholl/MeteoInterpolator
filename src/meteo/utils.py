import pandas as pd
import numpy as np

def aggregate_precipitation(data: pd.DataFrame, freq: str) -> pd.DataFrame:

    required_cols = ['datetime', 'station_id', 'precipitation']
    missing_cols = [i for i in required_cols if i not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in aggregating precipitation: {missing_cols}")

    target_timezone = data['datetime'].dt.tz
    if target_timezone is None:
        raise ValueError("Datetime column must be timezone aware to aggregate precipitation.")
    
    timestamps_utc = data["datetime"].dt.tz_convert("UTC")
    on_grid = timestamps_utc.eq(timestamps_utc.dt.floor(freq))

    tbl_on_grid = data.loc[on_grid].copy()

    rain = (
        data.assign(
            datetime=data["datetime"].dt.tz_convert("UTC")
        )
        .set_index("datetime")
        )
    rain['precipitation'] = rain['precipitation'].where((rain['precipitation'] >= 0) & np.isfinite(rain['precipitation'])) #mask out negative precipitation values
    rain =(
        rain.groupby("station_id")["precipitation"]
        .resample(freq, closed="right", label="right")
        .sum(min_count=1)
        .reset_index()
    )
    rain["datetime"] = rain["datetime"].dt.tz_convert(target_timezone)

    result = tbl_on_grid.drop(columns="precipitation").merge(
        rain,
        on=["station_id", "datetime"],
        how="outer",
        validate="one_to_one",
    )

    return result.sort_values(["datetime", "station_id"]).reset_index(drop=True)