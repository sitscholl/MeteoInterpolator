import pandas as pd

import logging
import math
from typing import Callable, Any, Iterable

from .domain.meteo_data import MeteoData

logger = logging.getLogger(__name__)

DEFAULT_RESAMPLE_COLMAP: dict[str, str | list[str]] = {
    "tair_2m": ["mean", 'min', 'max'],
    "relative_humidity": "mean",
    "precipitation": "sum",
    "sun_duration": "sum",
    "solar_radiation": "sum",
}

class MeteoResampler:

    _AGG_STR_TO_FUNC: dict[str, str | Callable[[pd.Series], Any]] = {
        "mean": "mean",
        "sum": "sum",
        "max": "max",
        "min": "min",
        "median": "median",
        "first": "first",
        "last": "last",
    }

    def __init__(
        self,
        resample_colmap: dict[str, str | Callable | list[str | Callable] | tuple[str | Callable, ...]] | None = None,
        default_aggfunc: str | Callable = 'mean',
        target_freq: str = "D",
        min_coverage: float = 0.4,
    ):

        self.resample_colmap = (
            resample_colmap.copy() if resample_colmap is not None else DEFAULT_RESAMPLE_COLMAP.copy()
        )
        self.default_aggfunc = default_aggfunc
        self.target_freq = target_freq
        self.min_coverage = min_coverage

        if not 0 < self.min_coverage <= 1:
            raise ValueError(f"min_coverage must be > 0 and <= 1. Got {self.min_coverage}")

    @staticmethod
    def _to_offset(freq: str):
        return pd.tseries.frequencies.to_offset(freq)

    @classmethod
    def frequencies_equal(cls, left: str, right: str) -> bool:
        return cls._to_offset(left) == cls._to_offset(right)

    def min_samples_for_coverage(self, source_freq: str, target_freq: str | None = None) -> int:
        target_freq = target_freq or self.target_freq
        source_offset = self._to_offset(source_freq)
        target_offset = self._to_offset(target_freq)

        try:
            expected_samples = target_offset.nanos / source_offset.nanos
        except ValueError as exc:
            raise ValueError(
                "min_coverage requires fixed source and target frequencies. "
                f"Got source_freq={source_freq!r}, target_freq={target_freq!r}."
            ) from exc

        return max(1, math.ceil(expected_samples * self.min_coverage))

    def _resolve_aggfunc(self, aggfunc: str | Callable):
        if callable(aggfunc):
            return aggfunc

        aggfunc_norm = aggfunc.strip().lower()
        
        if aggfunc_norm not in self._AGG_STR_TO_FUNC:
            raise ValueError(
                f"Invalid aggregation function '{aggfunc}'. "
                f"Choose one of {sorted(self._AGG_STR_TO_FUNC.keys())} or pass a callable."
            )
        return self._AGG_STR_TO_FUNC[aggfunc_norm]

    @staticmethod
    def _normalize_agg_list(aggfunc: str | Callable | Iterable[str | Callable]):
        if isinstance(aggfunc, (list, tuple)):
            return list(aggfunc), True
        if isinstance(aggfunc, Iterable) and not isinstance(aggfunc, (str, bytes)):
            return list(aggfunc), True
        return [aggfunc], False

    @staticmethod
    def _agg_name(aggfunc: str | Callable) -> str:
        if isinstance(aggfunc, str):
            return aggfunc.strip().lower()
        return getattr(aggfunc, "__name__", "custom")

    def _prepare_named_aggs(
        self,
        value_cols: list[str],
        default_aggfunc: Any,
    ) -> dict[str, tuple[str, Any]]:
        """
        Creates a flat dictionary for Named Aggregation.
        Example output: {'tair_2m_mean': ('tair_2m', 'mean'), 'tair_2m_max': ('tair_2m', 'max')}
        """
        named_aggs = {}
        
        for col in value_cols:
            mapped = self.resample_colmap.get(col)
            if mapped is None:
                if default_aggfunc is None:
                    continue
                mapped = default_aggfunc

            agg_list, _ = self._normalize_agg_list(mapped)
            
            for agg_item in agg_list:
                resolved_func = self._resolve_aggfunc(agg_item)
                suffix = self._agg_name(agg_item)
                
                # If only one agg and no suffix forced, use original name, otherwise append suffix
                out_name = f"{col}_{suffix}" if (len(agg_list) > 1) else col
                named_aggs[out_name] = (col, resolved_func)
        
        return named_aggs

    def apply_resampling(
        self,
        data: pd.DataFrame,
        freq: str,
        datetime_col: str = "datetime",
        groupby_cols: list[str] | None = None,
        min_sample_size: int = 1,
    ) -> pd.DataFrame:

        if min_sample_size < 1:
            raise ValueError(f"min_sample_size must be >= 1. Got {min_sample_size}")
        groupby_cols = list(groupby_cols) if groupby_cols else []

        if data.empty:
            return data.copy()
        df = data.copy()

        required_cols = [datetime_col] + groupby_cols
        missing_required = [c for c in required_cols if c not in df.columns]
        if missing_required:
            raise ValueError(f"Cannot resample without required columns: {missing_required}")

        # 2. Data Preparation
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
        df = df.dropna(subset=[datetime_col])

        # 3. Build Named Aggregations
        value_cols = [c for c in df.columns if c not in required_cols]
        named_aggs = self._prepare_named_aggs(value_cols, self.default_aggfunc)

        if not named_aggs:
            return df[required_cols].drop_duplicates().sort_values(by=[datetime_col] + groupby_cols)

        # 4. Perform Resampling
        # We use pd.Grouper inside groupby to handle everything in one go
        grouper = [pd.Grouper(key=datetime_col, freq=freq)] + groupby_cols
        resampled = df.groupby(grouper, dropna=False).agg(**named_aggs)

        # 5. Handle Min Sample Size (Efficiently)
        if min_sample_size > 1:
            # Count non-NA values for the original columns
            counts = df.groupby(grouper, dropna=False)[value_cols].count()
            
            # For every new column, mask it based on the count of its source column
            for out_col, (src_col, _) in named_aggs.items():
                resampled[out_col] = resampled[out_col].where(counts[src_col] >= min_sample_size)

        # 6. Final Formatting
        resampled = resampled.reset_index()
        
        # Ensure column order matches groupby_cols + datetime + values
        final_cols = groupby_cols + [datetime_col] + list(named_aggs.keys())
        resampled_prepared = resampled[final_cols].sort_values(by=[datetime_col] + groupby_cols)
        
        if 'tair_2m_mean' in resampled_prepared.columns:
            resampled_prepared = resampled_prepared.rename(columns = {'tair_2m_mean': 'tair_2m'})

        return resampled_prepared

    def resample_meteo_data(
        self,
        meteo_data: MeteoData,
        freq: str | None = None,
        datetime_col: str = "datetime",
        groupby_cols: list[str] | None = None,
        min_sample_size: int | None = None,
        source_freq: str | None = None,
    ) -> MeteoData:

        if not isinstance(meteo_data, MeteoData):
            raise TypeError(f"Expected MeteoData. Got {type(meteo_data)}")

        if meteo_data.n_stations == 0:
            return meteo_data

        target_freq = freq or self.target_freq
        if source_freq is not None and self.frequencies_equal(source_freq, target_freq):
            logger.info("Skipping meteo resampling because source frequency %s already matches target frequency %s", source_freq, target_freq)
            return meteo_data

        if min_sample_size is None:
            min_sample_size = (
                self.min_samples_for_coverage(source_freq, target_freq)
                if source_freq is not None
                else 1
            )

        resampled = self.apply_resampling(
            meteo_data.observations,
            freq=target_freq,
            datetime_col=datetime_col,
            groupby_cols=groupby_cols,
            min_sample_size=min_sample_size,
        )

        return MeteoData(
            stations=meteo_data.stations,
            observations=resampled,
        )


if __name__ == "__main__":
    import numpy as np

    logging.basicConfig(level=logging.INFO, force=True)

    rng = pd.date_range("2025-01-01 00:00:00", periods=48, freq="H", tz="UTC")
    df = pd.DataFrame(
        {
            "station_id": ["s1"] * len(rng),
            "model": ["m1"] * len(rng),
            "datetime": rng,
            "tair_2m": np.linspace(0, 20, len(rng)),
            "weather_code": [0, 1, 2, 2] * 12,
            "wind_speed": np.random.uniform(0, 10, len(rng)),
        }
    )

    resampler = MeteoResampler()
    out = resampler.apply_resampling(df, freq="1D", min_sample_size = 10)
    print(out.head())
