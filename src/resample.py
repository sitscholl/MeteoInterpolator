import pandas as pd

import logging
import functools
import re
from typing import Callable, Any, Iterable

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
    ):

        self.resample_colmap = (
            resample_colmap.copy() if resample_colmap is not None else DEFAULT_RESAMPLE_COLMAP.copy()
        )

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
        return [aggfunc], False

    @staticmethod
    def _agg_name(aggfunc: str | Callable) -> str:
        if isinstance(aggfunc, str):
            return aggfunc.strip().lower()
        return getattr(aggfunc, "__name__", "custom")

    def _get_mapped_aggfunc(self, column: str, resample_colmap: dict[str, str | Callable | list[str | Callable] | tuple[str | Callable, ...]]):
        if column in resample_colmap:
            return resample_colmap[column]
        return None

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
            mapped = self._get_mapped_aggfunc(col, self.resample_colmap)
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
        default_aggfunc: str | Callable = 'mean',
        datetime_col: str = "datetime",
        groupby_cols: list[str] | None = None,
        min_sample_size: int = 1,
    ) -> pd.DataFrame:

        if min_sample_size < 1:
            raise ValueError(f"min_sample_size must be >= 1. Got {min_sample_size}")
        groupby_cols = groupby_cols if groupby_cols else []

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
        named_aggs = self._prepare_named_aggs(value_cols, default_aggfunc, df[datetime_col])

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
        return resampled[final_cols].sort_values(by=[datetime_col] + groupby_cols)


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
