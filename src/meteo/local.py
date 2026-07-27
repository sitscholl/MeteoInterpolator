import glob
import logging
from pathlib import Path
from typing import Any, Dict
from zoneinfo import ZoneInfo

import pandas as pd

from .base import BaseMeteoHandler
from .sensors import SENSORS

logger = logging.getLogger(__name__)


class LocalFileMeteoHandler(BaseMeteoHandler):
    provider_name = "local_file"

    def __init__(
        self,
        observations_path: str | list[str],
        station_metadata_path: str,
        timezone: str,
        **kwargs,
    ):
        self.timezone = timezone
        self.observations_path = observations_path
        self.station_metadata_path = station_metadata_path

        if self.station_metadata_path is None:
            raise ValueError(
                "LocalFileMeteoHandler requires station_metadata_path."
            )

        self._station_info: dict[str, dict[str, Any]] | None = None
        self._observation_files: list[Path] | None = None
        self._station_file_map: dict[str, list[Path]] | None = None
        self._header_cache: dict[Path, list[str]] = {}
        self._data_cache: dict[Path, pd.DataFrame] = {}

    @classmethod
    def name(cls):
        return "local_file"

    @property
    def freq(self):
        return "D"

    @property
    def inclusive(self):
        return "left"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def _expand_observation_files(self) -> list[Path]:
        if self._observation_files is not None:
            return self._observation_files

        patterns = (
            [self.observations_path]
            if isinstance(self.observations_path, str)
            else list(self.observations_path)
        )
        files: list[Path] = []
        for pattern in patterns:
            matches = [Path(path) for path in glob.glob(str(pattern))]
            if matches:
                files.extend(matches)
                continue
            path = Path(pattern)
            if path.exists():
                files.append(path)

        unique_files = sorted({path.resolve() for path in files if path.is_file()})
        if not unique_files:
            raise ValueError(f"No observation files matched {self.observations_path!r}")

        self._observation_files = unique_files
        return self._observation_files

    def _read_header(self, path: Path) -> list[str]:
        if path not in self._header_cache:
            self._header_cache[path] = pd.read_csv(path, nrows=0).columns.tolist()
        return self._header_cache[path]

    def _station_ids_for_file(self, path: Path) -> list[str]:
        header = self._read_header(path)
        if "station_id" not in header:
            return [path.stem]

        ids = pd.read_csv(path, usecols=["station_id"])["station_id"].dropna().astype(str).unique()
        return sorted(ids.tolist())

    def _build_station_file_map(self) -> dict[str, list[Path]]:
        if self._station_file_map is not None:
            return self._station_file_map

        station_file_map: dict[str, list[Path]] = {}
        for path in self._expand_observation_files():
            for station_id in self._station_ids_for_file(path):
                station_file_map.setdefault(station_id, []).append(path)

        self._station_file_map = station_file_map
        return self._station_file_map

    def _read_observation_file(self, path: Path) -> pd.DataFrame:
        if path not in self._data_cache:
            data = pd.read_csv(path, na_values=["NA", "NaN", "nan", ""])
            if "station_id" not in data.columns:
                data["station_id"] = path.stem
            data["station_id"] = data["station_id"].astype(str)
            data.rename(columns=SENSORS.rename_map(self.provider_name), inplace=True)
            self._data_cache[path] = data
        return self._data_cache[path]

    def _normalize_sensor_codes(self, sensor_codes: object) -> list[str]:
        if isinstance(sensor_codes, str):
            sensor_codes = [sensor_codes]
        if sensor_codes is None:
            return []
        provider_codes = SENSORS.resolve_provider_codes(self.provider_name, sensor_codes)
        rename_map = SENSORS.rename_map(self.provider_name)
        return [rename_map.get(code, code) for code in provider_codes]

    def _normalize_observations(
        self,
        raw_data: pd.DataFrame,
    ) -> pd.DataFrame:
        data = raw_data.copy()
        if "datetime" not in data.columns:
            raise ValueError("Local observation files must contain a datetime column or a column renamed to datetime.")
        if "station_id" not in data.columns:
            raise ValueError("Local observation files must contain station_id or use one station per filename.")
        data["datetime"] = pd.to_datetime(data["datetime"])
        data["station_id"] = data["station_id"].astype(str)
        return data

    def _to_target_timestamp(self, value, target_timezone: ZoneInfo) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize(target_timezone)
        return timestamp.tz_convert(target_timezone)

    def _drop_empty_sensor_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        sensor_columns = [col for col in data.columns if col not in ("datetime", "station_id")]
        if not sensor_columns:
            return data

        n_before = len(data)
        data = data.dropna(subset=sensor_columns, how="all")
        n_dropped = n_before - len(data)
        if n_dropped > 0:
            logger.debug("Dropped %s local observation row(s) with only missing sensor values", n_dropped)
        return data

    async def get_station_info(self, station_id: str | None = None) -> Dict[str, Any]:
        if self._station_info is None:
            metadata = pd.read_csv(self.station_metadata_path)
            if "station_id" not in metadata.columns:
                raise ValueError("Station metadata file must contain a station_id column.")
            missing = [column for column in ("x", "y") if column not in metadata.columns]
            if missing:
                raise ValueError(f"Station metadata file is missing required column(s): {missing}")

            metadata["station_id"] = metadata["station_id"].astype(str)
            info: dict[str, dict[str, Any]] = {}
            for row in metadata.to_dict(orient="records"):
                station = {
                    "id": row["station_id"],
                    "x": row["x"],
                    "y": row["y"],
                    "elevation": row.get("elevation"),
                    "name": row.get("name"),
                }
                info[row["station_id"]] = station
            self._station_info = info

        if station_id is not None:
            return self._station_info.get(str(station_id), {})
        return self._station_info

    async def get_stations_for_sensors(self, sensors: str | list[str]) -> dict[str, list[str]]:
        requested_sensors = self._normalize_sensor_codes(sensors)
        result = {sensor: [] for sensor in requested_sensors}

        for station_id, paths in self._build_station_file_map().items():
            available_columns: set[str] = set()
            for path in paths:
                available_columns.update(SENSORS.rename_map(self.provider_name).get(col, col) for col in self._read_header(path))
            for sensor in requested_sensors:
                if sensor in available_columns:
                    result[sensor].append(station_id)

        return {sensor: sorted(station_ids) for sensor, station_ids in result.items()}

    async def get_raw_data(
        self,
        station_id: str,
        start,
        end,
        sensor_codes: list[str] | None = None,
        **kwargs,
    ):
        station_id = str(station_id)
        st_metadata = await self.get_station_info(station_id)
        if not st_metadata:
            raise ValueError(f"No station metadata found for station {station_id}")
        st_metadata = dict(st_metadata)
        st_metadata["crs"] = 4326

        paths = self._build_station_file_map().get(station_id, [])
        if not paths:
            logger.warning("No local observation file found for station %s", station_id)
            return None, st_metadata

        frames = []
        for path in paths:
            data = self._read_observation_file(path)
            frames.append(data.loc[data["station_id"].astype(str) == station_id].copy())

        raw_data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if raw_data.empty:
            logger.warning("No local observations found for station %s", station_id)
            return None, st_metadata

        data = self._normalize_observations(raw_data, self.timezone)
        
        if data["datetime"].dt.tz is None:
            data["datetime"] = data["datetime"].dt.tz_localize(self.timezone)
        elif data['datetime'].dt.tz != self.timezone:
            raise ValueError(f"Timezone from loaded data does not match configured timezone. Got {data['datetime'].dt.tz} vs {self.timezone}")

        start_ts = self._to_target_timestamp(start, self.timezone)
        end_ts = self._to_target_timestamp(end, self.timezone)
        data = data.loc[(data["datetime"] >= start_ts) & (data["datetime"] < end_ts)].copy()

        if sensor_codes is not None:
            selected_sensors = self._normalize_sensor_codes(sensor_codes)
            available_sensors = [col for col in selected_sensors if col in data.columns]
            if not available_sensors:
                logger.warning("None of the requested sensors are available for station %s. Skipping", station_id)
                return None, st_metadata
            columns = ["datetime", "station_id", *available_sensors]
            data = data.loc[:, columns]

        data = self._drop_empty_sensor_rows(data)

        if data.empty:
            logger.warning("No local observations found for station %s in requested date range", station_id)
            return None, st_metadata

        return data, st_metadata

    def transform(self, raw_data: pd.DataFrame | None, target_timezone: ZoneInfo):
        if raw_data is None:
            return None

        data = raw_data.copy()
        data["datetime"] = data["datetime"].dt.tz_convert(target_timezone)
        if data[["datetime", "station_id"]].duplicated().any():
            logger.warning("Found duplicate local observations for ['datetime', 'station_id']. They will be dropped")
            data = data.drop_duplicates(subset=["datetime", "station_id"])

        data = self._drop_empty_sensor_rows(data)
        return data.reset_index(drop=True)
