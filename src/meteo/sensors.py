from collections.abc import Iterable
from dataclasses import dataclass, field

PROVIDER_SENSOR_MAPS: dict[str, dict[str, str]] = {
    "province_api": {
        "LT": "tair_2m",
        "LF": "relative_humidity",
        "N": "precipitation",
        "WG": "wind_speed",
        "WR": "wind_direction",
        "WG.BOE": "wind_gust",
        "LD.RED": "air_pressure",
        "SD": "sun_duration",
        "GS": "solar_radiation",
        "HS": "snow_height",
        "W": "water_level",
        "Q": "discharge",
    },
    "local_file": {
        "tair": "tair_2m",
        "rad": "solar_radiation",
        "relative_air_humidity": "relative_humidity",
        "preci": "precipitation",
        "rad_h": "sun_duration",
        "wind_speed_boe": "wind_gust",
        "wind_dir": "wind_direction",
    }
}

PROVIDER_COLUMN_RENAMES: dict[str, dict[str, str]] = {
    "province_api": {
        "DATE": "datetime",
    },
    "local_file": {
        "st_id": "station_id",
        "date": "datetime"
    }
}

@dataclass
class SensorManager:
    sensor_maps: dict[str, dict[str, str]]
    extra_renames: dict[str, dict[str, str]] | None = None
    _reverse: dict[str, dict[str, str]] = field(init=False, repr=False)

    def __post_init__(self):
        reverse: dict[str, dict[str, str]] = {}
        for provider, mapping in self.sensor_maps.items():
            reverse_map: dict[str, str] = {}
            for provider_name, harmonized in mapping.items():
                if harmonized in reverse_map:
                    raise ValueError(
                        f"Duplicate harmonized name '{harmonized}' for provider '{provider}'"
                    )
                reverse_map[harmonized] = provider_name
            reverse[provider] = reverse_map
        object.__setattr__(self, "_reverse", reverse)

    def harmonized_names(self, provider: str) -> list[str]:
        return sorted(set(self.sensor_maps.get(provider, {}).values()))

    def rename_map(self, provider: str, include_extra: bool = True) -> dict[str, str]:
        merged = dict(self.sensor_maps.get(provider, {}))
        if include_extra and self.extra_renames is not None:
            merged.update(self.extra_renames.get(provider, {}))
        return merged

    def resolve_provider_codes(self, provider: str, sensor_codes: Iterable[str]) -> list[str]:
        codes = self._normalize_codes(sensor_codes)
        mapping = self.sensor_maps.get(provider, {})
        reverse = self._reverse.get(provider, {})

        if not codes:
            return []

        if all(code in reverse for code in codes):
            return [reverse[code] for code in codes]
        if all(code in mapping for code in codes):
            return list(codes)

        missing = [code for code in codes if code not in reverse and code not in mapping]
        raise KeyError(f"Unknown sensor codes for provider '{provider}': {missing}")

    def _normalize_codes(self, sensor_codes: Iterable[str]) -> list[str]:
        if isinstance(sensor_codes, str):
            sensor_codes = [sensor_codes]
        if not isinstance(sensor_codes, Iterable):
            raise TypeError(f"Sensor codes must be iterable. Got {type(sensor_codes)}")
        normalized: list[str] = []
        seen = set()
        for code in sensor_codes:
            if not isinstance(code, str):
                raise TypeError(f"Sensor code must be str. Got {type(code)}")
            code = code.strip()
            if not code or code in seen:
                continue
            seen.add(code)
            normalized.append(code)
        return normalized

SENSORS = SensorManager(
    sensor_maps=PROVIDER_SENSOR_MAPS,
    extra_renames=PROVIDER_COLUMN_RENAMES,
)
