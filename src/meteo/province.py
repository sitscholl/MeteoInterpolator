import pandas as pd

import datetime
import requests
import pytz
import time
from typing import Dict, Any
import logging

from .base import BaseMeteoHandler
from ..utils import split_dates

logger = logging.getLogger(__name__)

PROVINCE_RENAME = {
    "DATE": "datetime",
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
    "Q": "discharge"
}

class ProvinceAPI(BaseMeteoHandler):

    provider_name = 'province_api'

    base_url = "http://daten.buergernetz.bz.it/services/meteo/v1"
    sensors_url = base_url + "/sensors"
    stations_url = base_url + "/stations"
    timeseries_url = base_url + "/timeseries"

    def __init__(self, chunk_size_days: int = 365, **kwargs):
        self.timezone = "Europe/Rome"
        self.chunk_size_days = chunk_size_days
        self.station_codes = None
        self.station_sensors = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @property
    def freq(self):
        return "10min"

    @property
    def inclusive(self):
        return "both"

    def get_sensors_for_station(self, station_code: str):
        if self.station_sensors.get(station_code) is not None:
            return self.station_sensors.get(station_code)

        response = requests.get(self.sensors_url, params = {"station_code": station_code})
        response.raise_for_status()

        sensors_list = set([i['TYPE'] for i in response.json()])
        self.station_sensors[station_code] = sensors_list

        return sensors_list

    def get_station_codes(self):
        if self.station_codes is not None:
            return self.station_codes

        response = requests.get(self.stations_url)
        response.raise_for_status()
        stations_list = set([i['properties']['SCODE'] for i in response.json()['features']])
        self.station_codes = stations_list
        return stations_list

    def get_station_info(self, station_id: str) -> Dict[str, Any]:

        response = requests.get(self.stations_url)
        response.raise_for_status()

        response_data = response.json()
        station_info = [i for i in response_data['features'] if i['properties']['SCODE'] == station_id]

        if len(station_info) == 0 :
            logger.warning(f"No metadata found for {station_id}")
            return {}
        
        station_props = station_info[0]['properties']
        return {
            'latitude': station_props.get('LAT'),
            'longitude': station_props.get('LONG'),
            'elevation': station_props.get('ALT'),
            'name': station_props.get('NAME_D')
        }

    def get_raw_data(
            self,            
            station_id: str,
            start: datetime.datetime,
            end: datetime.datetime,
            sleep_time: int = 1,
            split_on_year = True,
            sensor_codes: list[str] | None = None,
            **kwargs
        ):

        if station_id not in self.get_station_codes():
            raise ValueError(f"Invalid station_id {station_id}. Choose one from {self.get_station_codes()}")

        start = start.astimezone(pytz.timezone(self.timezone))
        end = end.astimezone(pytz.timezone(self.timezone))
        
        dates_split = split_dates(start, end, freq = self.freq, n_days = self.chunk_size_days, split_on_year=split_on_year)
        if sensor_codes is None:
            sensor_codes = self.get_sensors_for_station(station_id)
        else:
            if not isinstance(sensor_codes, list):
                raise ValueError(f"Sensor_codes must be of type list. Got {type(sensor_codes)}")
            for sensor in sensor_codes:
                if sensor not in self.get_sensors_for_station(station_id):
                    raise ValueError(f"Invalid sensor {sensor}. Choose from: {self.get_sensors_for_station(station_id)}")

        raw_responses = []
        for query_start, query_end in dates_split:
            for sensor in sensor_codes:
                try:
                    data_params = {
                        "station_code": station_id,
                        "sensor_code": sensor,
                        "date_from": query_start.strftime("%Y%m%d%H%M"),
                        "date_to": query_end.strftime("%Y%m%d%H%M")
                    }
                    response = requests.get(self.timeseries_url, params = data_params)
                    response.raise_for_status()

                    response_data = pd.DataFrame(response.json())

                    if len(response_data) == 0:
                        logger.warning(f"No data found for {data_params}")
                        continue

                    response_data['sensor'] = sensor
                    response_data['station_id'] = station_id

                    raw_responses.append(response_data)
                except Exception as e:
                    logger.error(f"Error fetching data for {sensor} for {query_start} - {query_end}: {e}", exc_info = True)

                time.sleep(sleep_time)

        if len(raw_responses) > 0:
            return pd.concat(raw_responses, ignore_index = True)
        else:
            logger.warning(f"No data could be fetched for station {station_id} and sensors {sensor_codes}")
            return None

    def transform(self, raw_data: pd.DataFrame | None):

        if raw_data is None:
            return None

        if raw_data[['DATE', 'station_id', 'sensor']].duplicated().any():
            logger.warning("Found duplicates for ['DATE', 'station_id', 'sensor']. They will be dropped")
            raw_data.drop_duplicates(subset = ['DATE', 'station_id', 'sensor'], inplace = True)
        
        df_pivot = raw_data.pivot(columns = "sensor", values = "VALUE", index = ["DATE", "station_id"]).reset_index()
        df_pivot.rename(columns = PROVINCE_RENAME, inplace = True)

        try:
            # 1. Capture whether it was Summer Time (CEST) before stripping
            is_dst = df_pivot['datetime'].str.contains('CEST')

            # 2. Strip the strings
            df_pivot['datetime'] =  df_pivot['datetime'].str.replace('CEST', '', regex=False).str.replace('CET', '', regex=False)

            # 3. Convert to naive datetime
            df_pivot['datetime'] = pd.to_datetime(df_pivot['datetime'], format="%Y-%m-%dT%H:%M:%S")

            # 4. Localize using the mask to resolve ambiguity
            # ambiguous=is_dst tells pandas: "If this hour repeats, use the DST version if is_dst is True"
            df_pivot['datetime'] = df_pivot['datetime'].dt.tz_localize(
                self.timezone, 
                ambiguous=is_dst,
                nonexistent='shift_forward' # handle the spring "gap" too
            ).dt.tz_convert('UTC')

            df_pivot['datetime'] = df_pivot['datetime'].dt.floor(self.freq)
        except Exception as e:
            logger.error(f"Error transforming datetime: {e}")

        # Precipitation is available in 5min freq while all others are in 10min freq. Drop additional timestamps for precipitation
        df_pivot = df_pivot.dropna(subset = [i for i in df_pivot.columns if i not in ['datetime', 'station_id', 'precipitation']], how = 'all')

        return df_pivot

if __name__ == '__main__':

    start = datetime.datetime(2025, 1, 14)
    end = datetime.datetime(2025, 1, 16)

    pr_handler = ProvinceAPI(timezone = 'Europe/Rome')
    print(pr_handler.get_station_info("86900MS"))
    data = pr_handler.get_data(
        station_id = '86900MS',
        sensor_codes = ["LT"],
        start = start,
        end = end
    )

    print(data)
