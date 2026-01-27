import pandas as pd
import asyncio
import httpx

import datetime
import pytz
from typing import Dict, Any, Tuple
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

    def __init__(
            self, 
            chunk_size_days: int = 365, 
            timeout: int = 20, 
            max_concurrent_requests: int = 5,
            sleep_time: int = 1,
            **kwargs
        ):
        self.timezone = "Europe/Rome"
        self.chunk_size_days = chunk_size_days
        self.timeout = timeout

        if max_concurrent_requests < 1:
            raise ValueError(f"max concurrent requests should be greater than 0. Got {max_concurrent_requests}")

        self.max_concurrent_requests = max_concurrent_requests
        self.sleep_time = sleep_time

        self.station_info = None
        self._station_info_lock = asyncio.Lock()

        self.station_sensors = {}
        self._station_sensors_locks: dict[str, asyncio.Lock] = {}

        self._client = None

    async def __aenter__(self):
        logger.info("Opening ProvinceAPI session...")
        self._client = httpx.AsyncClient()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        logger.info("Closing ProvinceAPI session...")
        if self._client is not None:
            await self._client.aclose()

    def __enter__(self):
        raise RuntimeError("Use 'async with ProvinceAPI()' for async context management.")

    def __exit__(self, exc_type, exc_value, traceback):
        raise RuntimeError("Use 'async with ProvinceAPI()' for async context management.")

    @classmethod
    def name(cls):
        return "province_api"

    @property
    def freq(self):
        return "10min"

    @property
    def inclusive(self):
        return "both"

    async def get_sensors_for_station(self, station_code: str):
        if self.station_sensors.get(station_code) is not None:
            return self.station_sensors.get(station_code)

        lock = self._station_sensors_locks.get(station_code)
        if lock is None:
            lock = asyncio.Lock()
            self._station_sensors_locks[station_code] = lock

        async with lock:
            if self.station_sensors.get(station_code) is not None:
                return self.station_sensors.get(station_code)

            if self._client is None:
                raise ValueError("Initialize client before querying sensors")

            response = await self._client.get(
                    self.sensors_url, params = {"station_code": station_code},
                    timeout=self.timeout
                )
            response.raise_for_status()

            sensors_list = set([i['TYPE'] for i in response.json()])
            self.station_sensors[station_code] = sensors_list

            return sensors_list

    async def get_station_codes(self):
        if self.station_info is not None:
            return list(self.station_info.keys())
        else:
            info = await self.get_station_info()
            return list(info.keys())
        
    async def get_station_info(self, station_id: str | None = None) -> Dict[str, Any]:
        if self.station_info is not None:
            if station_id is not None:
                return self.station_info.get(station_id, {})
            return self.station_info

        async with self._station_info_lock:

            if self.station_info is not None:
                return self.station_info

            if self._client is None:
                raise ValueError("Initialize client before querying station info")

            response = await self._client.get(
                    self.stations_url, 
                    timeout=self.timeout
                )
            response.raise_for_status()

            response_data = response.json()

            if len(response_data['features']) == 0:
                raise ValueError("Error retrieving station info. Response data contains no features")

            info_dict = {}
            for i in response_data['features']:
                station_props = i['properties']
                station_info = {
                    'latitude': station_props.get('LAT'),
                    'longitude': station_props.get('LONG'),
                    'elevation': station_props.get('ALT'),
                    'name': station_props.get('NAME_D')
                }
                info_dict[station_props['SCODE']] = station_info

            self.station_info = info_dict

            if station_id is not None:
                return self.station_info.get(station_id, {})
            return self.station_info
        
    async def _create_request_task(
        self, station_id: str, date_range: Tuple[datetime, datetime], sensor: str
        ):

        if self._client is None:
            raise ValueError("Initialize client before requesting data")
        
        try:
            query_start, query_end = date_range
            data_params = {
                "station_code": station_id,
                "sensor_code": sensor,
                "date_from": query_start.strftime("%Y%m%d%H%M"),
                "date_to": query_end.strftime("%Y%m%d%H%M")
            }
            response = await self._client.get(
                    self.timeseries_url, params = data_params,
                    timeout=self.timeout
                )
            response.raise_for_status()

            response_data = pd.DataFrame(response.json())

            if len(response_data) == 0:
                logger.warning(f"No data found for {data_params}")
                return None

            response_data['sensor'] = sensor
            response_data['station_id'] = station_id

            return response_data
        except Exception as e:
            logger.error(f"Error fetching data for {sensor} for {query_start} - {query_end}: {e}", exc_info = True)
            return None
        finally:
            await asyncio.sleep(self.sleep_time)

    async def _worker(self, queue: asyncio.Queue, results: list[pd.DataFrame | None]):
        while True:
            job = await queue.get()

            if job is None:
                queue.task_done()
                break

            station_id, date_range, sensor = job
            try:
                raw_data = await self._create_request_task(station_id, date_range, sensor)
                if raw_data is not None:
                    results.append(raw_data)
            except Exception as e:
                logger.error(f"Worker error for station {station_id} with sensor {sensor}: {e}", exc_info=True)
            finally:
                queue.task_done()

    async def get_raw_data(
            self,            
            station_id: str,
            start: datetime.datetime,
            end: datetime.datetime,
            split_on_year = True,
            sensor_codes: list[str] | None = None,
            **kwargs
        ):

        if station_id not in await self.get_station_codes():
            raise ValueError(f"Invalid station_id {station_id}. Choose one from {await self.get_station_codes()}")

        queue = asyncio.Queue(maxsize = self.max_concurrent_requests *2)
        start = start.astimezone(pytz.timezone(self.timezone))
        end = end.astimezone(pytz.timezone(self.timezone))
        
        dates_split = split_dates(start, end, freq = self.freq, n_days = self.chunk_size_days, split_on_year=split_on_year)
        
        all_sensors = await self.get_sensors_for_station(station_id)
        if sensor_codes is None:
            sensor_codes = all_sensors
        else:
            if not isinstance(sensor_codes, list):
                raise ValueError(f"Sensor_codes must be of type list. Got {type(sensor_codes)}")
            for sensor in sensor_codes:
                if sensor not in all_sensors:
                    raise ValueError(f"Invalid sensor {sensor}. Choose from: {all_sensors}")

        # Start the workers
        raw_responses = []
        workers = [
            asyncio.create_task(self._worker(queue, raw_responses))
            for _ in range(self.max_concurrent_requests)
            ]

        # Put jobs in the queue
        for date_range in dates_split:
            for sensor in sensor_codes:
                await queue.put((station_id, date_range, sensor))

        # Put the stop signal into the queue
        for _ in workers:
            await queue.put(None)

        # Wait for all jobs to finish
        await queue.join()

        # Make sure all workers finish
        await asyncio.gather(*workers)

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
    
    import logging
    from ..validate.meteo import MeteoValidator

    logging.basicConfig(level = logging.DEBUG, force = True)

    async def run_test():
        start = datetime.datetime(2026, 1, 14)
        end = datetime.datetime(2026, 10, 21)

        pr_handler = ProvinceAPI()
        async with pr_handler as meteo_handler:

            st_info = await meteo_handler.get_station_info("86900MS")
            print(st_info)

            data = await meteo_handler.get_data(
                station_id = '86900MS',
                sensor_codes = ["LT", 'N', 'LF'],
                start = start,
                end = end,
                validator = MeteoValidator()
            )
        print(data)
        return data

    data = asyncio.run(run_test())
