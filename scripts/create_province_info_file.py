import asyncio
import pandas as pd

from src.meteo import BaseMeteoHandler

def main():
    meteo_loader = BaseMeteoHandler.create("province_api")

    async def load():
        async with meteo_loader as loader: 
            st_info = await loader.get_station_info()
        return st_info

    st_info = asyncio.run(load())
    return pd.DataFrame.from_dict(st_info, orient = 'index')

if __name__ == '__main__':

    st_info = main()
    st_info.rename(columns = {'id': 'station_id'}).to_csv('data/province/stations.csv', index = False)
