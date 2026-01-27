from dataclasses import dataclass
import logging
from typing import Optional

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class Station:
    id: str
    x: float
    y: float
    elevation: Optional[float]
    data: pd.DataFrame

    def __post_init__(self):
        if -90 > self.y or self.y > 90:
            raise ValueError("Latitude must be between -90 and 90")
        if -180 > self.x or self.x > 180:
            raise ValueError("Longitude must be between -180 and 180")

    @classmethod
    async def create(cls, id, x, y, data, elevation: Optional[float] = None, client: Optional[httpx.AsyncClient] = None):
        if elevation is None:
            try:
                elevation = await cls.fetch_elevation(x, y, client=client)
            except Exception as e:
                logger.warning(f"Fetching elevation for station {id} failed with error: {e}")
        return cls(id = id, x = x, y = y, elevation = elevation, data = data)

    @staticmethod
    async def fetch_elevation(x: float, y: float, client: Optional[httpx.AsyncClient] = None) -> float:
        api_template = "https://api.opentopodata.org/v1/eudem25m?locations={lat},{lon}"
        url = api_template.format(lat=y, lon=x)

        if client is None:
            async with httpx.AsyncClient() as temp_client:
                response = await temp_client.get(url)
                response.raise_for_status()
                return response.json()["results"][0]["elevation"]

        response = await client.get(url)
        response.raise_for_status()
        return response.json()["results"][0]["elevation"]
