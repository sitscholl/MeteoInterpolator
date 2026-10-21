from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

##Make sure order is the same as in the table definition!
DATABASE_VARIABLES = [
    "tair_2m", "tair_2m_min", "tair_2m_max", "solar_radiation", "relative_humidity", "precipitation", "sun_duration", "wind_gust", 
    "wind_direction", "snow_height", "wind_speed", "air_pressure", "discharge", "water_level",
    ]

class Station(Base):
    __tablename__ = "stations"

    id = Column(String, nullable=False, unique = True, primary_key=True)
    name = Column(String)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    elevation = Column(Float, nullable=False)
    crs = Column(Integer, nullable = False)

    measurements = relationship("Measurement", back_populates="station")
    ingestion_requests = relationship("IngestionRequest", back_populates="station")


class Measurement(Base):
    __tablename__ = "measurements"

    id = Column(Integer, primary_key=True)
    station_id = Column(String, ForeignKey("stations.id", onupdate="CASCADE", ondelete='CASCADE'), nullable=False)
    datetime = Column(DateTime(timezone=True), nullable=False)
    tair_2m = Column(Float, nullable=True)
    tair_2m_min = Column(Float, nullable=True)
    tair_2m_max = Column(Float, nullable=True)
    solar_radiation = Column(Float, nullable=True)
    relative_humidity = Column(Float, nullable=True)
    precipitation = Column(Float, nullable=True)
    sun_duration = Column(Float, nullable=True)
    wind_gust = Column(Float, nullable=True)
    wind_direction = Column(Float, nullable=True)
    snow_height = Column(Float, nullable=True)
    wind_speed = Column(Float, nullable=True)
    air_pressure = Column(Float, nullable=True)
    discharge = Column(Float, nullable=True)
    water_level = Column(Float, nullable=True)
    retrieved_at = Column(DateTime(timezone=True), nullable=False)

    station = relationship("Station", back_populates="measurements")

    __table_args__ = (
        UniqueConstraint("station_id", "datetime"),
    )

class IngestionRequest(Base):
    __tablename__ = 'ingestion_requests'

    id = Column(Integer, primary_key=True)
    requested_start = Column(DateTime(timezone=True), nullable = False)
    requested_end = Column(DateTime(timezone=True), nullable = False)
    status = Column(String, nullable = False)
    station_id = Column(String, ForeignKey("stations.id", onupdate="CASCADE", ondelete='CASCADE'), nullable=False)

    station = relationship("Station", back_populates="ingestion_requests")

    __table_args__ = (
        Index(
            "ix_ingestion_station_status_end",
            "station_id", "status", "requested_end",
        ),
    )