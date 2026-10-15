from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Station(Base):
    __tablename__ = "stations"

    id = Column(String, nullable=False, unique = True, primary_key=True)
    name = Column(String)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    elevation = Column(Float, nullable=False)

    measurements = relationship("Measurement", back_populates="station")


class Variable(Base):
    __tablename__ = "variables"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)   # e.g. "temperature_2m"
    unit = Column(String, nullable=True)
    description = Column(Text)

    measurements = relationship("Measurement", back_populates="variable")


class Measurement(Base):
    __tablename__ = "measurements"

    id = Column(Integer, primary_key=True)
    station_id = Column(String, ForeignKey("stations.id", onupdate="CASCADE", ondelete='CASCADE'), nullable=False)
    variable_id = Column(Integer, ForeignKey("variables.id", onupdate="CASCADE", ondelete='CASCADE'), nullable=False)
    datetime = Column(DateTime(timezone=True), nullable=False)
    value = Column(Float)
    retrieved_at = Column(DateTime(timezone=True), nullable=False)

    station = relationship("Station", back_populates="measurements")
    variable = relationship("Variable", back_populates="measurements")

    __table_args__ = (
        UniqueConstraint("station_id", "variable_id", "datetime"),
        Index(
            "ix_measurements_variable_datetime_station",
            "variable_id",
            "datetime",
            "station_id",
        ),
    )
