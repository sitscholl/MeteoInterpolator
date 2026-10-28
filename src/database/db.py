import logging
from contextlib import contextmanager, nullcontext
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, event, func
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session, sessionmaker

from ..utils import to_utc_timestamp, validate_crs, validate_station_coordinates
from . import models
from .models import DATABASE_VARIABLES

logger = logging.getLogger(__name__)

_INSERT_STATEMENT = sqlite_insert(models.Measurement)
_UPSERT_STATEMENT = _INSERT_STATEMENT.on_conflict_do_update(
    index_elements=["station_id", "datetime"],
    set_= {
    **{
        name: func.coalesce(
            _INSERT_STATEMENT.excluded[name],
            models.Measurement.__table__.c[name],
        )
        for name in DATABASE_VARIABLES
    },
    "retrieved_at": _INSERT_STATEMENT.excluded.retrieved_at,
    }
)
_BATCH_SIZE = 5_000

class StationAlreadyExistsError(ValueError):
    pass

class InterpolationDB:

    def __init__(self, engine: str = 'sqlite:///db/database.db'):
        self.engine = create_engine(engine)
        if self.engine.dialect.name == "sqlite":
            event.listen(self.engine, "connect", self._enable_sqlite_foreign_keys)
        models.Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )

    @staticmethod
    def _enable_sqlite_foreign_keys(dbapi_connection, connection_record) -> None:
        """Enable SQLite foreign-key enforcement for every pooled connection."""
        dbapi_connection.execute("PRAGMA foreign_keys=ON")
    
    @contextmanager
    def session_scope(self):
        session = self.Session()
        try:
            yield session
        finally:
            session.close()

    def get_stations(self, id: str | None = None, include_inactive: bool = False) -> list[models.Station]:
        with self.session_scope() as session:
            query = session.query(models.Station)
            if id is not None:
                query = query.filter(models.Station.id == id)
            if not include_inactive:
                query = query.filter(models.Station.status == 'active')
            return query.all()

    def get_data(
            self,
            station_ids: str | list[str],
            start_time: datetime,
            end_time: datetime | None = None,
        ):

        if isinstance(station_ids, str):
            station_ids = [station_ids]

        with self.session_scope() as session:
            query = (
                session.query(models.Measurement)
                .join(models.Station, models.Measurement.station_id == models.Station.id)
                .filter(
                    models.Station.id.in_(station_ids),
                )
            )

            start_time_utc = to_utc_timestamp(start_time, raise_missing_timezone=True)
            if end_time is not None:
                end_time_utc = to_utc_timestamp(end_time, raise_missing_timezone=True)

                query = query.filter(
                    models.Measurement.datetime >= start_time_utc,
                    models.Measurement.datetime < end_time_utc,
                )
            else:
                query = query.filter(models.Measurement.datetime >= start_time_utc)

            df = pd.read_sql_query(sql=query.statement, con=self.engine)

        if not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(start_time.tzinfo)

        return df

    def get_last_row(self, station_id: str) -> models.Measurement | None:
        if not self.station_exists(station_id):
            return None
        with self.session_scope() as session:
            query = (
                session.query(models.Measurement)
                .join(models.Station, models.Measurement.station_id == models.Station.id)
                .filter(models.Station.id == station_id)
                .order_by(models.Measurement.datetime.desc())
                )
            return query.first()

    def station_exists(self, id: str):
        existing_station = self.get_stations(id = id, include_inactive=True)
        return len(existing_station) > 0

    def add_station(
        self, 
        id: str, 
        latitude: float, 
        longitude: float, 
        elevation: float, 
        crs: int,
        status: str,
        name: str | None = None, 
        db_session: Session | None = None,
        ):
        if self.station_exists(id):
            raise StationAlreadyExistsError(
                f"Station with id {id} already exists"
            )

        if not isinstance(status, str):
            raise TypeError(f"Status must be of type string. Got {type(status)}")
        status = status.lower()
        if status not in ['active', 'inactive']:
            raise ValueError(f"Invalid status {status}. Choose one of 'active' or 'inactive'")
        latitude, longitude, elevation = validate_station_coordinates(
            latitude,
            longitude,
            elevation,
        )
        crs = validate_crs(crs)

        transaction = (
            self.Session.begin()
            if db_session is None
            else nullcontext(db_session)
        )

        with transaction as session:
            new_station = models.Station(
                id = id, latitude = latitude, longitude = longitude, elevation = elevation, crs = crs, name = name, status = status
                )
            session.add(new_station)
            session.flush()
            logger.info("New station with id %s inserted successfully.", id)
            return new_station

    def update_station(
        self,
        id: str,
        only_missing: bool = True,
        db_session: Session | None = None,
        **kwargs,
    ):

        transaction = (
            self.Session.begin()
            if db_session is None
            else nullcontext(db_session)
        )

        with transaction as session:
            stations = session.query(models.Station).filter(models.Station.id == id).all()

            if len(stations) == 0:
                raise ValueError(f"No station with id {id} for exists.")
            station = stations[0]

            allowed_fields = {"name", "latitude", "longitude", "elevation", "crs", "status"}
            invalid_fields = {k: v for k, v in kwargs.items() if k not in allowed_fields}
            if len(invalid_fields) > 0:
                raise ValueError(f"Invalid fields in update_station: {invalid_fields.keys()}. Allowed fields: {allowed_fields}")

            changed = False
            for key, value in kwargs.items():
                if value is None:
                    continue
                if only_missing and getattr(station, key) is not None:
                    continue
                if key == 'crs':
                    value = validate_crs(value)
                setattr(station, key, value)
                changed = True

            if changed:
                _ = validate_station_coordinates(station.latitude, station.longitude, station.elevation)
                _ = validate_crs(station.crs)
                session.add(station)
                session.flush()
                logger.info("Updated station metadata for station %s.", id)
            return station

    def sync_catalog(self, station_catalog: pd.DataFrame):
        catalog_ids = station_catalog.station_id.tolist()
        inactive_ids = [i.id for i in self.get_stations() if i.id not in catalog_ids]

        with self.Session.begin() as db_session:
            for row in station_catalog.to_dict(orient="records"):
                st_id = row.pop('station_id')
                if self.station_exists(st_id):
                    self.update_station(
                        id = st_id,
                        **row,
                        only_missing = False,
                        db_session=db_session
                    )
                else:
                    self.add_station(
                        id = st_id,
                        **row,
                        db_session=db_session
                    )
            for st_id in inactive_ids:
                self.update_station(id = st_id, status = 'inactive', db_session=db_session, only_missing=False)

    def variable_exists(self, name: str):
        return name in DATABASE_VARIABLES

    def add_station_data(
        self, 
        id: str,
        data: pd.DataFrame, 
        drop_invalid_variables: bool = False,
        db_session: Session | None = None
        ):

        if data.empty:
            raise ValueError("Empty dataframe provided to add_station_data")
        if not self.station_exists(id):
            raise ValueError(f"Cannot add data for station {id} because station does not exist in the database. Add station first.")
        if 'datetime' not in data.columns:
            raise ValueError("Missing column 'datetime' in data.")

        #Transform datetime
        data = data.copy()
        data["datetime"] = pd.to_datetime(data["datetime"], utc=True, errors='raise')
        
        # Validate and drop station_id column if present
        if 'station_id' in data.columns:
            station_ids = data["station_id"].dropna().astype(str).unique()
            if len(station_ids) != 1:
                raise ValueError(
                    "Station data must contain exactly one non-null station_id. "
                    f"Got {station_ids.tolist()}"
                )
            data_id = station_ids[0]
            if data_id != id:
                raise ValueError(
                    "Station id in data does not match provided id. "
                    f"Got {data_id} vs {id}"
                )
            data.drop(columns=["station_id"], inplace=True)

        #Validate variables
        variable_columns = [col for col in data.columns if col != 'datetime']
        if len(variable_columns) == 0:
            raise ValueError('No variable columns found in data')

        invalid_variables = [i for i in variable_columns if i not in DATABASE_VARIABLES]
        if invalid_variables and not drop_invalid_variables:
            raise ValueError(f"Cannot add station data to database due to the following invalid variables: {invalid_variables}")
        elif invalid_variables:
            logger.debug("Dropping invalid variables before adding data: %s", invalid_variables)
            data.drop(columns = invalid_variables, inplace = True)
        
        active_variables = [i for i in DATABASE_VARIABLES if i in variable_columns]
        if len(active_variables) == 0:
            raise ValueError("No valid variables available to insert into the database")
        
        # Create table for database insertion
        measurements = data[['datetime'] + active_variables].copy()
        measurements['retrieved_at'] = pd.Timestamp.now(tz = "UTC")
        measurements.dropna(subset=active_variables, how = 'all', inplace = True)

        measurements['station_id'] = id

        # Ensure column order matches the measurements table definition
        measurements = measurements[['station_id', 'datetime'] + active_variables + ['retrieved_at']]
        measurements = measurements.drop_duplicates(
            subset=["station_id", "datetime"],
            keep="last",
        )

        transaction = (
            self.Session.begin()
            if db_session is None
            else nullcontext(db_session)
        )

        written_rows = 0
        with transaction as session:
            for start in range(0, len(measurements), _BATCH_SIZE):
                batch = measurements.iloc[start : start + _BATCH_SIZE]
                session.execute(
                    _UPSERT_STATEMENT,
                    batch.to_dict(orient="records"),
                )
                written_rows += len(batch)

        return written_rows

    def log_ingestion_task(
        self, station_id: str, requested_start: datetime, requested_end: datetime, status: str, db_session: Session | None = None
    ):

        if status not in {'completed', 'skipped', 'failed'}:
            raise ValueError(f"Invalid status {status}")

        requested_start = to_utc_timestamp(requested_start)
        requested_end = to_utc_timestamp(requested_end)
        if requested_start >= requested_end:
            raise ValueError("requested_start must precede requested_end")

        transaction = (
            self.Session.begin() 
            if db_session is None
            else nullcontext(db_session)
        )

        with transaction as session:
            if session.get(models.Station, station_id) is None:
                raise ValueError(f"No station with id {station_id} in database")

            entry = models.IngestionRequest(
                station_id=station_id,
                requested_start=requested_start,
                requested_end=requested_end,
                status=status,
            )
            session.add(entry)
            session.flush()

        return entry
   
    def get_last_ingestion_task(self, station_id: str, status: str = 'completed'):
        with self.session_scope() as session:
            query = (
                session.query(models.IngestionRequest)                
                .join(models.Station, models.IngestionRequest.station_id == models.Station.id)
                .filter(models.Station.id == station_id)
                .filter(models.IngestionRequest.status == status)
                .order_by(models.IngestionRequest.requested_end.desc())
                )
            return query.first()

    def close(self):
        """
        Close the SQLAlchemy session and dispose of the engine connection pool.
        """
        try:
            if self.engine:
                self.engine.dispose()
                logger.debug("Database engine disposed.")
        except Exception as e:
            logger.warning(f"Error disposing database engine: {e}")
