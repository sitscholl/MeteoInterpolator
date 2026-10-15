import logging
import math
from contextlib import contextmanager
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import create_engine, event
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import sessionmaker

from . import models
from ..utils import validate_station_coordinates

logger = logging.getLogger(__name__)

_INSERT_STATEMENT = sqlite_insert(models.Measurement)
_UPSERT_STATEMENT = _INSERT_STATEMENT.on_conflict_do_update(
    index_elements=["station_id", "variable_id", "datetime"],
    set_={
        "value": _INSERT_STATEMENT.excluded.value,
        "retrieved_at": _INSERT_STATEMENT.excluded.retrieved_at,
    },
)
_BATCH_SIZE = 5_000

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

    def get_stations(self, id: str | None = None) -> list[models.Station]:
        with self.session_scope() as session:
            query = session.query(models.Station)
            if id is not None:
                query = query.filter(models.Station.id == id)
            return query.all()

    def get_variables(self, name: str | None = None):
        with self.session_scope() as session:
            query = session.query(models.Variable)
            if name is not None:
                query = query.filter(models.Variable.name == name)
            return query.all()

    def get_data(
            self,
            station_id: str,
            start_time: datetime,
            end_time: datetime | None = None,
            variables: list[str] | None = None,
        ):

        with self.session_scope() as session:
            query = (
                session.query(
                    models.Measurement.datetime.label("datetime"),
                    models.Measurement.value.label("value"),
                    models.Station.id.label("station_id"),
                    models.Variable.name.label("variable"),
                )
                .join(models.Station, models.Measurement.station_id == models.Station.id)
                .join(models.Variable, models.Measurement.variable_id == models.Variable.id)
                .filter(
                    models.Station.id == station_id,
                )
            )

            start_time = pd.Timestamp(start_time)
            orig_timezone = start_time.tzinfo
            if orig_timezone is None:
                raise ValueError("Start time has not timezone. Use timezone-aware datetimes to get data from the database")
            start_time_utc = start_time.astimezone(UTC)
            if end_time is not None:
                end_time = pd.Timestamp(end_time)
                end_timezone = end_time.tzinfo
                if orig_timezone != end_timezone:
                    raise ValueError(f"Start and end timezone must be the same. Got {orig_timezone} vs {end_timezone}")
                 
                end_time_utc = end_time.astimezone(UTC)
                query = query.filter(
                    models.Measurement.datetime >= start_time_utc,
                    models.Measurement.datetime < end_time_utc,
                )
            else:
                query = query.filter(models.Measurement.datetime >= start_time_utc)

            if variables is not None:
                available_variables = self.get_variables()
                variables_ids = []
                for v in variables:
                    variable_model = [i for i in available_variables if i.name == v]
                    if not variable_model:
                        logger.warning(f"Variable {v} not found in database")
                        continue
                    variables_ids.append(variable_model[0].id)

                query = query.filter(
                    models.Measurement.variable_id.in_(variables_ids)
                )

            df = pd.read_sql_query(sql=query.statement, con=self.engine)

        if not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(orig_timezone)
            df = df.pivot(columns = 'variable', values = 'value', index = ['station_id', 'datetime'])
            df.reset_index(inplace = True)

        return df

    def get_last_row_per_sensor(self, station_id: str, variables: list[str]) -> dict | None:
        if not self.station_exists(station_id):
            return None
        
        if isinstance(variables, str):
            variables = [variables]

        var_results = {}
        with self.session_scope() as session:
            for var in variables:
                query = (
                    session.query(models.Measurement)
                    .join(models.Station, models.Measurement.station_id == models.Station.id)
                    .join(models.Variable, models.Measurement.variable_id == models.Variable.id)
                    .filter(models.Station.id == station_id)
                    .filter(models.Variable.name == var)
                    .order_by(models.Measurement.datetime.desc())
                    )
                var_results[var] = query.first()
        return var_results

    def station_exists(self, id: str):
        existing_station = self.get_stations(id = id)
        return len(existing_station) > 0

    def add_station(
        self, 
        id: str, 
        latitude: float, 
        longitude: float, 
        elevation: float, 
        name: str | None = None, 
        ):
        if self.station_exists(id):
            raise ValueError(f"Station with id {id} already exists in the database")

        latitude, longitude, elevation = validate_station_coordinates(
            latitude,
            longitude,
            elevation,
        )

        session = self.Session()
        try:
            new_station = models.Station(id = id, latitude = latitude, longitude = longitude, elevation = elevation, name = name)
            session.add(new_station)
            session.commit()
            logger.info("New station with id %s for inserted successfully.", id)
            session.refresh(new_station)
            return new_station
        except Exception as e:
            session.rollback()
            logger.exception("Error inserting new station with id %s", id)
            raise
        finally:
            session.close()

    def update_station(
        self,
        id: str,
        only_missing: bool = True,
        **kwargs,
    ):

        with self.session_scope() as session:
            stations = session.query(models.Station).filter(models.Station.id == id).all()

            if len(stations) == 0:
                raise ValueError(f"No station with id {id} for exists.")
            station = stations[0]

            allowed_fields = {"name", "latitude", "longitude", "elevation"}
            invalid_fields = {k: v for k, v in kwargs.items() if k not in allowed_fields}
            if len(invalid_fields) > 0:
                raise ValueError(f"Invalid fields in update_station: {invalid_fields.keys()}. Allowed fields: {allowed_fields}")

            try:
                changed = False
                for key, value in kwargs.items():
                    if value is None:
                        continue
                    if only_missing and getattr(station, key) is not None:
                        continue
                    setattr(station, key, value)
                    changed = True

                if changed:
                    _ = validate_station_coordinates(station.latitude, station.longitude, station.elevation)
                    session.commit()
                    logger.info("Updated station metadata for station %s.", id)
                    session.refresh(station)
                return station
            except Exception as e:
                session.rollback()
                logger.exception("Error updating station metadata for station %s", id)
                raise

    def variable_exists(self, name: str):
        existing_variable = self.get_variables(name = name)
        return len(existing_variable) > 0

    def add_variable(self, name: str, unit: str | None = None, description: str | None = None):

        if self.variable_exists(name):
            raise ValueError(f"Variable {name} already exists in the database")

        session = self.Session()
        try:
            new_variable = models.Variable(name = name, unit = unit, description = description)
            session.add(new_variable)
            session.commit()
            logger.info(f"New variable {new_variable.name} inserted successfully.")
            session.refresh(new_variable)
            return new_variable
        except Exception as e: 
            session.rollback()
            logger.error(f"Error inserting new variable: {e}")
            raise
        finally:
            session.close()

    def add_station_data(
        self, 
        id: str,
        data: pd.DataFrame, 
        ignore_missing_variables: bool = False,
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

        variables_by_name = {
            variable.name: variable.id
            for variable in self.get_variables()
        }

        missing_variables = [name for name in variable_columns if name not in variables_by_name]
        if len(missing_variables) > 0 and ignore_missing_variables:
            logger.warning("Ignoring missing variables when adding station data: %s", missing_variables)
        elif len(missing_variables) > 0:
            raise ValueError(f"Cannot add station data to database because the following variables are missing: {missing_variables}. Avaiable variabels: {variables_by_name.keys()}")

        active_variables = [i for i in variable_columns if i not in missing_variables]
        if len(active_variables) == 0:
            raise ValueError("No variables available to insert into the database")
        
        # Create table for database insertion
        measurements = data[['datetime'] + active_variables].copy()
        measurements['retrieved_at'] = pd.Timestamp.now(tz = "UTC")
        measurements = measurements.melt(
            id_vars=['datetime', 'retrieved_at'],
            value_vars=active_variables,
            var_name='variable',
            value_name='value'
        )
        measurements.dropna(subset=['value'], inplace = True)

        measurements['station_id'] = id
        measurements['variable_id'] = measurements['variable'].map(variables_by_name)
        measurements.drop(columns = ['variable'], inplace = True)

        # Ensure column order matches the measurements table definition
        measurements = measurements[['station_id', 'variable_id', 'datetime', 'value', 'retrieved_at']]
        measurements = measurements.drop_duplicates(
            subset=["station_id", "variable_id", "datetime"],
            keep="last",
        )

        written_rows = 0
        try:
            with self.Session.begin() as session:
                for start in range(0, len(measurements), _BATCH_SIZE):
                    batch = measurements.iloc[start : start + _BATCH_SIZE]
                    session.execute(
                        _UPSERT_STATEMENT,
                        batch.to_dict(orient="records"),
                    )
                    written_rows += len(batch)
            logger.debug("Inserted %s measurements for station %s and %s variables.", len(measurements), id, len(active_variables))
        except Exception as e:
            logger.exception("Error inserting bulk measurement data for station %s", id)
            raise

        return written_rows
                    
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
