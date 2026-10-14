import logging
from contextlib import contextmanager
from datetime import datetime, timezone, UTC

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..meteo.base import BaseMeteoHandler
from . import models

logger = logging.getLogger(__name__)

class InterpDB:

    def __init__(self, engine: str = 'sqlite:///db/database.db'):
        self.engine = create_engine(engine)
        models.Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    
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
                query = query.filter(models.Measurement.datetime.between(start_time_utc, end_time_utc))
            else:
                query = query.filter(models.Measurement.datetime >= start_time_utc)

            if variables is not None:
                variables_ids = []
                for v in variables:
                    variable_model = self.get_variables(v)
                    if not variable_model:
                        logger.warning(f"Variable {v} not found in database")
                        continue
                    variables_ids.append(variable_model[0].id)

                query = query.filter(
                    models.Measurement.variable_id.in_(variables_ids)
                )

            df = pd.read_sql_query(sql=query.statement, con=self.engine)

        if not df.empty:
            try:
                df['datetime'] = df['datetime'].dt.tz_localize(UTC).dt.tz_convert(orig_timezone)
            except Exception as e:
                logger.warning(f"Could not convert timezone back to {orig_timezone}: {e}. Keeping UTC timezone.")
                # Ensure index is UTC-aware if conversion fails
                if df['datetime'].tz is None:
                    df['datetime'] = df['datetime'].tz_localize('UTC')

            df = df.pivot(columns = 'variable', values = 'value', index = ['station_id', 'datetime'])
            df.reset_index(inplace = True)

        return df

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
        """
        Add a new station to the database or update existing one
        """
        if self.station_exists(id):
            raise ValueError(f"Station with id {id} already exists in the database")

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
            logger.error(f"Error inserting new station: {e}")
            return
        finally:
            session.close()

    def update_station(
        self,
        id: str,
        only_missing: bool = True,
        **kwargs,
    ):

        stations = self.get_stations(id)

        if len(stations) == 0:
            raise ValueError(f"No station with id {id} for exists.")
        station = stations[0]

        allowed_fields = {"name", "latitude", "longitude", "elevation"}
        invalid_fields = {k: v for k, v in kwargs if k not in allowed_fields}
        if len(invalid_fields) > 0:
            raise ValueError(f"Invalid fields in update_station: {invalid_fields.keys()}. Allowed fields: {allowed_fields}")

        session = self.Session()
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
                session.commit()
                logger.info("Updated station metadata for station %s.", id)
                session.refresh(station)
            return station
        except Exception as e:
            session.rollback()
            logger.exception("Error updating station metadata for station %s", id)
            return None
        finally:
            session.close()

    def variable_exists(self, name: str):
        existing_variable = self.get_variables(name = name)
        return len(existing_variable) > 0

    def add_variable(self, name: str, unit: str, description: str | None = None):

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
            return
        finally:
            session.close()

    def add_station_data(
        self, 
        id: str,
        data: pd.DataFrame, 
        index=False, 
        index_label=None, 
        if_exists='append',
        ignore_missing_variables: bool = False,
        ):
        """
        Insert measurement data into the database. All columns other than 'datetime' are assumed to contain variables and will be inserted into the Measurement table.
        Stations and variables which do not exist in the database will be created automatically in the respective tables.
        """
        if data.empty:
            raise ValueError("Empty dataframe provided to add_station_data")

        if not self.station_exists(id):
            raise ValueError(f"Cannot add data for station {id} because station does not exist in the database. Add station first.")

        if 'datetime' not in data.columns:
            raise ValueError("Missing column 'datetime' in data.")

        variable_columns = [col for col in data.columns if col != 'datetime']
        if len(variable_columns) == 0:
            raise ValueError('No variable columns found in data')

        missing_variables = [i for i in variable_columns if not self.variable_exists(i)]
        if len(missing_variables) > 0 and ignore_missing_variables:
            logger.warning("Ignoring missing variables when adding station data: %s", missing_variables)
        elif len(missing_variables) > 0:
            available_variables = [i.name for i in self.get_variables()]
            raise ValueError(f"Cannot add station data to database because the following variables are missing: {missing_variables}. Avaiable variabels: {available_variables}")

        # Make a copy to avoid modifying the original DataFrame
        active_variables = [i for i in variable_columns if i not in missing_variables]
        if len(active_variables) == 0:
            raise ValueError("No variables available to insert into the database")
        measurements = data[['datetime'] + active_variables].copy()

        if measurements['datetime'].dt.tz is None:
            measurements['datetime'] = measurements['datetime'].dt.tz_localize('UTC')
        else:
            measurements['datetime'] = measurements['datetime'].dt.tz_convert('UTC')

        # Convert wide table (one column per variable) into long form for bulk insert
        measurements = measurements.melt(
            id_vars=['datetime'],
            value_vars=active_variables,
            var_name='variable',
            value_name='value'
        )
        # measurements.dropna(subset=['value'], inplace = True)

        variable_id_map = {i: self.get_variables(i)[0].id for i in active_variables}

        measurements['station_id'] = id
        measurements['variable_id'] = measurements['variable'].map(variable_id_map)
        measurements.drop(columns = ['variable'], inplace = True)

        # Ensure column order matches the measurements table definition
        measurements = measurements[['station_id', 'variable_id', 'datetime', 'value']]

        try:
            measurements.to_sql(
                name='measurements',
                con=self.engine,
                index=index,
                index_label=index_label,
                if_exists=if_exists
            )
            logger.debug("Inserted %s measurement rows for station %s and %s variables.", len(measurements), id, len(active_variables))
        except Exception as e:
            logger.error(f"Error inserting bulk measurement data: {e}")
                    
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
