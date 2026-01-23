import pandas as pd
import pandera.pandas as pa

class MeteoValidator:

    def __init__(self, allow_additional_columns: bool = True):
        self.allow_additional_columns = allow_additional_columns

    @property
    def output_schema(self) -> pa.DataFrameSchema:
        """
        Define the expected schema for SBR meteorological data output.
        """
        return pa.DataFrameSchema(
            {
                "datetime": pa.Column(pd.DatetimeTZDtype(tz="UTC"), coerce=True),
                "station_id": pa.Column(str),

                "tair_2m": pa.Column(float, nullable=True, required = False),
                "precipitation": pa.Column(float, nullable=True, required = False),
                "solar_radiation": pa.Column(float, nullable = True, required = False),
            },
            index = pa.Index(int),
            strict= not self.allow_additional_columns
        )

    def validate(self, transformed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate the transformed data against the output schema.
        
        Args:
            transformed_data (pd.DataFrame): Data to validate
            
        Returns:
            pd.DataFrame: Validated data
            
        Raises:
            pa.errors.SchemaError: If validation fails
        """
        return self.output_schema.validate(transformed_data)