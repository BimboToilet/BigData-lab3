import polars as pl
from deltalake import DeltaTable
from deltalake import write_deltalake

class LakehouseManager:
    def __init__(self, storage_options):
        self.__storage_options = storage_options
        self.__base_path = "s3://lakehouse"

    def run_bronze(self, csv_path: str):
        df = pl.read_csv(csv_path) 
        years = df["Year"].unique().sort().to_list()

        for year in years:
            print(f"Processing year: {year}...")
            year_data = df.filter(pl.col("Year") == year)
            write_deltalake(f"{self.__base_path}/bronze", year_data, mode="append", storage_options=self.__storage_options)

    def run_silver(self):
        bronze_df = pl.scan_delta(f"{self.__base_path}/bronze", storage_options=self.__storage_options)
        
        silver_df = (
            bronze_df
            .filter(
                (pl.col("Cancelled") == 0) & 
                (pl.col("Diverted") == 0) &
                (pl.col("ArrDelay").is_between(-60, 1440)) 
            )
            .drop_nulls([
                "FlightDate", "Year", "CRSDepTime", "Month", "DayOfWeek", "Origin", "Dest", 
                "Marketing_Airline_Network", "ArrDelay",
                "Flight_Number_Marketing_Airline", "Distance"
            ])
            
            .with_columns([
                pl.col("Origin").str.strip_chars().str.to_uppercase(),
                pl.col("Dest").str.strip_chars().str.to_uppercase(),
                pl.col("Marketing_Airline_Network").str.strip_chars().str.to_uppercase(),
            ])
            
            .with_columns([
                (pl.col("CRSDepTime") // 100).alias("Hour"),
                pl.when(pl.col("Month").is_in([12, 1, 2])).then(pl.lit("Winter"))
                  .when(pl.col("Month").is_in([3, 4, 5])).then(pl.lit("Spring"))
                  .when(pl.col("Month").is_in([6, 7, 8])).then(pl.lit("Summer"))
                  .otherwise(pl.lit("Autumn")).alias("Season"),
                (pl.col("Origin") + "-" + pl.col("Dest")).alias("Route")
            ])
            .select([
                "FlightDate", "Year", "Month", "Hour", "DayOfWeek", 
                "Season", "Origin", "Dest", "Route", "Marketing_Airline_Network", "ArrDelay",
                "Flight_Number_Marketing_Airline", "Distance"
            ])
        )

        try:
            dt = DeltaTable(f"{self.__base_path}/silver", storage_options=self.__storage_options)
            (
                dt.merge(
                    source=silver_df.collect(),
                    predicate="""
                        target.FlightDate = source.FlightDate AND 
                        target.Marketing_Airline_Network = source.Marketing_Airline_Network AND 
                        target.Flight_Number_Marketing_Airline = source.Flight_Number_Marketing_Airline AND 
                        target.Origin = source.Origin AND
                        target.CRSDepTime = source.CRSDepTime
                    """,
                    source_alias="source",
                    target_alias="target"
                )
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute()
            )
        except Exception:
            write_deltalake(
                f"{self.__base_path}/silver",
                silver_df.collect(),
                mode="append",
                partition_by=["Year", "Month"],
                storage_options=self.__storage_options
            )

    def run_gold(self):
        silver_df = pl.scan_delta(f"{self.__base_path}/silver", storage_options=self.__storage_options)
        
        analytics_df = (
            silver_df
            .group_by(["Origin", "Marketing_Airline_Network", "Season", "Hour"])
            .agg(pl.col("ArrDelay").mean().alias("AvgDelay"))
        )
        
        features_df = silver_df.select([
            "Hour", "DayOfWeek", "Season", "Origin", "Dest", "ArrDelay", "Distance"
        ]).drop_nulls()
        
        write_deltalake(f"{self.__base_path}/gold_analytics", analytics_df.collect(), 
                    mode="overwrite", storage_options=self.__storage_options)
        write_deltalake(f"{self.__base_path}/gold_features", features_df.collect(), 
                    mode="overwrite", storage_options=self.__storage_options)

    def maintenance(self):
        dt = DeltaTable(f"{self.__base_path}/silver", storage_options=self.__storage_options)
        dt.optimize.compact()
        dt.optimize.z_order(["Origin", "FlightDate", "Route"])
        dt.vacuum(retention_hours=168, dry_run=False)