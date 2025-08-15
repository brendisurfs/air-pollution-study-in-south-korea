from dataclasses import dataclass
from enum import StrEnum
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

# PM2.5 is generally considered Hazardous above 250 micrograms/m3
HAZARDOUS_PM2_LEVEL = 250.0


@dataclass
class Measurement:
    """
    Class defining the structure of data for each measurement reading from sensors.

    Attributes: 
        date: 
        station_code: 
        address: 
        latitude: 
        longitude: 
        so2: 
        no2: 
        oxide: 
        carbon_monoxide: 
        pm10: 
        pm25: 
    """
    date: str
    station_code: int
    address: str
    latitude: float
    longitude: float
    so2: float
    no2: float
    oxide: float
    carbon_monoxide: float
    pm10: float
    pm25: float


# Constants
class Column(StrEnum):
    CO = "CO"
    O3 = "O3"
    SO2 = "SO2"
    NO2 = "NO2"
    PM2 = "PM2.5"
    PM10 = "PM10"
    DATE = "Date"
    ADDRESS = "Address"
    LATITUDE = "Latitude"
    LONGITUDE = "Longitude"
    STATION_CODE = "StationCode"


def plot_correlation(df: pl.DataFrame):
    """
    Plots the dataframe as a correlation matrix

    Args:
        df: the Dataframe to operation on
    """
    data = df.to_numpy()
    labels = df.schema.names()
    plt.figure(figsize=(8, 6))
    sns.heatmap(data,
                annot=True,
                cmap="RdBu_r",
                center=0,
                xticklabels=labels,
                yticklabels=labels,
                square=True,
                linewidths=0.5)
    plt.title("Air Quality Parameters Correlation Matrix")
    plt.tight_layout()
    plt.show()


def rolling_correlation(df: pl.DataFrame,
                        col1: str,
                        col2: str,
                        window_size: str = "30d"):
    """
    a custom rolling correlation function for analyzing our df through a given period.

    Args:
        df: 
        col1: 
        col2: 
        window_size: 
    """
    pass


def clean_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Filters out NaN and Null values within a given dataset.
    Args:
        df: DataFrame

    Returns: DataFrame
    """
    df = df.filter(pl.all().is_not_nan())
    df = df.fill_nan(0.0)
    return df


def detect_iqr_outliers(df: pl.DataFrame,
                        column: str,
                        mult: float = 1.5) -> pl.DataFrame:
    """detect outliers using IQR

    Args:
        df: DataFrame
        column: str
        mult: float

    Returns: DataFrame
        
    """
    return df.with_columns([
        pl.col(column).quantile(0.25).alias(f"{column}_q1"),
        pl.col(column).quantile(0.75).alias(f"{column}_q3")
    ]).with_columns([
        # Calculates IQR
        (pl.col(f"{column}_q3") - pl.col(f"{column}_q1")
         ).alias(f"{column}_iqr")
    ]).with_columns([
        # Calculate Bounds
        (pl.col(f"{column}_q1") - mult * pl.col(f"{column}_iqr")
         ).alias(f"{column}_lower_bound"),
        (pl.col(f"{column}_q3") +
         mult * pl.col(f"{column}_iqr")).alias(f"{column}_upper_bound"),
    ]).with_columns([
        ((pl.col(column) < pl.col(f"{column}_lower_bound")) |
         (pl.col(column)
          > pl.col(f"{column}_upper_bound"))).alias(f"{column}_is_outlier"),
        # Calculate outlier severity (how far beyond bounds)
        pl.when(pl.col(column) > pl.col(f"{column}_upper_bound")).then(
            (pl.col(column) - pl.col(f"{column}_upper_bound")) /
            pl.col(f"{column}_iqr")
        ).when(pl.col(column) < pl.col(f"{column}_lower_bound")).then(
            (pl.col(f"{column}_lower_bound") - pl.col(column)) /
            pl.col(f"{column}_iqr")
        ).otherwise(0.0).alias(f"{column}_outlier_severity")
    ])


def cluster_outlier_frequency_severity(df: pl.DataFrame) -> pl.DataFrame:
    """Clusters outliers by their severity, returning frequency and mean severity.
    Args:
        df: DataFrame

    Returns: DataFrame
    """
    return df.group_by("compound_name").agg([
        pl.len().alias("frequency"),
        pl.col("outlier_severity").mean().alias("mean_severity").round(3),
        pl.col("outlier_severity").std().alias("std_severity").round(3),
    ])


def detect_compound_outliers(df: pl.DataFrame,
                             compound_cols: list[Column],
                             mult: float = 1.5) -> pl.DataFrame:
    """
    Detect outliers across multiple pollutants at once
    """
    result_df = df

    for col in compound_cols:
        result_df = detect_iqr_outliers(result_df, col, mult)

    outlier_cols = [f"{col}_is_outlier" for col in compound_cols]
    severity_cols = [f"{col}_outlier_severity" for col in compound_cols]

    return result_df.with_columns([
        pl.any_horizontal(outlier_cols).alias("any_pollutant_outlier"),
        pl.sum_horizontal(outlier_cols).alias("num_pollutants_outlier"),
        pl.max_horizontal(severity_cols).alias("max_outlier_severity"),
        pl.mean_horizontal(severity_cols).alias("avg_outlier_severity")
    ])


def aggregate_outliers(
    df: pl.DataFrame,
    pollutant_columns: list[Column],
) -> pl.DataFrame:
    # Get all relevant columns
    relevant_cols = [Column.DATE.value]
    for col in pollutant_columns:
        relevant_cols.extend([
            col, f"{col}_is_outlier", f"{col}_lower_bound",
            f"{col}_upper_bound", f"{col}_iqr", f"{col}_outlier_severity"
        ])

    # Select only rows where any pollutant is an outlier
    any_outlier = pl.any_horizontal(
        [pl.col(f"{col}_is_outlier") for col in pollutant_columns])

    df_filtered = df.filter(any_outlier).select(relevant_cols)

    # Process each compound and stack results
    compound_results: list[pl.DataFrame] = []

    for compound in pollutant_columns:
        compound_data = (df_filtered.filter(
            pl.col(f"{compound}_is_outlier")
        ).select([
            pl.col("Date"),
            pl.lit(compound).alias("compound_name"),
            pl.col(compound).alias("compound_value"),
            pl.col(f"{compound}_iqr").alias("compound_iqr"),
            pl.col(f"{compound}_lower_bound").alias("compound_lower_bound"),
            pl.col(f"{compound}_upper_bound").alias("compound_upper_bound"),
            pl.col(f"{compound}_outlier_severity").alias("outlier_severity"),
        ]))
        compound_results.append(compound_data)

    return pl.concat(compound_results).sort(["Date", "compound_name"])


def main():
    compound_list = [
        Column.CO,
        Column.O3,
        Column.SO2,
        Column.NO2,
        Column.PM2,
        Column.PM10,
    ]

    # Cast our features to the proper types
    df = pl.read_csv("./AirPollutionSeoul/Measurement_summary.csv",
                     schema={
                         "Date": pl.Datetime,
                         "StationCode": pl.Int64,
                         "Address": pl.String,
                         "Latitude": pl.Float64,
                         "Longitude": pl.Float64,
                         "SO2": pl.Float64,
                         "NO2": pl.Float64,
                         "O3": pl.Float64,
                         "CO": pl.Float64,
                         "PM10": pl.Float64,
                         "PM2.5": pl.Float64,
                     })
    df = df.drop(["Address"])
    df = df.filter(pl.col("SO2") > 0)

    # Display the mean values of each chemical compound
    mean_so2 = df.select(
        pl.col("SO2").mean().round(3)).get_column("SO2").first()
    print(f"Mean SO2: { mean_so2 }")

    mean_no2 = df.select(
        pl.col("NO2").mean().round(3)).get_column("NO2").first()
    print(f"Mean NO2: { mean_no2 }")

    mean_ozone = df.select(pl.col(Column.O3).mean().round(3)).get_column(
        Column.O3).first()
    print(f"Mean Ozone: {mean_ozone}")

    # Correlation
    particle_corr = df.drop(
        [Column.DATE, Column.STATION_CODE, Column.LATITUDE,
         Column.LONGITUDE]).corr().select(pl.col(pl.Float64).round(3))
    print(f"Particle Corr.: {particle_corr}")

    # Finding Where PM2.5 leves spike beyond 300ppm
    # Looking for dates where PM2 and PM10 particle emissions are highest.
    highest_particles = df.filter(pl.col("PM2.5") > HAZARDOUS_PM2_LEVEL)

    # Finding the average time across datetimes.
    mean_highest_particle_time = highest_particles.get_column("Date").cast(
        pl.Datetime).mean()

    print(f"Mean Highest Time: { mean_highest_particle_time }")

    # Outliers
    outlier_df = detect_compound_outliers(df, compound_list)
    compound_outliers_df = aggregate_outliers(outlier_df, compound_list)
    total_outliers = cluster_outlier_frequency_severity(
        compound_outliers_df).select(pl.col("frequency").sum())

    outlier_pct = float(
        (total_outliers / df.count()).get_column("frequency").round(5).cast(
            pl.Float64).to_list()[0])

    # Shows the outlier as a proper percentage.
    print(f"Outlier Percentage: {outlier_pct * 100.0}")

    # Plot our data
    # NOTE: Plots our particles correlation via heat map
    plot_correlation(particle_corr)


if __name__ == "__main__":
    main()
