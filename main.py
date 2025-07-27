from dataclasses import dataclass
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

# PM2.5 is generally considered Hazardous above 250micrograms/m3
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


def plot_linear_regression_corr(df: pl.DataFrame):
    data = df.to_numpy()
    labels = df.schema.names()
    print(labels)
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


def main():
    # Load file
    print("loading csv")
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

    print("Dropping address")
    df = df.drop(["Address"])
    df = df.filter(pl.col("SO2") > 0)

    # mean values
    mean_so2 = df.select(
        pl.col("SO2").mean().round(3)).get_column("SO2").first()
    print(f"Mean SO2: { mean_so2 }")

    mean_no2 = df.select(
        pl.col("NO2").mean().round(3)).get_column("NO2").first()
    print(f"Mean NO2: { mean_no2 }")

    mean_ozone = df.select(pl.col("O3").mean().round(3)).get_column(O3).first()
    print(f"Mean Ozone: {mean_ozone}")

    # Temporal Findings
    # Finding Where PM2.5 leves spike beyond 300ppm
    particle_corr = df.drop([DATE, STATION_CODE, LATITUDE, LONGITUDE
                             ]).corr().select(pl.col(pl.Float64).round(3))
    print(f"Particle Corr.: {particle_corr}")

    # Looking for dates where PM2 and PM10 particle emissions are highest.
    highest_particles = df.filter(pl.col("PM2.5") > HAZARDOUS_PM2_LEVEL)

    # Finding the average time across datetimes.
    mean_highest_particle_time = highest_particles.get_column("Date").cast(
        pl.Datetime).mean()

    print(f"Mean Highest Time: { mean_highest_particle_time }")
    plot_linear_regression_corr(particle_corr)


if __name__ == "__main__":
    main()
