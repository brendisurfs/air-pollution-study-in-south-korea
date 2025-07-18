import polars as pd


def main():
    # Load file

    df = pd.read_csv("./AirPollutionSeoul/Measurement_summary.csv")

    print(df.head(5))


if __name__ == "__main__":
    main()
