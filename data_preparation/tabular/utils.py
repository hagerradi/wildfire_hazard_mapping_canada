import numpy as np
import pandas as pd

# features of fire weather list to include
weather_column_names = [
    "WeatherZone",
    "Season",
    "Temperature",
    "RelativeHumidity",
    "Precipitation",
    "FineFuelMoistureCode",
    "DuffMoistureCode",
    "DroughtCode",
    "InitialSpreadIndex",
    "BuildupIndex",
    "FireWeatherIndex",
    "WindSpeed",
    "WindDirection",
]


def check_column_format(df: pd.DataFrame, col_name: str) -> np.bool:
    # Regex Explanation:
    # ^   = Start of string
    # s   = Literal letter 's'
    # \d+ = One or more digits
    # $   = End of string
    pattern = r"^[a-zA-Z]+\d+$"

    # 1. Coerce to string (in case some are ints)
    # 2. Check match
    # 3. .all() ensures EVERY row matches
    is_valid = df[col_name].astype(str).str.match(pattern).all()
    return is_valid


def check_weather_list(weather_list: pd.DataFrame) -> pd.DataFrame:
    """
    Check if the weather list is of the required format (columns) and the season and WeatherZone column
    """
    if set(list(weather_column_names)).issubset(weather_list.columns):
        return weather_list
    if not set(list(weather_column_names)).issubset(weather_list.columns):
        raise ValueError("Missing columns/ weather df not in required format")
    print("==============hexel did not have the req columns===================")
    if check_column_format(weather_list, "Season"):
        print("=============Season check not passed==================")
        weather_list["Season"] = weather_list["Season"].astype(str).str.extract(r"(\d+)").astype(int)
    if check_column_format(weather_list, "WeatherZone"):
        print("=============WeatherZone check not passed==================")
        weather_list["WeatherZone"] = weather_list["WeatherZone"].astype(str).str.extract(r"(\d+)").astype(int)

    return weather_list
