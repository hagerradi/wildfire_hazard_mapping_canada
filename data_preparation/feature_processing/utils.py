import numpy as np
import pandas as pd

# Renaming the columns of df
column_full_form_abrevation_map = {
    "WeatherZone": "wx_zone",
    "Season": "season",
    "Temperature": "temp",
    "RelativeHumidity": "rh",
    "WindSpeed": "ws",
    "WindDirection": "wd",
    "Precipitation": "prec",
    "FineFuelMoistureCode": "ffmc",
    "DuffMoistureCode": "dmc",
    "DroughtCode": "dc",
    "InitialSpreadIndex": "isi",
    "BuildupIndex": "bui",
    "FireWeatherIndex": "fwi",
}


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
    Check if the weather list is of the required format (columns) and the season and wx_zone column
    """
    if set(list(column_full_form_abrevation_map.values())).issubset(weather_list.columns):
        return weather_list
    if not set(list(column_full_form_abrevation_map.keys())).issubset(weather_list.columns):
        raise ValueError("Missing columns/ weather df not in required format")
    print("==============hexel didnot have the req columns===================")
    weather_list = weather_list.rename(columns=column_full_form_abrevation_map)
    if check_column_format(weather_list, "season"):
        print("=============Season check not passed==================")
        weather_list["season"] = weather_list["season"].astype(str).str.extract(r"(\d+)").astype(int)
    if check_column_format(weather_list, "wx_zone"):
        print("=============wx_zone check not passed==================")
        weather_list["wx_zone"] = weather_list["wx_zone"].astype(str).str.extract(r"(\d+)").astype(int)
    return weather_list
