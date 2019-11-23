import requests
import datetime as dt
import json
import pandas as pd

BASEURL = "https://api.meteostat.net/v1/history/hourly?"

HAMBURG_ID = "2911298"
## According to container dataset
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
STD_TYPE="hour"
API_KEY = "Bj22nPm3"
STATION_ID = "10147"
TIME_ZONE ="Europe/Berlin"
TIME_FORMAT="Y-m-d%2520H:i"

DATE_IN_FORMAT = "%d.%m.%Y"



def request_weather(start, end):
    """ Returns JSON data for all day between start and end

    date_string format : Y-m-d

    """
    # Parse Date string to API format
    request_formatted = BASEURL + "station={0}&start={1}&end={2}&time_zone={3}&time_format={4}&key={5}".format(STATION_ID, start, end, TIME_ZONE, TIME_FORMAT, API_KEY)
    r = requests.get(request_formatted)
    print(r.status_code)
    data = r.json()['data']

    return data


def parse_request(data):
    df = pd.DataFrame(columns=["time", "temp", "prec", "wind"])

    for single in data:
        row = [single["time"], single["temperature"], single["precipitation"], single["windspeed"]]
        df.loc[len(df)] = row


    # format date as datetime
    df['time'] = pd.to_datetime(df['time'], infer_datetime_format=True)
    df['hour_of_day'] = df['time'].dt.hour
    df['date'] = df['time'].dt.date
    del df['time']

    return df

def weather_at_date(date):
    """ Get a dataframe of weather information for the requested date"""
    start = dt.datetime.strftime(date, "%Y-%m-%d")
    end = start
    data = request_weather(start, end)
    df = parse_request(data)
    return df.loc[df['date'] == date]


def weather_at_datetime(datetime):
    """
    Get a dataframe of weather information (one row) for the requested datetime

    Only parses hour-wise information between 00:00 and 21:00  at the moment

    :param datetime: datetime object out of container dataset with hours
    :return:
    """
    df = weather_at_date(datetime.date())
    return df.loc[df['hour_of_day'] == datetime.hour]




