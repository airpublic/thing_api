from thingapi.pollutionrose import get_pollution_rose_data

import pandas as pd

test_csv = 'data/dummy_wind.csv'

def test_get_pollution_rose_data():
    df = pd.read_csv(test_csv)
    wind_rose_data, pollution_rose_data = get_pollution_rose_data(df, pollutant_col='no2')
    assert len(wind_rose_data) == len(pollution_rose_data)
    assert len(wind_rose_data) == 24
    assert wind_rose_data['0 - 5 knots'].sum() == 60
    assert pollution_rose_data['0 - 5 knots'].sum() == 180


def test_weather_input_format():
    # dummy weather loader
    # 15 knots, rotates
