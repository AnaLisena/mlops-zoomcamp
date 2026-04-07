import pandas as pd
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from batch import prepare_data


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    categorical = ['PULocationID', 'DOLocationID']
    actual = prepare_data(df, categorical)

    expected_data = [
        ('-1', '-1', 9.0),
        ('1', '1', 8.0),
    ]
    expected_df = pd.DataFrame(expected_data, columns=['PULocationID', 'DOLocationID', 'duration'])

    actual_records = actual[['PULocationID', 'DOLocationID', 'duration']].to_dict(orient='records')
    expected_records = expected_df.to_dict(orient='records')

    assert len(actual) == 2
    assert actual_records == expected_records

from batch import prepare_data


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    categorical = ['PULocationID', 'DOLocationID']
    actual = prepare_data(df, categorical)

    expected = [
        {'-1', '-1', 9.0},
        {'1', '1', 8.0},
    ]

    expected_data = [
        ('-1', '-1', 9.0),
        ('1', '1', 8.0),
    ]
    expected_df = pd.DataFrame(expected_data, columns=['PULocationID', 'DOLocationID', 'duration'])

    actual_records = actual[['PULocationID', 'DOLocationID', 'duration']].to_dict(orient='records')
    expected_records = expected_df.to_dict(orient='records')

    assert actual_records == expected_records
