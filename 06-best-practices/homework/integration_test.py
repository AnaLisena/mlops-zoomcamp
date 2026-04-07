from datetime import datetime
import os
import subprocess
import sys

import pandas as pd


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def main():
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    os.environ.setdefault("S3_ENDPOINT_URL", "http://localhost:4566")
    os.environ.setdefault("INPUT_FILE_PATTERN", "s3://nyc-duration/in/{year:04d}-{month:02d}.parquet")
    os.environ.setdefault("OUTPUT_FILE_PATTERN", "s3://nyc-duration/out/{year:04d}-{month:02d}.parquet")

    endpoint = os.environ["S3_ENDPOINT_URL"]

    subprocess.run(
        ["aws", "s3", "mb", "s3://nyc-duration", "--endpoint-url", endpoint],
        check=False,
    )

    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = [
        "PULocationID",
        "DOLocationID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
    ]

    df_input = pd.DataFrame(data, columns=columns)

    input_file = "s3://nyc-duration/in/2023-01.parquet"

    local_input_file = "integration_input_2023-01.parquet"
    df_input.to_parquet(
        local_input_file,
        engine="pyarrow",
        compression=None,
        index=False,
    )
    subprocess.run(
        ["aws", "s3", "cp", local_input_file, input_file, "--endpoint-url", endpoint],
        check=True,
    )

    print(f"Wrote test data to {input_file}")

    code = os.system(f"{sys.executable} batch.py 2023 1")
    if code != 0:
        raise SystemExit(code)

    output_file = "s3://nyc-duration/out/2023-01.parquet"
    local_output_file = "integration_output_2023-01.parquet"
    subprocess.run(
        ["aws", "s3", "cp", output_file, local_output_file, "--endpoint-url", endpoint],
        check=True,
    )

    df_output = pd.read_parquet(local_output_file)
    print(f"sum of predicted durations: {df_output.predicted_duration.sum():.2f}")


if __name__ == "__main__":
    main()
