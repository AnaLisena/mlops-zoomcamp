#!/usr/bin/env python

import argparse
import os
import pickle

import pandas as pd


CATEGORICAL = ["PULocationID", "DOLocationID"]


def get_input_path(year: int, month: int) -> str:
    return (
        "https://d37ci6vzurychx.cloudfront.net/trip-data/"
        f"yellow_tripdata_{year:04d}-{month:02d}.parquet"
    )


def get_output_path(year: int, month: int) -> str:
    return f"yellow_tripdata_{year:04d}-{month:02d}.parquet"


def read_data(filename: str) -> pd.DataFrame:
    df = pd.read_parquet(filename)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[CATEGORICAL] = df[CATEGORICAL].fillna(-1).astype("int").astype("str")

    return df


def main(year: int, month: int) -> None:
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    with open("model.bin", "rb") as f_in:
        dv, model = pickle.load(f_in)

    df = read_data(input_file)
    dicts = df[CATEGORICAL].to_dict(orient="records")

    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print(f"mean predicted duration: {y_pred.mean():.2f}")

    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")
    df_result = pd.DataFrame(
        {
            "ride_id": df["ride_id"],
            "predicted_duration": y_pred,
        }
    )

    df_result.to_parquet(
        output_file,
        engine="pyarrow",
        compression=None,
        index=False,
    )

    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"output file: {output_file}")
    print(f"output size: {size_mb:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("year", type=int)
    parser.add_argument("month", type=int)
    args = parser.parse_args()

    main(args.year, args.month)




