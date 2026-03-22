#!/usr/bin/env python
# coding: utf-8

import pickle
from pathlib import Path

from prefect import flow, task
import pandas as pd

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


@task(log_prints=True)
def load_dataframe(year, month, color='yellow'):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{color}_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    print(f"Loaded {len(df)} records from {url}")
    return df


def read_dataframe(df):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df


@task(log_prints=True)
def prepare_dataframe(df):
    prepared = read_dataframe(df)
    print(f"Prepared dataframe has {len(prepared)} records")
    return prepared


def create_X(df, dv=None):
    from sklearn.feature_extraction import DictVectorizer

    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


@task(log_prints=True)
def train_linear_regression_model(df):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LinearRegression

    categorical = ['PULocationID', 'DOLocationID']
    train_dicts = df[categorical].to_dict(orient='records')

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)

    y_train = df['duration'].values
    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"Linear regression intercept: {model.intercept_}")
    return dv, model, float(model.intercept_)


def train_model(X_train, y_train, X_val, y_val, dv):
    import mlflow
    import xgboost as xgb

    from sklearn.metrics import root_mean_squared_error

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("nyc-taxi-experiment")

    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        return run.info.run_id


@flow(name='duration-prediction-flow')
def run(year, month, color='yellow', load_only=False, prepare_only=False, train_linear_only=False):
    df_train_raw = load_dataframe(year=year, month=month, color=color)

    if load_only:
        return len(df_train_raw)

    df_train = prepare_dataframe(df_train_raw)

    if prepare_only:
        return len(df_train)

    if train_linear_only:
        _, _, intercept = train_linear_regression_model(df_train)
        return intercept

    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val_raw = load_dataframe(year=next_year, month=next_month, color=color)
    df_val = prepare_dataframe(df_val_raw)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"MLflow run_id: {run_id}")
    return run_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    parser.add_argument('--color', type=str, default='yellow', help='Taxi dataset color to train on')
    parser.add_argument('--load-only', action='store_true', help='Only load the data and print the number of records')
    parser.add_argument('--prepare-only', action='store_true', help='Only load and prepare the data, then print prepared size')
    parser.add_argument('--train-linear-only', action='store_true', help='Only load, prepare, and train linear regression for intercept')
    args = parser.parse_args()

    run_id = run(
        year=args.year,
        month=args.month,
        color=args.color,
        load_only=args.load_only,
        prepare_only=args.prepare_only,
        train_linear_only=args.train_linear_only,
    )

    with open("run_id.txt", "w") as f:
        f.write(str(run_id))