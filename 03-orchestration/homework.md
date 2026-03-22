## Homework

The goal of this homework is to create a simple training pipeline, use mlflow to track experiments and register best model, and a modern data workflow orchestration tool such as Mage or Prefect or others as listed in the course material under '03-orchestration'.

We'll use [the same NYC taxi dataset](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page), the **Yellow** taxi data for March, 2023. 

## Question 1. Select the Tool
You can use the same tool you used when completing the module,
or choose a different one for your homework.
What's the name of the orchestrator you chose? 

Prefect

## Question 2. Version
What's the version of the orchestrator? 

3.6.23

## Question 3. Creating a pipeline
Let's read the March 2023 Yellow taxi trips data.

mlflow_env/bin/python -m pip install pandas pyarrow

DO_NOT_TRACK=1 mlflow_env/bin/python 03-orchestration/code/duration-prediction.py --year 2023 --month 3 --color yellow --load-only

How many records did we load? 

- 3,403,766

(Include a print statement in your code)

## Question 4. Data preparation

Let's continue with pipeline creation.

We will use the same logic for preparing the data we used previously. 

This is what we used (adjusted for yellow dataset):

```python
def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df
```

DO_NOT_TRACK=1 mlflow_env/bin/python 03-orchestration/code/duration-prediction.py --year 2023 --month 3 --color yellow --prepare-only

Let's apply to the data we loaded in question 3. 

What's the size of the result? 

- 3,316,216 

## Question 5. Train a model

We will now train a linear regression model using the same code as in homework 1.

* Fit a dict vectorizer.
* Train a linear regression with default parameters.
* Use pick up and drop off locations separately, don't create a combination feature.

Let's now use it in the pipeline. We will need to create another transformation block, and return both the dict vectorizer and the model.

What's the intercept of the model? 

Hint: print the `intercept_` field in the code block

DO_NOT_TRACK=1 mlflow_env/bin/python 03-orchestration/code/duration-prediction.py --year 2023 --month 3 --color yellow --train-linear-only

- 24.77

## Question 6. Register the model 

The model is trained, so let's save it with MLFlow.

Find the logged model, and find MLModel file. What's the size of the model? (`model_size_bytes` field):

mlflow_env/bin/python -m pip install mlflow

cd /workspaces/mlops-zoomcamp && DO_NOT_TRACK=1 mlflow_env/bin/python - <<'PY'
import os
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import mlflow

url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'
df = pd.read_parquet(url)

df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
df.duration = df.duration.dt.total_seconds() / 60
df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

categorical = ['PULocationID', 'DOLocationID']
df[categorical] = df[categorical].astype(str)

train_dicts = df[categorical].to_dict(orient='records')
dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)
y_train = df['duration'].values

lr = LinearRegression()
lr.fit(X_train, y_train)

mlflow.set_tracking_uri('file:///workspaces/mlops-zoomcamp/03-orchestration/mlruns')
mlflow.set_experiment('orchestration-hw3-q6')
with mlflow.start_run() as run:
    model_info = mlflow.sklearn.log_model(lr, artifact_path='model')
    print('RUN_ID', run.info.run_id)
    print('MODEL_URI', model_info.model_uri)
PY

* 4,534


## Submit the results

* Submit your results here: https://courses.datatalks.club/mlops-zoomcamp-2025/homework/hw3
* If your answer doesn't match options exactly, select the closest one.