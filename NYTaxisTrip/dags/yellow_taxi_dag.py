from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

import pandas as pd
from sqlalchemy import create_engine
from time import time

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

#  url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
CSV_TMP_NAME = "output.csv"
PARQUET_TMP_NAME = "output.parquet"


# -- using aws
#  dataset_file = "yellow_tripdata_2021-01.csv"
#  dataset_url = f"https://s3.amazonaws.com/nyc-tlc/trip+data/{dataset_file}"


def build_url(year: int, month: int, taxi_color: str) -> str:
    """
    Parameters
    ----------
    year: str
        > Format: YYYY
    month: str
        > Format: MM
    taxi_color: str
        > should be either 'green' or 'yellow'

    Examples
    --------
    """
    # - TODO: check params values

    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_color}_tripdata_{year}-{month}.parquet"
    return url


def download_dataset(url: str) -> None:
    df = pd.read_parquet(url)
    df.to_csv(CSV_TMP_NAME, index=False)


def push_to_postgres():  # TODO
    # - establish connection with db

    # - iteratively push batches of data to db
    pass


default_args = {
    "owner": "airflow",
    "start_date": days_ago(1),
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    dag_id="data_ingestion",
    schedule_interval="@monthly",
    default_args=default_args,
    catchup=False,
    max_active_runs=1,
    tags=["taxis"],
) as dag:
    build_url_task = PythonOperator(
        task_id="build_url_task",
        python_callable=build_url,
        op_kwargs={"year": "2023", "month": "01", "taxi_color": "yellow"},
    )

    build_url_task
