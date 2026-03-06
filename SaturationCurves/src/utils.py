"""Utilities for BigQuery execution and Secret Manager access."""

import json

import pandas as pd
from google.cloud import bigquery, secretmanager
from google.oauth2 import service_account


def get_secret(secret_name):
    """Fetch the latest value of a secret from Secret Manager.

    Args:
        secret_name: Secret name in project `relax-server`.

    Returns:
        Decoded secret payload.
    """
    client = secretmanager.SecretManagerServiceClient()
    response = client.access_secret_version(
        request={"name": f"projects/relax-server/secrets/{secret_name}/versions/latest"}
    )
    decoded = response.payload.data.decode("UTF-8")
    return decoded


def get_bigquery_client():
    """Create a BigQuery client using Application Default Credentials.

    Returns:
        Initialized BigQuery client.
    """
    #  Note: need to define $GOOGLE_APPLICATION_CREDENTIALS josn path in environment variable
    #  credentials = get_secret("GOOGLE_APPLICATION_CREDENTIALS_MAIN_BIGQUERY")
    #  project_id = get_secret("GOOGLE_APPLICATION_PROJECT_ID_MAIN_BIGQUERY")

    #  info = json.loads(credentials)
    #  credentials = service_account.Credentials.from_service_account_info(info)
    #  client = bigquery.Client(credentials=credentials, project=project_id)
    client = bigquery.Client()
    return client


def run_query(client: bigquery.Client, query: str) -> pd.DataFrame:
    """Run a SQL query and return results as a pandas DataFrame.

    Args:
        client: BigQuery client instance.
        query: SQL query string.

    Returns:
        Query result as a pandas DataFrame.

    Raises:
        ValueError: If the client is null or query execution fails.
    """
    if client == None:
        raise ValueError("client is null")
    try:
        df = client.query(query).to_dataframe()
    except Exception as e:
        raise ValueError(f"Error when running query \n{query}: {e}")

    return df
