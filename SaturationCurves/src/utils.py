import json
import pandas as pd
from google.cloud import secretmanager
from google.oauth2 import service_account
from google.cloud import bigquery


def get_secret(secret_name):
    client = secretmanager.SecretManagerServiceClient()
    response = client.access_secret_version(
        request={"name": f"projects/relax-server/secrets/{secret_name}/versions/latest"}
    )
    decoded = response.payload.data.decode("UTF-8")
    return decoded


def get_bigquery_client():
    #  Note: need to define $GOOGLE_APPLICATION_CREDENTIALS josn path in environment variable
    credentials = get_secret("GOOGLE_APPLICATION_CREDENTIALS_MAIN_BIGQUERY")
    project_id = get_secret("GOOGLE_APPLICATION_PROJECT_ID_MAIN_BIGQUERY")

    info = json.loads(credentials)
    credentials = service_account.Credentials.from_service_account_info(info)
    client = bigquery.Client(credentials=credentials, project=project_id)
    #  client = bigquery.Client()
    return client


def run_query(client: bigquery.Client, query: str) -> pd.DataFrame:
    try:
        df = client.query(query).to_dataframe()
    except:
        raise ValueError(f"Error when running query \n{query}")

    return df


client = get_bigquery_client()
print(client)
