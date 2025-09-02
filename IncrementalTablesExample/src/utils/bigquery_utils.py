import os
import json

from dotenv import load_dotenv
from google.cloud import bigquery
from google.oauth2 import service_account


def get_bigquery_client():
    load_dotenv()
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_KEY")
    project_id = os.getenv("GOOGLE_PROJECT_ID")

    # read google json key
    try:
        with open(key_path, "r") as f:
            info = json.load(f)
    except Exception as e:
        raise FileNotFoundError(f"File {key_path} could not be found: {e}")

    credentials = service_account.Credentials.from_service_account_info(info)
    client = bigquery.Client(credentials=credentials, project=project_id)
    return client
