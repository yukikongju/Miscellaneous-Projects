#  import os
#  import json

#  from dotenv import load_dotenv
from google.cloud import bigquery

#  from google.cloud import secretmanager
#  from google.cloud.bigquery import client
#  from google.oauth2 import service_account


def get_bigquery_client():
    client = bigquery.Client()
    return client


#  def get_secret(secret_name: str):
#      client = secretmanager.SecretManagerServiceClient()
#      response = client.access_secret_version(
#          request={"name": f"projects/relax-server/secrets/{secret_name}/versions/latest"}
#      )
#      decoded = response.payload.data.decode("UTF-8")
#      return decoded


#  def get_bigquery_client_from_secret():
#      credentials = get_secret("GOOGLE_APPLICATION_CREDENTIALS_MAIN_BIGQUERY")
#      project_id = get_secret("GOOGLE_APPLICATION_PROJECT_ID_MAIN_BIGQUERY")
#      info = json.loads(credentials)
#      credentials = service_account.Credentials.from_service_account_info(info)
#      client = bigquery.Client(credentials=credentials, project=project_id)
#      return client


#  def get_bigquery_client_from_json():
#      load_dotenv()
#      key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_KEY")
#      project_id = os.getenv("GOOGLE_PROJECT_ID")

#      # read google json key
#      try:
#          with open(key_path, "r") as f:
#              info = json.load(f)
#      except Exception as e:
#          raise FileNotFoundError(f"File {key_path} could not be found: {e}")

#      credentials = service_account.Credentials.from_service_account_info(info)
#      client = bigquery.Client(credentials=credentials, project=project_id)
#      return client
