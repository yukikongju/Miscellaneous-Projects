from google.cloud import bigquery
from src.utils.bigquery_utils import get_bigquery_client
from src.tables.constants import PROJECT_ID, DATASET_ID
from google.api_core.exceptions import NotFound

SCHEMA = [
    bigquery.SchemaField("segments_date", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("platform", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("country", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("metrics_cost_micros", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("metrics_impressions", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("metrics_clicks", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("metrics_installs", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("metrics_conversions", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("metrics_paid", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("metrics_revenue", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("loaded_datetime", "FLOAT64", mode="NULLABLE"),
]

TABLE_NAME = "googleads"
TABLE_PATH = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}"
TABLE_API = bigquery.Table(TABLE_PATH, schema=SCHEMA)

client = get_bigquery_client()
#  client.delete_table(TABLE_PATH, not_found_ok=True)
try:
    table = client.get_table(TABLE_API)
    print(f"Table {TABLE_PATH} already exists.")
except NotFound:
    print(f"Table {TABLE_PATH} does not exist. Creating..")
    table = client.create_table(TABLE_API)
