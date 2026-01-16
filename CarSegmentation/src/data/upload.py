"""
This script is used to load the car segmentation dataset to GCP buckets from the repo
"https://github.com/alexgkendall/SegNet-Tutorial", which reside in `SegNet-Tutorial/CamVid`

This script assumes that you have clone the repository locally:
> !git clone https://github.com/alexgkendall/SegNet-Tutorial ./data
"""

import io
import json
import os
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import storage
from google.cloud import secretmanager
from google.oauth2 import service_account
from PIL import Image
from tqdm import tqdm


def get_secret(secret_name):
    client = secretmanager.SecretManagerServiceClient()
    response = client.access_secret_version(
        request={"name": f"projects/relax-server/secrets/{secret_name}/versions/latest"}
    )
    decoded = response.payload.data.decode("UTF-8")
    return decoded


def get_storage_client() -> storage.Client:
    """
    Returns a client to interact with our own buckets
    """
    credentials = get_secret("GOOGLE_APPLICATION_CREDENTIALS_DATA_LOCKER")
    info = json.loads(credentials)
    credentials = service_account.Credentials.from_service_account_info(info)
    client = storage.Client(credentials=credentials)
    return client


def get_file_extension(file_name: str):
    return file_name.split(".")[-1]


def replace_file_extension(file_name: str, file_extension: str):
    return file_name.split(".")[0] + "." + file_extension


def convert_images_to_parquet(images_dir: str, output_dir: str, image_extension: str):
    """
    Convert all files extension to parquet
    """
    if not os.path.isdir(images_dir):
        raise FileNotFoundError("The path '{images_dir}' doesn't exist. Please check!")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    image_ids = [
        file_name
        for file_name in os.listdir(images_dir)
        if get_file_extension(file_name) == image_extension
    ]
    for img_id in image_ids:
        image_path = os.path.join(images_dir, img_id)
        output_path = os.path.join(output_dir, replace_file_extension(img_id, "parquet"))

        if os.path.exists(output_path):  # skip image compression if already converted
            continue

        with Image.open(image_path) as img:
            img_bytes_io = io.BytesIO()
            img.save(
                img_bytes_io, format=image_extension.upper()
            )  # save as bytes in original format
            img_bytes = img_bytes_io.getvalue()

            schema = pa.schema([pa.field("image_bytes", pa.binary())])
            table = pa.Table.from_pydict({"image_bytes": [img_bytes]}, schema=schema)

            pq.write_table(table, output_path)


def upload_files_to_gcs_blob(
    bucket: storage.Client.bucket, local_dir: str, folder_name: str, override: bool = False
):
    """
    Uploading all files from local directory to Google Cloud Storage Blob (ie folder)
    """
    image_ids = os.listdir(local_dir)
    for img_id in tqdm(image_ids, desc="Uploading"):
        img_path = os.path.join(local_dir, img_id)
        blob_path = f"{folder_name}/{img_id}"

        if bucket.blob(blob_path).exists():  # skip if image is already uploaded
            continue

        blob = bucket.blob(folder_name)
        blob.upload_from_filename(img_path)


def run():
    # -- check if file directory path exists
    data_dir = "data/CamVid/"
    output_dir = "data/CamVidParquet"
    bucket_name = "test-camvid"
    project_id = "relax-server"

    if not os.path.isdir(data_dir):
        raise FileNotFoundError("The path '{data_dir}' doesn't exist. Please check!")

    # -- compress image files to parquet files from subdirectories path
    subdirectories = ["train", "trainannot", "val", "valannot", "test", "testannot"]
    for subdir in subdirectories:
        convert_images_to_parquet(
            os.path.join(data_dir, subdir), os.path.join(output_dir, subdir), "png"
        )

    # -- upload files to gcs if not uploaded yet
    # client = storage.Client(project=project_id) # note: ensure that you have exported $GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
    client = get_storage_client()
    bucket = client.bucket(bucket_name)
    for subdir in subdirectories[0:1]:
        upload_files_to_gcs_blob(bucket, os.path.join(output_dir, subdir), subdir)


run()
