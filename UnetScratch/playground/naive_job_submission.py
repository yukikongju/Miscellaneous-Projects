"""
Submitting torch job to run on vertex ai using out-of-the-box solution

Pre-requisites:
1. Creating buckets
    gcloud storage buckets create gs://project-torch-submission \
    --default-storage-class=STANDARD \
    --location=northamerica-northeast1 \
    --uniform-bucket-level-access \
    --public-access-prevention \
    --project=relax-server

2. Enable Vertex AI
    gcloud services enable aiplatform.googleapis.com --project=relax-server

3. Create service account with permission:
    - "aiplatform.customJobs.create"
    - "storage.objects.objectAdmin"
    - "storage.objects.create"

    gcloud auth application-default login

    gcloud config set project relax-server

    # torch-job-submission@relax-server.iam.gserviceaccount.com
    gcloud iam service-accounts create torch-job-submission \
        --display-name="Service account for custom torch training"

    gcloud projects add-iam-policy-binding relax-server \
    --member="user:user@email.com" \
    --role="roles/aiplatform.admin"

    gcloud projects add-iam-policy-binding relax-server \
    --member="serviceAccount:torch-job-submission@relax-server.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

    gcloud projects add-iam-policy-binding relax-server \
    --member="serviceAccount:torch-job-submission@relax-server.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

    gcloud projects get-iam-policy relax-server \
    --flatten="bindings[].members" \
    --filter="bindings.members:YOUR_EMAIL" \
    --format="table(bindings.role)"

Notes:
- credentials used to execute the script is `echo $GOOGLE_APPLICATION_CREDENTIALS`
- To create key from Service Account: IAM & Admin > Service Accounts
- To see jobs: Vertex AI > Training > Custom Jobs
- In order to use GPU, we need to request GPU quotas
"""

import google.auth
from google.cloud import aiplatform
from google.oauth2 import service_account

# using $GOOGLE_APPLICATION_CREDENTIALS
# creds, project = google.auth.default()
# print(creds.service_account_email if hasattr(creds, 'service_account_email') else "User credentials")

aiplatform.init(
    project="relax-server",
    location="northamerica-northeast1",
    staging_bucket="gs://project-torch-submission",
)

credentials = service_account.Credentials.from_service_account_file(
    "/Users/emulie/keys/relax-server-f2a6c1210ba1-torch.json"
)

job = aiplatform.CustomJob.from_local_script(
    project="relax-server",
    credentials=credentials,
    display_name="pytorch-training-job",
    script_path="playground/dummy_train.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-xla.2-4.py310:latest",  # CPU only: remove accelerator
    requirements=["torch==2.1.0", "torchvision", "tensorboard", "tqdm"],
    replica_count=1,
    machine_type="n1-standard-4",
    # for gpu
    # container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest", # GPU only
    # accelerator_type="NVIDIA_TESLA_T4", # google.api_core.exceptions.ResourceExhausted: 429 The following quota metrics exceed quota limits: aiplatform.googleapis.com/custom_model_training_nvidia_t4_gpus
    # accelerator_count=1,
    # staging_bucket="gs://project-torch-submission",
    # service_account="your-sa@relax-server.iam.gserviceaccount.com"
)

job.run(sync=True)
print(job.resource_name)
print(job._dashboard_uri())
