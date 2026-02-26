#!/bin/bash

# https://console.cloud.google.com/run/detail/northamerica-northeast1/scoping-calculator-app/networking?project=relax-server

set -e

# Configuration
GCP_PROJECT="relax-server"
# GCP_REGION="us-central1"
GCP_REGION="northamerica-northeast1"
# AR_REPO="scoping-calculator"
AR_REPO="gcf-artifacts"
SERVICE_NAME="scoping-calculator-app"
APP_PORT="8501"

# Create Artifact Registry repository (check if not exists!)
if ! gcloud artifacts repositories describe "$AR_REPO" \
  --location="$GCP_REGION" \
  --project="$GCP_PROJECT" &>/dev/null; then
  echo "Creating Artifact Registry repository: $AR_REPO"
  gcloud artifacts repositories create "$AR_REPO" \
    --location="$GCP_REGION" \
    --repository-format=Docker \
    --project="$GCP_PROJECT"
else
  echo "Artifact Registry repository '$AR_REPO' already exists, skipping."
fi

# Build and submit container image
gcloud builds submit \
  --tag "$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME"

# Deploy to Cloud Run
gcloud beta run deploy "$SERVICE_NAME" \
  --image="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME" \
  --region="$GCP_REGION" \
  --project="$GCP_PROJECT" \
  --platform=managed \
  --allow-unauthenticated \
  --port="$APP_PORT" \
  # --no-allow-unauthenticated \
  # --iap
  # --set-env-vars="GCP_PROJECT=$GCP_PROJECT,GCP_REGION=$GCP_REGION" \

  # Note: After deployment, you must grant the Cloud Run Invoker role to the IAP service account (service-[PROJECT-NUMBER]@gcp-sa-iap.iam.gserviceaccount.com) and assign the IAP-secured Web App User role to the principals (users/groups) who should have access


# fix Error: Forbidden - Your client does not have permission to get URL
gcloud run services add-iam-policy-binding scoping-calculator-app \
  --project=relax-server \
  --region=northamerica-northeast1 \
  --member="allUsers" \
  --role="roles/run.invoker"

# gcloud run services get-iam-policy scoping-calculator-app \
#   --project=relax-server \
#   --region=northamerica-northeast1

# # Error: Page not found The requested URL was not found on this server.
# gcloud run services describe scoping-calculator-app \
#     --project=relax-server \
#     --region=northamerica-northeast1 \
#     --format="value(status.url)"

# gcloud run services describe scoping-calculator-app \
#     --project=relax-server \
#     --region=northamerica-northeast1 \
#     --format="yaml(status.url,status.traffic,status.latestReadyRevisionName)"

# --- works!
# gloud run deploy --source .
