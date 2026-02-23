#!/bin/bash

set -e

# Configuration
GCP_PROJECT="relax-server"
GCP_REGION="us-central1"
AR_REPO="scoping-calculator"
SERVICE_NAME="scoping-calculator-app"
APP_PORT="8051"

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
gcloud run deploy "$SERVICE_NAME" \
  --port="$APP_PORT" \
  --image="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME" \
  --allow-unauthenticated \
  --region="$GCP_REGION" \
  --platform=managed \
  --project="$GCP_PROJECT" \
  --set-env-vars="GCP_PROJECT=$GCP_PROJECT,GCP_REGION=$GCP_REGION"

# fix Error: Forbidden - Your client does not have permission to get URL
gcloud run services add-iam-policy-binding scoping-calculator-app \
  --project=relax-server \
  --region=us-central1 \
  --member="allUsers" \
  --role="roles/run.invoker"

gcloud run services get-iam-policy scoping-calculator-app \
  --project=relax-server \
  --region=us-central1
