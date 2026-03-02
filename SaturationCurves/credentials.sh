#!/bin/sh

# docs: https://docs.cloud.google.com/run/docs/securing/identity-aware-proxy-cloud-run
# - https://docs.cloud.google.com/run/docs/securing/identity-aware-proxy-cloud-run
# - https://cloud.google.com/iap/docs/enabling-cloud-run

# Notes
# - This is running cloud native IAP (compares to external application load balancer IAP)

PROJECT_ID="relax-server"
REGION="northamerica-northeast1"
SERVICE_NAME="saturationcurves"
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')

# 0) verify roles `run.admin`, `iap-admin`, `iam.serviceAccountUser`

# 1) Enable IAP
gcloud beta run services update "$SERVICE_NAME" \
--project "$PROJECT_ID" \
--region "$REGION" \
--iap

# 2) Let IAP invoke Cloud Run
gcloud run services add-iam-policy-binding "$SERVICE_NAME" \
    --project "$PROJECT_ID" \
    --region "$REGION" \
    --member="serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-iap.iam.gserviceaccount.com" \
    --role="roles/run.invoker"

# 3) Grant app access to specific users/groups
# Note:   (Use group:team@yourcompany.com for groups; user:name@company.com)
gcloud beta iap web add-iam-policy-binding \
    --project "$PROJECT_ID" \
    --region "$REGION" \
    --resource-type=cloud-run \
    --service="$SERVICE_NAME" \
    --member="domain:ipnos.com" \
    --role="roles/iap.httpsResourceAccessor"

# 4) Verify
gcloud beta run services describe "$SERVICE_NAME" \
    --project "$PROJECT_ID" \
    --region "$REGION"


# ============================ DEPRECATED ============================


# Before deploying, check these things:

# **1. Your user account (`emulie@ipnos.com`) needs:**
# Check your roles on the project
# gcloud projects get-iam-policy relax-melodies-android \
#     --flatten="bindings[].members" \
#     --filter="bindings.members:emulie@ipnos.com"
# You need at least `roles/run.admin` and `roles/iam.serviceAccountUser`.

# **2. Your service account (`bigquery-admin@...`) needs:**
# Check its roles
# gcloud projects get-iam-policy relax-melodies-android \
#     --flatten="bindings[].members" \
#     --filter="bindings.members:bigquery-admin@relax-melodies-android.iam.gserviceaccount.com"
# It needs `roles/bigquery.admin` (or at least `roles/bigquery.dataEditor` + `roles/bigquery.jobUser`).

# **3. Verify the service account exists:**
# gcloud iam service-accounts describe \
#     bigquery-admin@relax-melodies-android.iam.gserviceaccount.com \
#     --project=relax-melodies-android

# **4. Verify your active account and project:**
# gcloud config list
# Make sure `account` and `project` are pointing to the right things before running any deploy.
