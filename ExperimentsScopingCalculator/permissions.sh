#!/bin/sh

# docs: https://docs.cloud.google.com/run/docs/securing/identity-aware-proxy-cloud-run
# - https://docs.cloud.google.com/run/docs/securing/identity-aware-proxy-cloud-run
# - https://cloud.google.com/iap/docs/enabling-cloud-run

# Notes
# - This is running cloud native IAP (compares to external application load balancer IAP)

PROJECT_ID="relax-server"
REGION="northamerica-northeast1"
SERVICE_NAME="experimentsscopingcalculator"
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')

# 0) verify roles `run.admin`, `iap-admin`, `iam.serviceAccountUser`

# 1) Enable IAP
# gcloud beta run services update "$SERVICE_NAME" \
# --project "$PROJECT_ID" \
# --region "$REGION" \
# --iap

# 2) Let IAP invoke Cloud Run
# gcloud run services add-iam-policy-binding "$SERVICE_NAME" \
#     --project "$PROJECT_ID" \
#     --region "$REGION" \
#     --member="serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-iap.iam.gserviceaccount.com" \
#     --role="roles/run.invoker"

# 3) Grant app access to specific users/groups
# Note:   (Use group:team@yourcompany.com for groups.)
# gcloud beta iap web add-iam-policy-binding \
#     --project "$PROJECT_ID" \
#     --region "$REGION" \
#     --resource-type=cloud-run \
#     --service="$SERVICE_NAME" \
#     --member="user:emulie@ipnos.com" \
#     --role="roles/iap.httpsResourceAccessor"

gcloud beta iap web add-iam-policy-binding \
    --project "$PROJECT_ID" \
    --region "$REGION" \
    --resource-type=cloud-run \
    --service="$SERVICE_NAME" \
    --member="domain:ipnos.com" \
    --role="roles/iap.httpsResourceAccessor"

gcloud beta iap web add-iam-policy-binding \
    --project "$PROJECT_ID" \
    --region "$REGION" \
    --resource-type=cloud-run \
    --service="$SERVICE_NAME" \
    --member="domain:bettersleep.com" \
    --role="roles/iap.httpsResourceAccessor"


# 4) Verify
# gcloud beta run services describe "$SERVICE_NAME" \
#     --project "$PROJECT_ID" \
#     --region "$REGION"
