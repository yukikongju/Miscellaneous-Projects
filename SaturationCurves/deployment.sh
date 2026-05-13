#!/bin/sh

# PROJECT_ID="relax-server"
PROJECT_ID="relax-melodies-android"
REGION="northamerica-northeast1"
SERVICE="saturation-utils"
REPO="ds-containers"
IMAGE="saturation-utils"
# TAG="v1"
# GROUP_EMAIL="ua-team@yourcompany.com"

gcloud config set project "$PROJECT_ID"
# gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com iam.googleapis.com

# 2) Create Artifact Registry repo (one-time)
gcloud artifacts repositories create "$REPO" \
    --repository-format=docker \
    --location="$REGION" \
    --description="Container repo for Cloud Run"

# 3) Build and push image
# gcloud builds submit --tag "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:$TAG"
gcloud builds submit --tag "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE"

# 4) Deploy Cloud Run (requires auth)
gcloud run deploy "$SERVICE" \
    --image "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:$TAG" \
    --region "$REGION" \
    --platform managed \
    --no-allow-unauthenticated

# 5) Allow only internal users (recommended: Google Group)
gcloud run services add-iam-policy-binding "$SERVICE" \
    --region "$REGION" \
    --member="group:$GROUP_EMAIL" \
    --role="roles/run.invoker"

# Optional (broader): allow whole company domain
gcloud run services add-iam-policy-binding "$SERVICE" \
    --region "$REGION" \
    --member="domain:yourcompany.com" \
    --role="roles/run.invoker"

# Test access:

# SERVICE_URL=$(gcloud run services describe "$SERVICE" --region
# "$REGION" --format='value(status.url)')
# TOKEN=$(gcloud auth print-identity-token --audiences="$SERVICE_URL")
# curl -H "Authorization: Bearer $TOKEN" "$SERVICE_URL"

# ERROR: (gcloud.run.deploy) PERMISSION_DENIED: Permission 'iam.serviceaccounts.actAs' denied on service account bigquery-admin@relax-melodies-android.iam.gserviceaccount.com (or it may not exist). This command is authenticated as emulie@ipnos.com which is the
#  giving myself "actAs"

gcloud projects add-iam-policy-binding relax-server \
    --member="user:emulie@ipnos.com" \
    --role="roles/iam.serviceAccountUser" \
    --project=relax-melodies-android

# ERROR: (gcloud.iam.service-accounts.add-iam-policy-binding) NOT_FOUND: Service account projects/relax-server/serviceAccounts/bigquery-admin@relax-melodies-android.iam.gserviceaccount.com does not exist. This command is authenticated as emulie@ipnos.com which is the active account specified by the [core/account] property.

# Note: workaround - doesn't work when get_bigquery_client() reads get_secret()
# bigquery.Client() with no arguments automatically uses Application Default Credentials (ADC), which on Cloud Run means it will use the service account you specified with --service-account
# gcloud run deploy saturationcurves \
#     --source . \
#     --region northamerica-northeast1 \
#     --service-account bigquery-admin@relax-melodies-android.iam.gserviceaccount.com \
#     --project=relax-melodies-android \
#     --allow-unauthenticated

# clear env variables and secreats
gcloud run services update saturationcurves \
    --region northamerica-northeast1 \
    --clear-secrets \
    --clear-env-vars

# streamlit works, but Access Denied: Table relax-melodies-android:ua_dashboard_prod.final_table: User does not have permission to query table relax-melodies-android:ua_dashboard_prod.final_table, or perhaps it does not exist. Location: US Job ID: b2d76419-9cf2-400f-99d0-d805f7f7d2b2
gcloud run deploy saturationcurves \
    --source . \
    --region northamerica-northeast1 \
    --allow-unauthenticated
    # --no-allow-unauthenticated
#
# gcloud run deploy saturationcurves \
#     --source . \
#     --region northamerica-northeast1 \
#     --service-account bigquery-admin@relax-melodies-android.iam.gserviceaccount.com \
#     --platform=managed \
#     --allow-unauthenticated

# works when we give the proper credentials
gcloud run deploy saturationcurves \
    --source . \
    --region northamerica-northeast1 \
    --platform=managed \
    --project relax-server \
    --allow-unauthenticated
    # --service-account bigquery-admin@relax-melodies-android.iam.gserviceaccount.com \
    # --project relax-melodies-android \

gcloud run services add-iam-policy-binding saturation-utils \
    --region northamerica-northeast1 \
    --member="user:emulie@ipnos.com" \
    --role="roles/run.invoker"


# NOTE: need to fix
# ERROR: (gcloud.run.deploy) PERMISSION_DENIED: Permission 'iam.serviceaccounts.actAs' denied on service account bigquery-admin@relax-melodies-android.iam.gserviceaccount.com (or it may not exist). This command is authenticated as emulie@ipnos.com which is the active account specified by the [core/account] property.


# gcloud secrets versions list GOOGLE_APPLICATION_CREDENTIALS_MAIN_BIGQUERY

# gcloud run deploy saturationcurves \
#     --source . \
#     --region northamerica-northeast1

gcloud run deploy --source .

# secret manager: Security > Secret Manager


# -----
# 1. Create the secret
# gcloud secrets create my-keyfile \
#     --data-file=keyfile.json \
#     --project=relax-melodies-android

# 2. Deploy referencing the secret
# gcloud run deploy saturationcurves \
#     --source . \
#     --region northamerica-northeast1 \
#     --allow-unauthenticated \
#     --set-secrets=GOOGLE_APPLICATION_CREDENTIALS=GOOGLE_APPLICATION_CREDENTIALS_MAIN_BIGQUERY:latest

# -----
# ERROR: (gcloud.run.deploy) spec.template.spec.containers[0].env[0].value_from.secret_key_ref.name: Permission denied on secret: projects/2531232713/secrets/GOOGLE_APPLICATION_CREDENTIALS_MAIN_BIGQUERY/versions/latest for Revision service account 2531232713-compute@developer.gserviceaccount.com. The service account used must be granted the 'Secret Manager Secret Accessor' role (roles/secretmanager.secretAccessor) at the secret, project or higher level.

gcloud secrets add-iam-policy-binding GOOGLE_APPLICATION_CREDENTIALS_MAIN_BIGQUERY \

# --------------------- ADDING ENV VARIABLES FROM SECRET MANAGER ----------------

# Context: Secret Manager Variables live in project `relax-server`, but the
# service account used is `relax-melodies-android`. To reference secret manager
# variable, we have 2 options:
# - Option 1: Copy key from `relax-server` secret manager to `relax-melodies-android` => pros: avoid cross-project reference; cons: variable is copied
# - Option 2: Cross-reference between project by giving access to
# We will go with Option 1, as Option 2 does not work

## 1. Reading Secret Variable from Secret Manager

# did not work
gcloud secrets versions access latest \
  --secret="APPSFLYER_TOKEN_API" \
  --project="relax-server" \
  --impersonate-service-account=secret-access@relax-server.iam.gserviceaccount.com

# worked!!
GOOGLE_APPLICATION_CREDENTIALS=/Users/emulie/keys/relax-server-06b2a61c0080.json \
gcloud secrets versions access latest \
  --secret="APPSFLYER_API_TOKEN" \
  --project="relax-server"

## 2. Copy secret key from `relax-server` to `relax-melodies-android` (In GCP: Security > Secret Manager)

## 2.1. Give access to service account
gcloud secrets add-iam-policy-binding APPSFLYER_API_TOKEN \
  --project="relax-server" \
  --member="serviceAccount:2531232713-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"


## 3. Deploy with secret manager
gcloud run deploy --source . \
  --set-secrets="APPSFLYER_API_TOKEN=APPSFLYER_API_TOKEN:latest"

## DEPRECATED STEPS

# Give access to secret manager for
gcloud secrets add-iam-policy-binding APPSFLYER_API_TOKEN \
  --project="relax-server" \
  --member="serviceAccount:secret-access@relax-server.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud run deploy --source . \
  --set-secrets="APPSFLYER_API_TOKEN=projects/relax-server/secrets/APPSFLYER_API_TOKEN:latest"



 # ERROR: (gcloud.run.deploy) 'projects/relax-server/secrets/APPSFLYER_API_TOKEN' is not a valid secret name.

 # ERROR: (gcloud.run.deploy) 'projects/relax-server/secrets/APPSFLYER_API_TOKEN' is not a valid secret name.


 # ERROR: (gcloud.run.deploy) spec.template.spec.containers[0].env[0].value_from.secret_key_ref.name: Permission denied on secret: projects/2531232713/secrets/APPSFLYER_API_TOKEN/versions/latest for Revision service account 2531232713-compute@developer.gserviceaccount.com. The service account used must be granted the 'Secret Manager Secret Accessor' role (roles/secretmanager.secre
