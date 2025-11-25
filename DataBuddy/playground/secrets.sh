# Notes:
# - Ensure that user is properly logged in
# - Ensure that `echo $GOOGLE_APPLICATION_CREDENTIALS` points to service account

gcloud secrets list
gcloud secrets describe OPENAI_API_KEY

gcloud projects add-iam-policy-binding relax-server \
  --member="user:emulie@ipnos.com" \
  --role="roles/secretmanager.secretAccessor"


# https://console.cloud.google.com/security/secret-manager/secret/GOOGLE_APPLICATION_CREDENTIALS_MAIN_BIGQUERY/versions?project=relax-server&supportedpurview=project
gcloud secrets versions access latest --secret=GOOGLE_APPLICATION_CREDENTIALS_MAIN_BIGQUERY --project relax-server


# https://console.cloud.google.com/security/secret-manager/secret/OPENAI_API_KEY/versions?project=relax-server&supportedpurview=project
gcloud secrets versions access latest --secret=OPENAI_API_KEY --project=2531232713



ERROR: (gcloud.secrets.versions.access) PERMISSION_DENIED: Permission 'secretmanager.versions.access' denied for resource 'projects/relax-server/secrets/OPENAI_API_KEY/versions/latest' (or it may not exist). This command is authenticated as emulie@ipnos.com which is the active account specified by the [core/account] property.
