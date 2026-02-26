# Saturation Curve

```
> uv run marimo edit
> uv run streamlit run src/Pages.py
```



Notes:
- running streamlit from uv doesn't work because Pages.py and pages/ need
  to be in root (from chat) => not true, need to upgrade streamlit version

Killing processes: `lsof -i :8501; kill -9 61372`


Build and run:

  docker build -t saturation-utils .
  %% docker run -p 8501:8501 saturation-utils
  docker run -p 8501:8501 -v /Users/emulie/keys/relax-server-06b2a61c0080.json:/tmp/keys/keyfile.json:ro -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/keyfile.json saturation-utils

  If you want BigQuery auth via mounted key file, use:

  docker run --rm \
    -v /local/path/key.json:/app/key.json:ro \
    -e GOOGLE_APPLICATION_CREDENTIALS=/app/key.json \
    saturation-utils

 gcloud run deploy SERVICE_NAME \
    --image IMAGE_URL \
    --region us-central1 \
    --service-account my-sa@PROJECT_ID.iam.gserviceaccount.com

  gcloud run services add-iam-policy-binding "$SERVICE" \
    --region "$REGION" \
    --member="group:$GROUP_EMAIL" \
    --role="roles/run.invoker"

# Reference Docs

- [Deploy streamlit app on GCP](https://medium.com/bitstrapped/step-by-step-guide-deploying-streamlit-apps-on-google-cloud-platform-gcp-96fca6a4f331)
- [gcloud run deploy](https://docs.cloud.google.com/sdk/gcloud/reference/run/deploy#--clear-secrets)
- [adding iap - authentication](https://www.youtube.com/watch?v=ayTGOuCaxuc)
