# Saturation Curve

```
> uv run marimo edit
> uv run streamlit run src/Pages.py
```

Notes:
- running streamlit from uv doesn't work because Pages.py and pages/ need
  to be in root (from chat) => not true, need to upgrade streamlit version

Build and run:

  docker build -t saturation-utils .
  docker run -p 8501:8501 saturation-utils
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

# Reference Docs

- [Deploy streamlit app on GCP](https://medium.com/bitstrapped/step-by-step-guide-deploying-streamlit-apps-on-google-cloud-platform-gcp-96fca6a4f331)
