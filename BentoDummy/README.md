# Bento Dummy

Containerizing model with BentoML on Iris Dataset

**Steps**

1. Train model: `python training/train.py`
2. Save into Bento store: `python serving/save.py`
3. Build bento: `bentoml build`
4. Containerize: `bentoml containerize guided_content_service:latest`
5. Push to GCP: `docker push gcr.io/PROJECT/IMAGE`
6. Deploy: `gcloud run deploy `
