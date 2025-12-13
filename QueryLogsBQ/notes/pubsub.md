PROJECT_ID=relax-melodies-android
TOPIC_ID=test_sink
SUBSCRIPTION_ID=test_sub_sink
DOCKER_IMAGE_ID=test_slack_image
SERVICE_ID=test_slack_service

Example with Docker:

```{sh}
### Build docker image
docker build -t gcr.io/$(PROJECT_ID)/$(DOCKER_IMAGE_ID) .

### deploy docker image to Artifacts Registry
docker push -t gcr.io/$(PROJECT_ID)/$(DOCKER_IMAGE_ID):latest

### deploy cloud run service with deployed image
gcloud run deploy $(SERVICE_ID) \
    --image gcr.io/$(PROJECT_ID)/$(DOCKER_IMAGE_ID) \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
    --set-env-vars SLACK_TOKEN=<..>
```


Example without Docker:
```{sh}
gcloud pubsub topics create test_slack

gcloud pubsub subscriptions create test_slack_sub \
  --topic=test_slack

gcloud pubsub topics publish test_slack --message="hello" \
  --attribute="origin=gcloud-sample,username=gcp,eventTime='2021-01-01T12:00:00Z'"

```

```{sh}
gcloud pubsub topics create $(TOPIC_ID)

gcloud pubsub topics publish $(TOPIC_ID) \
  --message=MESSAGE_DATA \
  [--attribute=KEY="VALUE",...]

gcloud pubsub topics publish $(TOPIC_ID) --message="hello" \
  --attribute="origin=gcloud-sample,username=gcp,eventTime='2021-01-01T12:00:00Z'"

--- create pull subscription
gcloud pubsub subscriptions create $(SUBSCRIPTION_ID) \
--topic=TOPIC_ID \
--push-endpoint=PUSH_ENDPOINT

--- create push subscription
gcloud pubsub subscriptions create $(SUBSCRIPTION_ID) \
--topic=TOPIC_ID \

--- EXAMPLES ---

{
  "data": string,
  "attributes": {
    string: string,
    ...
  },
  "messageId": string,
  "publishTime": string,
  "orderingKey": string
}


```

## Docs

- [GCP - Configure Export Permission](https://cloud.google.com/logging/docs/export/configure_export_v2#dest-auth)
