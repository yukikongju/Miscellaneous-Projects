TOPIC_ID=test_sink
SUBSCRIPTION_ID=test_sub_sink

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

```{sh}
gcloud pubsub topics create test_sink

gcloud pubsub subscriptions create test_sub_sink \
  --topic=test_sink

gcloud pubsub topics publish test_sink --message="hello" \
  --attribute="origin=gcloud-sample,username=gcp,eventTime='2021-01-01T12:00:00Z'"

```
