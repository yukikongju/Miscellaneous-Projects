# TODOs

---

1. We publish a message to a topic using `gcloud pubsub topics publish`
2. When the message is published, it gets rerouted to Docker,
   which print the message received in the slack channel

To figure out:
    * How to define environment variables in dockerfile / in Cloud run
    * How to deploy dockerfile in GCP Artifacts
    * How to reroute pub/sub to docker


---

- [ ] Create Cloud Function to send slack message with image

- [ ] Get Permissions + Create Sink
- [ ] Create Pub/sub messages that route message to Docker

- [ ] Dockerfile that execute code when it receives messages
    - [ ] Modify Docker code such that it send slack message
    - [ ] Read variable from GCP


- [gcp environment variables](https://cloud.google.com/run/docs/configuring/services/environment-variables#gcloud)

```
gcloud beta run deploy SERVICE --image=IMAGE_URL --env-vars-file=ENV_FILE_PATH

```
