
Aug 21, 2025
- [Pub/sub tutorial](https://cloud.google.com/run/docs/tutorials/pubsub)
    * `roles/cloudbuild.builds.editor`
    * `roles/run.admin`
    * `roles/iam.serviceAccountCreator`
    * `roles/resourcemanager.projectIamAdmin`
    * `roles/pubsub.editor`
    * `roles/iam.serviceAccountUser`
    * `roles/serviceusage.serviceUsageConsumer`
    * `roles/storage.admin`


Aug 20, 2025
```{}
gcloud projects add-iam-policy-binding relax-melodies-android \
    --member="user:emulie@ipnos.com" \
    --role="roles/logging.logWriter"

gcloud pubsub topics add-iam-policy-binding test_sink \
  --member="serviceAccount:service-PROJECT_NUMBER@gcp-sa-logging.iam.gserviceaccount.com" \
  --role="roles/pubsub.publisher"

gcloud logging sinks list


```
