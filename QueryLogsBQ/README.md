# BigQuery Query Logs

The goal of this project is to monitor bigquery query execution with
real-time processing.

We want to:
- Identify keys changes in the BigQuery tables:
    * when production tables and views get updated => table creation, deletion and updates in `ua_extract_prod`, `ua_transform_prod`, `ua_dashboard_prod`
    * query cost and time for every user in the team
- Send slack notification

Tech:
- Pub/Sub
- Log Sink

## Resources

- [Route Log Entries](https://cloud.google.com/logging/docs/routing/overview)
- [Sink Creation](https://cloud.google.com/logging/docs/central-log-storage)
- [Log Explorer Overview](https://cloud.google.com/logging/docs/view/logs-explorer-interface)
- [stack overflow] [GCP - How to search for wildcards](https://stackoverflow.com/questions/71922754/google-cloud-platform-logging-how-to-search-wildcard-strings-in-all-logs)
- [Roles and permissions reference](https://cloud.google.com/iam/docs/roles-permissions/pubsub)
