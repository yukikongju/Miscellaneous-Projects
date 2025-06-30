# Partition Events Table

**The Problem**

Currently, the events table is sharded on "event_date", which is a STRING.
This means that for a single day, we have to perform a linear scan through
all the data, regardless of the predicate clause. This is an issue because
we get around 100GB of data everyday. This means that querying a month of data
costs around 15$. The T2P query runs about 3 times a day.

We can verify the costs as follows:

```{sh}
# Get the location project_id:dataset_id
bq show --format=prettyjson relax-melodies-android:analytics_151587246
```

```{sql}
--- to get all queries executed within last 7 days
SELECT
  job_id,
  creation_time,
  start_time,
  end_time,
  user_email,
  statement_type,
  query,
  total_bytes_processed,
  total_bytes_billed
FROM
  `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE
  creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
  AND state = 'DONE'
  AND job_type = 'QUERY'
ORDER BY
  creation_time DESC
```

```{sql}
# --- to get costs from pipeline
SELECT
  extract(date from creation_time) as creation_date,
  user_email,
  statement_type,
  sum(total_bytes_processed) as total_bytes_processed,
  sum(total_bytes_billed) as total_bytes_billed,
  sum(total_bytes_billed) / 1e12 * 5 as total_cost_usd_billed,
FROM
  `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE
  creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
  AND state = 'DONE'
  AND job_type = 'QUERY'
  AND user_email = 'bigquery-admin@relax-melodies-android.iam.gserviceaccount.com'
GROUP BY
  extract(date from creation_time), user_email, statement_type
order by creation_date desc, statement_type asc
```

It costs about 25$ everyday as of Jun 30, 2025. This costs is only for the
T2P. It doesn't account for ad-hoc analyses or model retraining

**Use Cases**

Our current use cases are the following:
- Trial2Paid Model: we want to query hau/utm/trial/paid/refund/renewal at
  the user level
- day 0: we want to query user event on a daily level
- Late Revenue:

Later:
- Recommender System: use user level events
- Churn Prediction

**Goal**

- [ ] Repartition the `analytics.events` table
- [ ] Optimize Query used for T2P

## Problem 1 - Repartition `analytics.events` table (INC)

- partition: event_date_partition (timestamp)
- cluster: event_name

Ideally, we want to cluster on 'user_id' and 'user_pseudo_id', however, those
are sometimes null. I believe this happens in the following cases:
- on web:
    * if a user has an account, they won't have a 'user_pseudo_id'
    * if a user doesn't have an account, their 'user_id' and 'user_pseudo_id' in null
- on android:
    * 'user_pseudo_id' missing
- on ios:
    * 'user_pseudo_id' is always there

## Problem 2 - Optimize Query used for T2P (TBD)
