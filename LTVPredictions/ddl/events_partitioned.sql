CREATE OR REPLACE TABLE
  `relax-melodies-android.sandbox.analytics_events_20250327_20250610_partitioned`
PARTITION BY
  DATE(event_date_partition)
CLUSTER BY
  event_date_partition, event_name, user_id, user_pseudo_id
  AS (
  SELECT
    *,
    TIMESTAMP_TRUNC(TIMESTAMP_MICROS(event_timestamp), DAY) AS event_date_partition
  FROM
    `relax-melodies-android.analytics_151587246.events_*`
  WHERE
    _table_suffix >= '20250327');
