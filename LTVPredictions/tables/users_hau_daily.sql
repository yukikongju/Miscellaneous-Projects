create or replace table `relax-melodies-android.late_conversions.users_hau_daily` (
    event_date DATE,
    user_id STRING,
    user_pseudo_id STRING,
    platform STRING,
    country STRING,
    traffic_source_name STRING,
    traffic_source_medium STRING,
    traffic_source STRING,
    hau STRING
)
PARTITION BY
    event_date
CLUSTER BY
    user_id, user_pseudo_id, platform, country;
