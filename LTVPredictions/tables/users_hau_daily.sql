create or replace table `relax-melodies-android.late_conversions.users_hau_daily` (
    event_date DATE,
    event_timestamp INT,
    user_id STRING,
    user_pseudo_id STRING,
    platform STRING,
    country STRING,
    traffic_source_name STRING,
    traffic_source_medium STRING,
    traffic_source STRING,
    hau STRING,
    load_timestamp TIMESTAMP
)
PARTITION BY
    event_date
CLUSTER BY
    user_id, user_pseudo_id, platform, country;
