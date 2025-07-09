create or replace table `relax-melodies-android.late_conversions.users_utm_daily` (
    event_date DATE,
    event_timestamp INT,
    user_id STRING,
    user_pseudo_id STRING,
    platform STRING,
    country STRING,
    traffic_source_name STRING,
    traffic_source_medium STRING,
    traffic_source STRING,
    campaign STRING,
    utm_source STRING,
    page_title STRING,
    page_location STRING,
    page_referrer STRING,
    content STRING,
)
PARTITION BY
    event_date
CLUSTER BY
    user_id, user_pseudo_id, platform, country;
