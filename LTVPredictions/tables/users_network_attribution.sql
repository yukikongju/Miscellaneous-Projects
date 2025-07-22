create or replace table `relax-melodies-android.late_conversions.users_network_attribution` (
    user_id STRING,
    user_pseudo_id STRING,
    platform STRING,
    country_name STRING,
    country_code STRING, -- ALWAYS NULL
    old_hau STRING,
    old_utm_source STRING,
    hau STRING,
    utm_source STRING,
    traffic_source STRING,
    traffic_source_name STRING,
    campaign STRING,
    network_attribution STRING,
    hau_timestamp INT64,
    utm_timestamp INT64,
    load_timestamp TIMESTAMP
)
cluster by user_id, user_pseudo_id, country_code, hau_timestamp;

--- no need to cluster on "platform" because it has low cardinality
