create or replace table `relax-melodies-android.late_conversions.users_network_attribution` (
    user_id STRING,
    user_pseudo_id STRING,
    platform STRING,
    country_code STRING,
    network_attribution STRING,
)
cluster by user_id, user_pseudo_id, platform, country_code;
