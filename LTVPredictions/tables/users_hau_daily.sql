create table `relax-melodies-android.late_conversions.users_hau_daily` (
    event_date DATE,
    user_id STRING,
    user_pseudo_id STRING,
    platform STRING,
    country STRING,
    hau STRING
)
PARTITION BY
    event_date
CLUSTER BY
    user_id, user_pseudo_id, platform, country;
