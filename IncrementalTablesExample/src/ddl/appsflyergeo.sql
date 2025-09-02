create or replace table `relax-melodies-android.test_incremental.appsflyergeo` (
    date DATE,
    media_source_pid STRING,
    platform STRING,
    country STRING,
    cost FLOAT64,
    impressions FLOAT64,
    clicks FLOAT64,
    installs FLOAT64,
    conversions FLOAT64,
    paid FLOAT64,
    revenue FLOAT64,
    loaded_timestamp TIMESTAMP
)
partition by date
cluster by media_source_pid, platform, country;
