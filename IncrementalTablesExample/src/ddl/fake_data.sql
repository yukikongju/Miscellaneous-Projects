create or replace table `relax-melodies-android.test_incremental.fake_data` (
    day DATE,
    platform STRING,
    network STRING,
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
partition by day
cluster by network, platform, country;
