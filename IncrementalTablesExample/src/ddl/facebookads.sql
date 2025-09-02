create or replace table `relax-melodies-android.test_incremental.facebookads` (
    day DATE,
    platform STRING,
    country STRING,
    spend FLOAT64,
    impressions FLOAT64,
    clicks FLOAT64,
    mobile_app_installs FLOAT64,
    fb_mobile_conversions FLOAT64,
    fb_mobile_app_purchase FLOAT64,
    fb_mobile_revenue FLOAT64,
    loaded_timestamp TIMESTAMP
)
partition by day
cluster by platform, country;
