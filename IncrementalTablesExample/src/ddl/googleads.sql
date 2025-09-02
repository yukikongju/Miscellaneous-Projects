create or replace table `relax-melodies-android.test_incremental.googleads` (
    segments_date DATE,
    platform STRING,
    country STRING,
    metrics_cost_micros FLOAT64,
    metrics_impressions FLOAT64,
    metrics_clicks FLOAT64,
    metrics_installs FLOAT64,
    metrics_conversions FLOAT64,
    metrics_paid FLOAT64,
    metrics_revenue FLOAT64,
    loaded_timestamp TIMESTAMP
)
partition by segments_date
cluster by platform, country;
