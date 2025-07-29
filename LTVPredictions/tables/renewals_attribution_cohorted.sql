create or replace `relax-melodies-android.late_conversions.renewals_attribution_cohorted` (
    paid_year_month TIMESTAMP,
    network STRING,
    platform STRING,
    country_code STRING,
    `1-Year` FLOAT64,
    `2-Year` FLOAT64,
    `3-Year` FLOAT64,
)
partition by paid_year_month
cluster by network, platform, country_code
