create or replace table `relax-melodies-android.late_conversions.latest_renewal_rates_cohorted` (
    year_month date,
    network STRING,
    platform STRING,
    country_code STRING,
    renewal_bucket STRING,
    num_paid INT,
    num_renewals INT,
    paid_proceeds FLOAT64,
    renewal_proceeds FLOAT64,
    renewal_percentage FLOAT64,
    loaded_timestamp timestamp
)
partition by
    year_month
cluster by
    network, platform, country_code;
