create or replace table `relax-melodies-android.late_conversions.monthly_renewal_proceeds` (
    year_month date,
    platform STRING,
    network STRING,
    country_code STRING,
    `1-Year` FLOAT64,
    `2-Years` FLOAT64,
    `3-Years` FLOAT64,
    loaded_timestamp timestamp,
)
partition by date_trunc(year_month, MONTH)
cluster by platform, network, country_code
