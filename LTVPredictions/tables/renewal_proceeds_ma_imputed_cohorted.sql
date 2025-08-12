create or replace table `relax-melodies-android.late_conversions.renewal_proceeds_ma_imputed_cohorted` (
    year_month DATE,
    network STRING,
    platform STRING,
    country_code STRING,
    `1-Year` FLOAT64,
    `2-Years` FLOAT64,
    `3-Years` FLOAT64,
    loaded_timestamp timestamp
)
partition by year_month
cluster by network, platform, country_code;
