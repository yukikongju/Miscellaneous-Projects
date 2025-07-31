--- cost: 82.9KB
DECLARE renewal_threshold FLOAT64 DEFAULT 0.8;

drop table `relax-melodies-android.late_conversions.renewal_rates_ma_imputed`;
create or replace table `relax-melodies-android.late_conversions.renewal_rates_ma_imputed`
cluster by platform, network, country_code
as
WITH pivot_table AS (
  SELECT
    platform,
    network,
    country_code,
    `1-Year`,
    `2-Years`,
    `3-Years`
  FROM (
    SELECT
      platform,
      network,
      country_code,
      renewal_bucket,
      renewal_percentage
    FROM
      `relax-melodies-android.late_conversions.latest_renewal_rates`
  )
  PIVOT (
    AVG(renewal_percentage)
    FOR renewal_bucket IN ('1-Year', '2-Years', '3-Years')
  )
),
pivot_table_imputed AS (
  SELECT
    platform,
    network,
    country_code,
    -- Impute if null or above threshold
    CASE
      WHEN `1-Year` IS NULL OR `1-Year` > renewal_threshold
        THEN AVG(`1-Year`) OVER (PARTITION BY platform, country_code)
      ELSE `1-Year`
    END AS `1-Year`,
    CASE
      WHEN `2-Years` IS NULL OR `2-Years` > renewal_threshold
        THEN AVG(`2-Years`) OVER (PARTITION BY platform, country_code)
      ELSE `2-Years`
    END AS `2-Years`,
    CASE
      WHEN `3-Years` IS NULL OR `3-Years` > renewal_threshold
        THEN AVG(`3-Years`) OVER (PARTITION BY platform, country_code)
      ELSE `3-Years`
    END AS `3-Years`
  FROM pivot_table
)

-- Use this to select either raw or imputed version
SELECT * FROM pivot_table_imputed
