--- cost: 84.9KB
DECLARE renewal_threshold FLOAT64 DEFAULT 0.8;
DECLARE renewal_rate_depreciation_rate float64 default 0.6;

drop table `relax-melodies-android.late_conversions.renewal_proceeds_ma_imputed`;
create or replace table `relax-melodies-android.late_conversions.renewal_proceeds_ma_imputed`
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
      renewal_proceeds
    FROM
      `relax-melodies-android.late_conversions.latest_renewal_rates`
  )
  PIVOT (
    AVG(renewal_proceeds)
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
), pivot_table_default as (
  select
    platform, network, country_code,
    `1-Year`,
    coalesce(`2-Years`, `1-Year` * renewal_rate_depreciation_rate) as `2-Years`,
    coalesce(`3-Years`,
      case
        when `2-Years` is not null then `2-Years` * renewal_rate_depreciation_rate
        else `1-Year` * renewal_rate_depreciation_rate * renewal_rate_depreciation_rate
      end
    ) as `3-Years`
  from pivot_table_imputed
)

-- Use this to select either raw or imputed version
SELECT * FROM pivot_table_default
