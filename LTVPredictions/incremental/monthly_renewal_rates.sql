-- cost: 92KB
DECLARE renewal_threshold FLOAT64 DEFAULT 0.8;
DECLARE renewal_rate_depreciation_rate float64 default 0.6;
DECLARE default_renewal_rate float64 default 0.3;

-- `monthly_renewal_rates_cohorted` => compute available renewal rates monthly
-- `monthly_renewal_rates_imputed` => view that impute renewal rates for missing with average and default
--

-- TODO: use network renewal rate to impute instead of defaulting to default rate
-- TODO: impute missing values with network-platform average

declare today default current_date();

insert into `relax-melodies-android.late_conversions.monthly_renewal_rates`
WITH cohorts_ranked as (
  select
    format_date('%Y-%m', today) as year_month,
    *,
  rank() over (partition by network, platform, country_code, renewal_bucket order by paid_year_month desc) as rn
  from `relax-melodies-android.late_conversions.mature_renewal_cohorts`
  where
    ((renewal_bucket = '1-Year' AND paid_year_month < FORMAT_DATE('%Y-%m', DATE_SUB(today, INTERVAL 1 year)))
    OR (renewal_bucket = '2-Years' AND paid_year_month < FORMAT_DATE('%Y-%m', DATE_SUB(today, INTERVAL 2 year)))
    OR (renewal_bucket = '3-Years' AND paid_year_month < FORMAT_DATE('%Y-%m', DATE_SUB(today, INTERVAL 3 year))))
), latest_renewal_rates as (
    select
      year_month,
      network,
      platform,
      country_code,
      renewal_bucket,
      num_renewals,
      num_paid,
      paid_proceeds,
      renewal_proceeds,
      renewal_percentage
    from cohorts_ranked
    where rn = 1
), pivot_table AS (
  SELECT
    year_month,
    platform,
    network,
    country_code,
    `1-Year`,
    `2-Years`,
    `3-Years`
  FROM (
    SELECT
      year_month,
      platform,
      network,
      country_code,
      renewal_bucket,
      renewal_percentage
    FROM latest_renewal_rates
  )
  PIVOT (
    AVG(renewal_percentage)
    FOR renewal_bucket IN ('1-Year', '2-Years', '3-Years')
  )
),
pivot_table_imputed AS (
  SELECT
    parse_date('%Y-%m', year_month) as year_month,
    -- extract(year from parse_date('%Y-%m', year_month)) as year,
    -- extract(month from parse_date('%Y-%m', year_month)) as month,
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
