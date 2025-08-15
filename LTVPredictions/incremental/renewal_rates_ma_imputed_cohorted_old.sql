-- cost: 77.06 MB
DECLARE renewal_threshold FLOAT64 DEFAULT 0.8;
DECLARE renewal_rate_depreciation_rate float64 default 0.6;
DECLARE default_renewal_rate float64 default 0.3;

-- `monthly_renewal_rates_cohorted` => compute available renewal rates monthly
-- `monthly_renewal_rates_imputed` => view that impute renewal rates for missing with average and default
--

-- TODO: use network renewal rate to impute instead of defaulting to default rate
-- TODO: impute missing values with network-platform average

declare today default current_date();

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
    year_month,
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
    year_month, platform, network, country_code,
    `1-Year`,
    coalesce(`2-Years`, `1-Year` * renewal_rate_depreciation_rate) as `2-Years`,
    coalesce(`3-Years`,
      case
        when `2-Years` is not null then `2-Years` * renewal_rate_depreciation_rate
        else `1-Year` * renewal_rate_depreciation_rate * renewal_rate_depreciation_rate
      end
    ) as `3-Years`,
    -- current_timestamp() as loaded_timestamp,
  from pivot_table_imputed
), pivot_table_complete as (
    select
	*
    from pivot_table_default
    union all ( -- copy tatari_streaming rates as tatari_linear
	select
	    year_month, platform,
      'tatari_linear' as network,
      country_code, `1-Year`, `2-Years`, `3-Years`
	from pivot_table_default
	where
	    network = 'tatari_streaming'
    )
), keys as (
  select distinct
    format_date('%Y-%m', today) as year_month,
    network, platform, country,
  from `relax-melodies-android.ua_transform_prod.ua_metrics`
), complete_table as (
  select
    k.year_month,
    k.network,
    k.platform,
    k.country,
    coalesce(`1-Year`, default_renewal_rate) as `1-Year`,
    coalesce(`2-Years`, default_renewal_rate) as `2-Years`,
    coalesce(`3-Years`, default_renewal_rate) as `3-Years`,
    current_timestamp() as loaded_timestamp,
  from keys as k
    left join pivot_table_complete t
    on k.year_month = t.year_month
      and k.network = t.network
      and k.platform = t.platform
      and k.country = t.country_code
)

-- Use this to select either raw or imputed version
SELECT * FROM complete_table
ORDER BY year_month, network, platform, country
