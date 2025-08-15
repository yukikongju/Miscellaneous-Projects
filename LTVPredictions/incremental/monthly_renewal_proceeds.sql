-- cost: 92.81 KB
DECLARE renewal_threshold FLOAT64 DEFAULT 0.8;
--  DECLARE renewal_rate_depreciation_rate float64 default 0.6;
DECLARE default_renewal_proceeds float64 default 50.0;


declare today default current_date();

insert into `relax-melodies-android.late_conversions.monthly_renewal_proceeds`
WITH defaults_network_platform_country as (
  select
    network,
    platform,
    country_code,
    avg(`1-Year`) as avg_1y,
    avg(`2-Years`) as avg_2y,
    avg(`3-Years`) as avg_3y,
  from `relax-melodies-android.late_conversions.monthly_renewal_proceeds`
  group by
    network, platform, country_code
), defaults_platform_country as (
  select
    platform,
    country_code,
    avg(`1-Year`) as avg_1y,
    avg(`2-Years`) as avg_2y,
    avg(`3-Years`) as avg_3y,
  from  `relax-melodies-android.late_conversions.monthly_renewal_proceeds`
  group by
    platform, country_code
), defaults_country as (
  select
    country_code,
    avg(`1-Year`) as avg_1y,
    avg(`2-Years`) as avg_2y,
    avg(`3-Years`) as avg_3y,
  from  `relax-melodies-android.late_conversions.monthly_renewal_proceeds`
  group by
    country_code
), cohorts_ranked as (
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
      renewal_proceeds
    FROM latest_renewal_rates
  )
  PIVOT (
    AVG(renewal_proceeds)
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
    END AS `3-Years`,
  FROM pivot_table
), pivot_table_complete as (
  select
	  *
  from pivot_table_imputed
  union all ( -- copy tatari_streaming rates as tatari_linear
	select
	    year_month, platform,
      'tatari_linear' as network,
      country_code, `1-Year`, `2-Years`, `3-Years`
	from pivot_table_imputed
	where
	    network = 'tatari_streaming'
    )
), keys as (
  select
    month_start,
    platform,
    network,
    c.country_code
  from UNNEST(GENERATE_DATE_ARRAY(today, today, INTERVAL 1 MONTH)) AS month_start
  cross join unnest(['ios', 'android']) as platform
  cross join unnest(['Facebook Ads', 'tiktokglobal_int', 'snapchat_int', 'googleadwords_int',
    'tatari_streaming', 'tatari_linear', 'Apple Search Ads', 'Organic']) as network
  cross join (select country_code from `relax-melodies-android.mappings.country_maps`) as c
), complete_table as (
  select
    -- format_date('%Y-%m', k.month_start) as year_month,
    k.month_start,
    k.platform,
    k.network,
    k.country_code,
    coalesce(ptc.`1-Year`, npc.avg_1y, pc.avg_1y, c.avg_1y, default_renewal_proceeds) as `1-Year`,
    coalesce(ptc.`2-Years`, npc.avg_2y, pc.avg_2y, c.avg_2y, default_renewal_proceeds) as `2-Years`,
    coalesce(ptc.`3-Years`, npc.avg_3y, pc.avg_3y, c.avg_3y, default_renewal_proceeds) as `3-Years`,
    current_timestamp() as loaded_timestamp,
  from keys k
  left join pivot_table_complete ptc
    -- on k.month_start = ptc.year_month
    on k.network = ptc.network
      and k.platform = ptc.platform
      and k.country_code = ptc.country_code
  left join defaults_network_platform_country npc
    on k.network = npc.network
      and k.platform = npc.platform
      and k.country_code = npc.country_code
  left join defaults_platform_country pc
    on k.platform = pc.platform
    and k.country_code = pc.country_code
  left join defaults_country c
    on k.country_code = c.country_code
)

-- Use this to select either raw or imputed version
SELECT * FROM complete_table
