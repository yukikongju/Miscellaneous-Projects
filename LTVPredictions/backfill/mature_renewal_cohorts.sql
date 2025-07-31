--- cost: 1.32GB run monthly
DECLARE
  start_date string DEFAULT "2023-01-01";
-- DECLARE end_date string DEFAULT "2025-07-01";
DECLARE end_date string DEFAULT FORMAT_TIMESTAMP('%F', TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), MONTH)); --- current month

create or replace table `relax-melodies-android.late_conversions.mature_renewal_cohorts`
-- partition by paid_year_month
cluster by network, platform, country_code, paid_year_month
as
WITH
  paid_data AS (
  SELECT
    user_id,
    event_name,
    -- platform,
    event_timestamp_s AS paid_timestamp_s,
    ep.value.float_value AS proceeds,
    ROW_NUMBER() OVER (PARTITION BY user_id, event_name, platform ORDER BY event_timestamp_s DESC) AS rn
  FROM
    `relax-melodies-android.backend.events`,
    UNNEST(event_params) AS ep
  WHERE
    TIMESTAMP_TRUNC(event_timestamp_s, DAY) >= TIMESTAMP(start_date)
    AND TIMESTAMP_TRUNC(event_timestamp_s, DAY) <= TIMESTAMP(end_date)
    AND event_name = 'subscription_start_paid'
    AND user_id IS NOT NULL
    AND ep.key = 'converted_procceds' ),
  paid_unique AS (
  SELECT
    user_id,
    event_name,
    -- platform,
    proceeds,
    paid_timestamp_s
  FROM
    paid_data
  WHERE
    rn = 1 ),
  renewal_data AS (
  SELECT
    user_id,
    event_name,
    -- platform,
    CASE
      WHEN ep.key = 'converted_procceds' THEN ep.value.float_value
  END
    AS proceeds,
    CASE
      WHEN ep.key = 'feature_id' THEN ep.value.string_value
  END
    AS renewal_sku,
    event_timestamp_s AS renewal_timestamp_s,
    LAG(event_timestamp_s) OVER (PARTITION BY user_id ORDER BY event_timestamp_s) AS prev_renewal_timestamp_s
  FROM
    `relax-melodies-android.backend.events`,
    UNNEST(event_params) AS ep
  WHERE
    TIMESTAMP_TRUNC(event_timestamp_s, DAY) >= TIMESTAMP(start_date)
    AND TIMESTAMP_TRUNC(event_timestamp_s, DAY) <= TIMESTAMP(end_date)
    AND event_name = 'subscription_renew_paid'
    AND (user_id IS NOT NULL
      OR user_id != '')
    AND ep.key IN ('converted_procceds',
      'feature_id') ),
  renewal_unique AS ( -- remove renewal processed twice
  SELECT
    user_id,
    event_name,
    -- platform,
    proceeds,
    renewal_timestamp_s
  FROM
    renewal_data
  WHERE
    prev_renewal_timestamp_s IS NULL
    OR TIMESTAMP_SUB(renewal_timestamp_s, INTERVAL 1 DAY) > prev_renewal_timestamp_s
    AND renewal_sku LIKE '%year%' --- only keep yearly sku
    AND proceeds is not null
  ORDER BY
    user_id,
    renewal_timestamp_s ),
  joined_table AS (
  SELECT
    p.user_id,
    u.user_pseudo_id,
    u.network_attribution,
    u.platform,
    u.country_code,
    -- p.platform,
    FORMAT_TIMESTAMP('%Y-%m', p.paid_timestamp_s) AS paid_year_month,
    p.paid_timestamp_s,
    r.renewal_timestamp_s,
    p.proceeds AS paid_proceeds,
    r.proceeds AS renewal_proceeds,
    DATE_DIFF(r.renewal_timestamp_s, p.paid_timestamp_s, DAY) AS days_to_renewal,
    CASE
      WHEN DATE_DIFF(r.renewal_timestamp_s, p.paid_timestamp_s, DAY) <= 31 THEN '1-Month'
      WHEN DATE_DIFF(r.renewal_timestamp_s, p.paid_timestamp_s, DAY) BETWEEN 31 AND 60 THEN '2-Months'
      WHEN DATE_DIFF(r.renewal_timestamp_s, p.paid_timestamp_s, DAY) BETWEEN 60 AND 180 THEN 'Half-Year'
      WHEN DATE_DIFF(r.renewal_timestamp_s, p.paid_timestamp_s, DAY) BETWEEN 180 AND 365 THEN '1-Year'
      WHEN DATE_DIFF(r.renewal_timestamp_s, p.paid_timestamp_s, DAY) BETWEEN 365 AND 730 THEN '2-Years'
      WHEN DATE_DIFF(r.renewal_timestamp_s, p.paid_timestamp_s, DAY) BETWEEN 730 AND 1095 THEN '3-Years'
      WHEN DATE_DIFF(r.renewal_timestamp_s, p.paid_timestamp_s, DAY) BETWEEN 1095 AND 1460 THEN '4-Years'
      WHEN DATE_DIFF(r.renewal_timestamp_s, p.paid_timestamp_s, DAY) BETWEEN 1460 AND 1825 THEN '5-Years'
      when DATE_DIFF(r.renewal_timestamp_s, p.paid_timestamp_s, DAY) > 1825 THEN '>5-Years'
      ELSE 'No Renewal'
  END
    AS renewal_bucket
  FROM
    paid_unique AS p
  LEFT JOIN
    renewal_data r
  ON
    p.user_id = r.user_id
  LEFT JOIN
    `relax-melodies-android.late_conversions.users_network_attribution` u
  ON
    p.user_id = u.user_id
  -- WHERE
  --   p.proceeds is not null and r.proceeds is not null
  ORDER BY
    user_id,
    days_to_renewal ),
  joined_table_filtered as (
    select
      user_id,
      user_pseudo_id,
      network_attribution,
      platform,
      country_code,
      paid_timestamp_s,
      renewal_timestamp_s,
      paid_year_month,
      paid_proceeds,
      renewal_proceeds,
      renewal_bucket
    from joined_table
    where
      renewal_bucket not in (
        '1-Month',
        '2-Months',
        'Half-Year'
      )
      and network_attribution is not null
      and ((renewal_timestamp_s is not null and renewal_proceeds is not null) or (renewal_timestamp_s is null and renewal_proceeds is null))
      and country_code is not null
  ),
  paid_cohorts AS (
  SELECT
    paid_year_month,
    network_attribution,
    country_code,
    platform,
    COUNT(*) AS num_paid,
    AVG(paid_proceeds) AS paid_proceeds
  FROM
    joined_table_filtered
  WHERE paid_timestamp_s is not null
  GROUP BY
    paid_year_month,
    platform,
    network_attribution,
    country_code
  ),
  renewal_cohorts AS (
  SELECT
    paid_year_month,
    platform,
    network_attribution,
    country_code,
    renewal_bucket,
    COUNT(*) AS num_renewals,
    AVG(renewal_proceeds) AS renewal_proceeds
  FROM
    joined_table_filtered
  WHERE
    renewal_timestamp_s IS NOT NULL
  GROUP BY
    paid_year_month,
    platform,
    network_attribution,
    country_code,
    renewal_bucket
    ),
  aggregate_cohorts AS (
  SELECT
    r.paid_year_month,
    r.platform,
    p.network_attribution,
    p.country_code,
    r.renewal_bucket,
    r.renewal_proceeds,
    p.paid_proceeds,
    p.num_paid,
    r.num_renewals,
    ROUND(r.num_renewals / p.num_paid * 100.0, 4) AS renewal_percentage
  FROM
    renewal_cohorts AS r
  LEFT JOIN
    paid_cohorts AS p
  ON
    r.paid_year_month = p.paid_year_month
    AND r.platform = p.platform
    and r.network_attribution = p.network_attribution
    and r.country_code = p.country_code
    ),
  mature_cohorts AS (
  SELECT
    paid_year_month,
    platform,
    network_attribution AS network,
    country_code,
    renewal_bucket,
    renewal_proceeds,
    paid_proceeds,
    num_paid,
    num_renewals,
    renewal_percentage
  FROM
    aggregate_cohorts
  WHERE
    ( (renewal_bucket = '1-Year'AND paid_year_month < FORMAT_DATE('%Y-%m', DATE_SUB(CURRENT_DATE(), INTERVAL 1 year)))
      OR (renewal_bucket = '2-Years'AND paid_year_month < FORMAT_DATE('%Y-%m', DATE_SUB(CURRENT_DATE(), INTERVAL 2 year)))
      OR (renewal_bucket = '3-Years'AND paid_year_month < FORMAT_DATE('%Y-%m', DATE_SUB(CURRENT_DATE(), INTERVAL 3 year)))
      OR (renewal_bucket = '4-Years'AND paid_year_month < FORMAT_DATE('%Y-%m', DATE_SUB(CURRENT_DATE(), INTERVAL 4 year)))
      OR (renewal_bucket = '5-Years'AND paid_year_month < FORMAT_DATE('%Y-%m', DATE_SUB(CURRENT_DATE(), INTERVAL 5 year))) )
  ), mature_cohorts_ma as (
    select
    paid_year_month, network, platform, country_code, renewal_bucket,
    round(avg(num_paid) over (partition by network, platform, country_code, renewal_bucket order by paid_year_month rows between 2 preceding and current row), 3) as num_paid,
    round(avg(num_renewals) over (partition by network, platform, country_code, renewal_bucket order by paid_year_month rows between 2 preceding and current row), 3) as num_renewals,
    round(avg(paid_proceeds) over (partition by network, platform, country_code, renewal_bucket order by paid_year_month rows between 2 preceding and current row), 3) as paid_proceeds,
    round(avg(renewal_proceeds) over (partition by network, platform, country_code, renewal_bucket order by paid_year_month rows between 2 preceding and current row), 3) as renewal_proceeds,
    ROUND(
    AVG(num_renewals) OVER (
        PARTITION BY network, platform, country_code, renewal_bucket
        ORDER BY paid_year_month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) /
    NULLIF(AVG(num_paid) OVER (
        PARTITION BY network, platform, country_code, renewal_bucket
        ORDER BY paid_year_month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 0),
    5) AS renewal_percentage,
    -- rank() over (partition by network, platform, country_code, renewal_bucket order by paid_year_month desc) as rn
    from mature_cohorts
  )


  ---
  -- select * from joined_table
  -- where
  --   renewal_timestamp_s is not null and network_attribution is not null

-- select * from aggregate_cohorts

-- select * from joined_table_filtered
-- select count(*) from paid_unique --- 378180
-- select count(*) from renewal_unique --- 181634
-- select count(*) from joined_table --- 69502

-- select * from paid_cohorts --- script_job_9031264432ebd72e3c46d46ac5664082_0.csv
-- select * from renewal_cohorts --- script_job_10c458d7cc38fd359c3139c394c9337d_0.csv
-- select * from mature_cohorts order by platform, network, country_code, paid_year_month

select * from mature_cohorts_ma

-- select * from latest_renewal_rates

---- LATEST MATURE RENEWAL RATE & PROCEEDS ----
--- for "renewal_percentage"
-- SELECT
--   *
-- FROM (
--   SELECT
--     platform,
--     network,
--     country_code,
--     renewal_bucket,
--     renewal_percentage
--   FROM
--     latest_rates )
-- PIVOT
--   ( AVG(renewal_percentage) FOR renewal_bucket IN ('1-Year',
--       '2-Years',
--       '3-Years'
--       -- '4-Years',
--       -- '5-Years',
--       -- '>5-Years'
--       ))
-- order by
--   platform, network, country_code

--- for "renewal_proceeds"

---- RENEWAL RATE & PROCEEDS PER PAID_YEAR_MONTH ----

--- for "renewal_percentage"
-- SELECT
--   *
-- FROM (
--   SELECT
--     paid_year_month,
--     platform,
--     network,
--     country_code,
--     renewal_bucket,
--     renewal_percentage
--   FROM
--     mature_cohorts_ma )
-- PIVOT
--   ( AVG(renewal_percentage) FOR renewal_bucket IN ('1-Year',
--       '2-Years',
--       '3-Years',
--       '4-Years',
--       '5-Years',
--       '>5-Years') )
-- ORDER BY
--   paid_year_month,
--   network,
--   platform,
--   country_code


  --- for "renewal_proceeds"
  -- select
  -- *
  -- from (
  -- select
  --  paid_year_month,
  --  platform,
  -- network,
  -- country_code,
  --  renewal_bucket,
  --  renewal_proceeds
  -- from mature_cohorts_ma
  -- )
  -- pivot (
  -- avg(renewal_proceeds)
  -- FOR renewal_bucket in ('1-Year', '2-Years', '3-Years', '4-Years', '5-Years', '>5-Years')
  -- )
  -- order by paid_year_month, network, platform, country_code
