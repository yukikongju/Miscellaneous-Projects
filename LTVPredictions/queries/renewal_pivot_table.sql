-- `late_conversions.renewal_rates_cohorted`
-- `late_conversions.renewal_proceeds_cohorted`

DECLARE
  start_date string DEFAULT "2020-01-01";
DECLARE
  end_date string DEFAULT "2025-07-01";
WITH
  paid_data AS (
  SELECT
    user_id,
    event_name,
    platform,
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
    platform,
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
    platform,
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
    platform,
    proceeds,
    renewal_timestamp_s
  FROM
    renewal_data
  WHERE
    prev_renewal_timestamp_s IS NULL
    OR TIMESTAMP_SUB(renewal_timestamp_s, INTERVAL 1 DAY) > prev_renewal_timestamp_s
    AND renewal_sku LIKE '%year%' --- only keep yearly sku
  ORDER BY
    user_id,
    renewal_timestamp_s ),
  joined_table AS (
  SELECT
    p.user_id,
    p.platform,
    FORMAT_TIMESTAMP('%Y-%m', p.paid_timestamp_s) AS paid_year_month,
    p.paid_timestamp_s,
    r.renewal_timestamp_s,
    p.proceeds AS paid_proceeds,
    r.proceeds AS renewal_proceeds,
    DATE_DIFF(r.renewal_timestamp_s, p.paid_timestamp_s, DAY) AS days_to_renewal,
    CASE
      WHEN DATE_DIFF(r.renewal_timestamp_s, p.paid_timestamp_s, DAY) <= 31 THEN '1-Month'
      WHEN DATE_DIFF(r.renewal_timestamp_s, p.paid_timestamp_s, DAY) BETWEEN 31
    AND 60 THEN '2-Months'
      WHEN DATE_DIFF(r.renewal_timestamp_s, p.paid_timestamp_s, DAY) BETWEEN 60 AND 180 THEN 'Half-Year'
      WHEN DATE_DIFF(r.renewal_timestamp_s, p.paid_timestamp_s, DAY) BETWEEN 180
    AND 365 THEN '1-Year'
      WHEN DATE_DIFF(r.renewal_timestamp_s, p.paid_timestamp_s, DAY) BETWEEN 365 AND 730 THEN '2-Years'
      WHEN DATE_DIFF(r.renewal_timestamp_s, p.paid_timestamp_s, DAY) BETWEEN 730
    AND 1095 THEN '3-Years'
      WHEN DATE_DIFF(r.renewal_timestamp_s, p.paid_timestamp_s, DAY) BETWEEN 1095 AND 1460 THEN '4-Years'
      WHEN DATE_DIFF(r.renewal_timestamp_s, p.paid_timestamp_s, DAY) BETWEEN 1460
    AND 1825 THEN '5-Years'
      ELSE '>5-Years'
  END
    AS renewal_bucket
  FROM
    paid_unique AS p
  LEFT JOIN
    renewal_data r
  ON
    p.user_id = r.user_id
  ORDER BY
    user_id,
    days_to_renewal ),
  paid_cohorts AS (
  SELECT
    paid_year_month,
    platform,
    COUNT(*) AS num_paid,
    AVG(paid_proceeds) AS paid_proceeds
  FROM
    joined_table
  GROUP BY
    paid_year_month,
    platform ),
  renewal_cohorts AS (
  SELECT
    paid_year_month,
    platform,
    renewal_bucket,
    COUNT(*) AS num_renewals,
    AVG(renewal_proceeds) AS renewal_proceeds
  FROM
    joined_table
  WHERE
    renewal_timestamp_s IS NOT NULL
    AND renewal_bucket NOT IN ('1-Month',
      '2-Months',
      'Half-Year')
  GROUP BY
    paid_year_month,
    platform,
    renewal_bucket ),
  aggregate_cohorts AS (
  SELECT
    r.paid_year_month,
    r.platform,
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
    AND r.platform = p.platform ),
  mature_cohorts AS (
  SELECT
    paid_year_month,
    platform,
    renewal_bucket,
    renewal_proceeds,
    paid_proceeds,
    num_paid,
    num_renewals,
    renewal_percentage
  FROM
    aggregate_cohorts
  WHERE (
      (renewal_bucket = '1-Year'AND paid_year_month < FORMAT_DATE('%Y-%m', DATE_SUB(CURRENT_DATE(), INTERVAL 1 year))) OR
      (renewal_bucket = '2-Years'AND paid_year_month < FORMAT_DATE('%Y-%m', DATE_SUB(CURRENT_DATE(), INTERVAL 2 year))) OR

      (renewal_bucket = '3-Years'AND paid_year_month < FORMAT_DATE('%Y-%m', DATE_SUB(CURRENT_DATE(), INTERVAL 3 year))) OR
      (renewal_bucket = '4-Years'AND paid_year_month < FORMAT_DATE('%Y-%m', DATE_SUB(CURRENT_DATE(), INTERVAL 4 year))) OR
      (renewal_bucket = '5-Years'AND paid_year_month < FORMAT_DATE('%Y-%m', DATE_SUB(CURRENT_DATE(), INTERVAL 5 year)))
  )
  )



--- for "renewal_percentage"
SELECT
  *
FROM (
  SELECT
    paid_year_month,
    platform,
    renewal_bucket,
    renewal_percentage
  FROM
    mature_cohorts )
PIVOT
  ( AVG(renewal_percentage) FOR renewal_bucket IN ('1-Year',
      '2-Years',
      '3-Years',
      '4-Years',
      '5-Years',
      '>5-Years') )
ORDER BY
  paid_year_month,
  platform


--- for "renewal_proceeds"
select
*
from (
select
 paid_year_month,
 platform,
 renewal_bucket,
 renewal_proceeds
from mature_cohorts
)
pivot (
avg(renewal_proceeds)
FOR renewal_bucket in ('1-Year', '2-Years', '3-Years', '4-Years', '5-Years', '>5-Years')
)
order by paid_year_month, platform
