-- `late_conversions.renewal_rates_cohorted`
-- `late_conversions.renewal_proceeds_cohorted`

declare start_date string default "2020-01-01";
declare end_date string default "2025-07-01";

with paid_data as (
  SELECT
    user_id,
    event_name,
    platform,
    event_timestamp_s as paid_timestamp_s,
    ep.value.float_value AS proceeds,
    row_number() over (partition by user_id, event_name, platform order by event_timestamp_s desc) as rn
  FROM `relax-melodies-android.backend.events`,
    UNNEST(event_params) as ep
  WHERE
    TIMESTAMP_TRUNC(event_timestamp_s, DAY) >= TIMESTAMP(start_date)
      and TIMESTAMP_TRUNC(event_timestamp_s, DAY) <= TIMESTAMP(end_date)
    and event_name = 'subscription_start_paid'
    and user_id is not null
    and ep.key = 'converted_procceds'
), paid_unique as (
  select
    user_id,
    event_name,
    platform,
    proceeds,
    paid_timestamp_s
  from paid_data
  where rn = 1
), renewal_data as (
  SELECT
    user_id,
    event_name,
    platform,
    case when ep.key = 'converted_procceds' then ep.value.float_value end AS proceeds,
    case when ep.key = 'feature_id' then ep.value.string_value end as renewal_sku,
    event_timestamp_s as renewal_timestamp_s,
    lag(event_timestamp_s) over (partition by user_id order by event_timestamp_s) as prev_renewal_timestamp_s
  FROM `relax-melodies-android.backend.events`,
    UNNEST(event_params) as ep
  WHERE
    TIMESTAMP_TRUNC(event_timestamp_s, DAY) >= TIMESTAMP(start_date)
      and TIMESTAMP_TRUNC(event_timestamp_s, DAY) <= TIMESTAMP(end_date)
    and event_name = 'subscription_renew_paid'
    and (user_id is not null or user_id != '')
    and ep.key in ('converted_procceds', 'feature_id')
), renewal_unique as ( -- remove renewal processed twice
  SELECT
    user_id,
    event_name,
    platform,
    proceeds,
    renewal_timestamp_s
  FROM renewal_data
  WHERE
    prev_renewal_timestamp_s is NULL or TIMESTAMP_SUB(renewal_timestamp_s, INTERVAL 1 DAY) > prev_renewal_timestamp_s
    and renewal_sku like '%year%' --- only keep yearly sku
  ORDER BY user_id, renewal_timestamp_s
), joined_table as (
    SELECT
      p.user_id,
      p.platform,
      format_timestamp('%Y-%m', p.paid_timestamp_s) as paid_year_month,
      p.paid_timestamp_s,
      r.renewal_timestamp_s,
      p.proceeds as paid_proceeds,
      r.proceeds as renewal_proceeds,
      DATE_DIFF(r.renewal_timestamp_s, p.paid_timestamp_s, DAY) as days_to_renewal,
      case
        when DATE_DIFF(r.renewal_timestamp_s, p.paid_timestamp_s, DAY) <= 31 then '1-Month'
        when DATE_DIFF(r.renewal_timestamp_s, p.paid_timestamp_s, DAY) between 31 and 60 then '2-Months'
        when DATE_DIFF(r.renewal_timestamp_s, p.paid_timestamp_s, DAY) between 60 and 180 then 'Half-Year'
        when DATE_DIFF(r.renewal_timestamp_s, p.paid_timestamp_s, DAY) between 180 and 365 then '1-Year'
        when DATE_DIFF(r.renewal_timestamp_s, p.paid_timestamp_s, DAY) between 365 and 730 then '2-Years'
        when DATE_DIFF(r.renewal_timestamp_s, p.paid_timestamp_s, DAY) between 730 and 1095 then '3-Years'
        when DATE_DIFF(r.renewal_timestamp_s, p.paid_timestamp_s, DAY) between 1095 and 1460 then '4-Years'
        when DATE_DIFF(r.renewal_timestamp_s, p.paid_timestamp_s, DAY) between 1460 and 1825 then '5-Years'
        else '>5-Years'
      end as renewal_bucket
  FROM paid_unique as p
    LEFT JOIN renewal_data r on p.user_id = r.user_id
  ORDER BY user_id, days_to_renewal
), paid_cohorts as (
  select
    paid_year_month,
    platform,
    count(*) as num_paid,
    avg(paid_proceeds) as paid_proceeds
  from joined_table
  group by
    paid_year_month,
    platform
), renewal_cohorts as (
  select
    paid_year_month,
    platform,
    renewal_bucket,
    count(*) as num_renewals,
    avg(renewal_proceeds) as renewal_proceeds
  from joined_table
  where
    renewal_timestamp_s is not null
    and renewal_bucket not in ('1-Month', '2-Months', 'Half-Year')
  group by
    paid_year_month,
    platform,
    renewal_bucket
), aggregate_cohorts as (
  select
    r.paid_year_month,
    r.platform,
    r.renewal_bucket,
    r.renewal_proceeds,
    p.paid_proceeds,
    p.num_paid,
    r.num_renewals,
    round(r.num_renewals / p.num_paid * 100.0, 4) as renewal_percentage
  from renewal_cohorts as r
  left join paid_cohorts as p
  on r.paid_year_month = p.paid_year_month and r.platform = p.platform
)

--- for "renewal_percentage"
select
  *
from (
  select
    paid_year_month,
    platform,
    renewal_bucket,
    renewal_percentage
  from aggregate_cohorts
)
pivot (
  avg(renewal_percentage)
  FOR renewal_bucket in ('1-Year', '2-Years', '3-Years', '4-Years', '5-Years', '>5-Years')
)
order by paid_year_month, platform

--- for "renewal_proceeds"
select
  *
from (
  select
    paid_year_month,
    platform,
    renewal_bucket,
    renewal_proceeds
  from aggregate_cohorts
)
pivot (
  avg(renewal_proceeds)
  FOR renewal_bucket in ('1-Year', '2-Years', '3-Years', '4-Years', '5-Years', '>5-Years')
)
order by paid_year_month, platform
