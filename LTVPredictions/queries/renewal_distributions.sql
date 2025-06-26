declare start_date string default "2024-01-01";
declare end_date string default "2025-06-26";

with paid_data as (
  SELECT
    user_id,
    event_name,
    platform,
    -- traffic_source.medium,  -- always null
    -- geo.country as country, -- always null
    -- user_first_touch_timestamp, -- always null
    event_timestamp_s as paid_timestamp_s,
    row_number() over (partition by user_id, event_name, platform order by event_timestamp_s desc) as rn
  FROM `relax-melodies-android.backend.events`
  WHERE
    TIMESTAMP_TRUNC(event_timestamp_s, DAY) >= TIMESTAMP(start_date)
      and TIMESTAMP_TRUNC(event_timestamp_s, DAY) <= TIMESTAMP(end_date)
    and event_name = 'subscription_start_paid'
    and user_id is not null
), paid_unique as (
  select
    user_id,
    event_name,
    platform,
    paid_timestamp_s
  from paid_data
  where rn = 1
), renewal_data as (
  SELECT
    user_id,
    event_name,
    platform,
    event_timestamp_s as renewal_timestamp_s,
    row_number() over (partition by user_id, event_name, platform order by event_timestamp_s desc) as rn
  FROM `relax-melodies-android.backend.events`
  WHERE
    TIMESTAMP_TRUNC(event_timestamp_s, DAY) >= TIMESTAMP(start_date)
      and TIMESTAMP_TRUNC(event_timestamp_s, DAY) <= TIMESTAMP(end_date)
    and event_name = 'subscription_renew_paid'
    and user_id is not null
), renewal_unique as (
  select
    user_id,
    event_name,
    platform,
    renewal_timestamp_s
  from renewal_data
  where rn = 1
), joined_table as (
    SELECT
      p.user_id,
      p.platform,
      p.paid_timestamp_s,
      r.renewal_timestamp_s,
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
    JOIN renewal_unique r on p.user_id = r.user_id
), conversions_table as (
    select
      renewal_bucket,
      count(*) as counts,
      round(100 * count(*) / sum(count(*)) over (), 2) as percentage,
    from joined_table
    group by renewal_bucket
)

select * from conversions_table
