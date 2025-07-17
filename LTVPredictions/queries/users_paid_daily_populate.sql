insert into `relax-melodies-android.late_conversions.users_paid_daily`

with daily_paid as (
  select
    PARSE_DATE('%Y%m%d', event_date) as event_date,
    event_timestamp,
    user_id,
    user_pseudo_id,
    lower(platform) as platform,
    geo.country as country,
    ep1.value.float_value as proceeds,
    ep2.value.string_value as currency,
    ep3.value.string_value as feature_id,
    current_timestamp() as load_timestamp,
  from `relax-melodies-android.backend.events`,
    unnest(event_params) as ep1,
    unnest(event_params) as ep2,
    unnest(event_params) as ep3
  where
    timestamp_trunc(event_timestamp_s, DAY) >= TIMESTAMP("2025-01-01")
    and timestamp_trunc(event_timestamp_s, DAY) <= TIMESTAMP("2025-07-13")
    and event_name = 'subscription_start_paid'
    and user_id is not null
    and ep1.key in ('converted_procceds', 'converted_proceeds')
    and ep2.key = 'currency'
    and ep3.key = 'feature_id'
)

select * from daily_paid

--  Notes:
--  => Populated "2025-06-01"
