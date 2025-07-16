insert into `relax-melodies-android.late_conversions.users_renewal_daily`

with daily_renewals as (
  select
    date(timestamp_micros(event_timestamp)) as event_date,
    event_timestamp,
    user_id,
    user_pseudo_id,
    lower(platform) as platform,
    geo.country as country,
    current_timestamp()
  from `relax-melodies-android.backend.events`
  where
    timestamp_trunc(event_timestamp_s, DAY) >= TIMESTAMP("2025-01-01")
    and timestamp_trunc(event_timestamp_s, DAY) <= TIMESTAMP("2025-07-13")
    and event_name = 'subscription_renew_paid'
    and user_id is not null
)

select * from daily_renewals

--  Notes:
--  => Populated "2025-06-01"
