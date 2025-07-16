insert into `relax-melodies-android.late_conversions.users_trials_daily`

with daily_trials as (
  select
    date(timestamp_micros(event_timestamp)) as event_date,
    event_timestamp,
    user_id,
    user_pseudo_id,
    lower(platform) as platform,
    geo.country as country,
    traffic_source.name as traffic_source_name,
    traffic_source.medium as traffic_source_medium,
    traffic_source.source as traffic_source,
  from `relax-melodies-android.sandbox.analytics_events_pc`
  where
    timestamp_trunc(event_date_partition, DAY) >= TIMESTAMP("2025-06-01")
    and timestamp_trunc(event_date_partition, DAY) <= TIMESTAMP("2025-06-01")
    and event_name = 'subscription_process_succeed'
    and user_pseudo_id is not null
)

select * from daily_trials

--  Notes:
--  => Populated "2025-06-01"
