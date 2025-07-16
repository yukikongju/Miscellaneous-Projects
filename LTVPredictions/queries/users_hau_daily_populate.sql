insert into `relax-melodies-android.late_conversions.users_hau_daily`

with daily_hau as (
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
    ep2.value.string_value as hau,
    current_timestamp()
  from `relax-melodies-android.sandbox.analytics_events_pc`,
    unnest(event_params) as ep1,
    unnest(event_params) as ep2
  where
    timestamp_trunc(event_date_partition, DAY) >= TIMESTAMP("2025-01-01")
    and timestamp_trunc(event_date_partition, DAY) <= TIMESTAMP("2025-07-13")
    and event_name = 'answer_question'
    and ep1.key in ('question_id')
    and lower(ep1.value.string_value) = 'hearaboutus'
    and ep2.key in ('answers', 'answer')
    and user_pseudo_id is not null
)

select * from daily_hau
