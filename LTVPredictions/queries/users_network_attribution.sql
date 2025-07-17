DECLARE conversion_window INT64 DEFAULT 60;
DECLARE single_day STRING DEFAULT '2025-05-01';

with hau as (
  select
    user_id,
    user_pseudo_id,
    hau,
    traffic_source,
    traffic_source_name
  from `relax-melodies-android.late_conversions.users_hau_daily`
  where
    event_date between (date(single_day) - interval conversion_window day) and date(single_day)
    and user_pseudo_id is not null
), utm as (
  select
    user_id,
    user_pseudo_id,
    platform,
    utm_source,
    campaign
  from `relax-melodies-android.late_conversions.users_utm_daily`
  where
    event_date between (date(single_day) - interval conversion_window day) and date(single_day)
    and user_pseudo_id is not null
), hau_utm as (
  select
    hau.user_id,
    hau.user_pseudo_id,
    hau.hau,
    hau.traffic_source,
    hau.traffic_source_name,
    utm.utm_source,
    utm.campaign,
  from hau
  full outer join utm
  on
    hau.user_pseudo_id = utm.user_pseudo_id
    and hau.user_id = utm.user_id
)

select
  user_id,
  user_pseudo_id,
  max(hau) as hau,
  max(traffic_source) as traffic_source,
  max(traffic_source_name) as traffic_source_name,
  max(utm_source) as utm_source,
  max(campaign) as campaign,
  case
    when max(utm_source) != 'no user consent' then max(utm_source)
    when max(utm_source) = 'no user consent' and max(hau) = "['other']" then 'unavailable'
    else max(hau)
  end as network_attribution
from hau_utm
group by user_id, user_pseudo_id
