DECLARE conversion_window INT64 DEFAULT 60;
DECLARE single_day STRING DEFAULT '2025-05-01';

with trials as (
  select
    user_id,
    user_pseudo_id,
    platform,
    max(event_date) as event_date
  from `relax-melodies-android.late_conversions.users_trials_daily`
  where
    event_date = date(single_day)
    and user_id is not null
  group by user_id, user_pseudo_id, platform
), paid as (
  select
    user_id,
    platform,
    max(event_date) as event_date,
    avg(proceeds) as proceeds
  from `relax-melodies-android.late_conversions.users_paid_daily`
  where
    event_date between date(single_day) and (date(single_day) + interval conversion_window day)
    and user_id is not null
  group by user_id, platform
), hau as (
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
    hau.user_pseudo_id,
    hau.hau,
    hau.traffic_source,
    hau.traffic_source_name,
    utm.utm_source,
    utm.campaign,
  from hau
  full outer join utm
  on hau.user_pseudo_id = utm.user_pseudo_id
), trial_paid as (
  select
    trials.user_id,
    trials.user_pseudo_id,
    trials.platform,
    trials.event_date as trial_date,
    paid.event_date as paid_date,
    paid.proceeds as paid_proceeds
  from trials
  full outer join paid
  on trials.user_id = paid.user_id
)

select
  *
from trial_paid tp
left join hau_utm hu
on tp.user_pseudo_id = hu.user_pseudo_id
where
  tp.user_id is not null
  and hu.user_pseudo_id is not null
