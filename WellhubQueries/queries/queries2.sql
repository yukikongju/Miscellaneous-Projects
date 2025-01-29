declare start_date STRING DEFAULT '20240912';
declare end_date STRING DEFAULT '20240912';

with firebase_users as (
  SELECT
    distinct user_pseudo_id
    , max(user_id) as user_id
    , max(case when param.key = 'em' then param.value.string_value end) as email
  FROM
    `relax-melodies-android.analytics_151587246.events_*`
  , UNNEST (event_params) AS param
  WHERE
    _table_suffix >=  start_date
    AND _table_suffix <= end_date
    AND event_name IN (
      'component_viewed',
      'stop_recorder',
      'edit_alarm',
      'activity_time_saved',
      'screen_content',
      'screen_content_playing',
      'screen_mixer',
      'screen_sounds',
      'screen_music',
      'screen_mixes',
      'screen_recorder',
      'play_recording',
      'listening'
    )
    AND user_pseudo_id IS NOT NULL
    AND param.key in ('origin', 'external_id', 'em', 'app_platform')
    AND EXISTS (
      SELECT 1
      FROM UNNEST(event_params) AS param
      WHERE
        (param.key = 'origin' AND param.value.string_value IS NOT NULL)
    )
    AND EXISTS (
      SELECT 1
      FROM UNNEST(event_params) AS param
      WHERE
        (param.key = 'em' and param.value.string_value is not null)
    )
  GROUP BY
    user_pseudo_id
  order by user_pseudo_id, user_id, email
), screen_home_counts as (
SELECT
  user_id, user_pseudo_id
  , count(distinct event_date) AS distinct_event_days
FROM
  `relax-melodies-android.analytics_151587246.events_*`
, UNNEST (event_params) AS ep1
WHERE
  _table_suffix >= start_date
  AND _table_suffix <= end_date
  AND event_name IN ('screen_home')
  AND user_pseudo_id IS NOT NULL
  AND user_id is not null
  -- AND user_pseudo_id in (select user_pseudo_id from firebase_users)
  AND user_id in (select user_id from firebase_users)
GROUP BY user_id, user_pseudo_id
ORDER BY user_id, user_pseudo_id
), wellhub_usage as (
  select
    a.user_id,
    a.user_pseudo_id,
    a.email,
    b.distinct_event_days
  from firebase_users a
  join screen_home_counts b on a.user_id = b.user_id
)

SELECT *
FROM wellhub_usage
