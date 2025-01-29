SELECT
  user_pseudo_id
  , user_id
  , max(device.operating_system) as os
  , max(case when param.key = 'em' then param.value.string_value end) as email
  , max(case when param.key = 'origin' then param.value.string_value end) as origin
  , sum(case when event_name = 'play_content' then 1 else 0 end) as play_content
  , SUM(CASE WHEN event_name = 'activity_time_saved' THEN 1 ELSE 0 END) AS activity_time_saved
  , SUM(CASE WHEN event_name = 'screen_content' THEN 1 ELSE 0 END) AS screen_content
  , SUM(CASE WHEN event_name = 'screen_content_playing' THEN 1 ELSE 0 END) AS screen_content_playing
  , SUM(CASE WHEN event_name = 'screen_mixer' THEN 1 ELSE 0 END) AS screen_mixer
  , SUM(CASE WHEN event_name = 'screen_sounds' THEN 1 ELSE 0 END) AS screen_sounds
  , SUM(CASE WHEN event_name = 'screen_music' THEN 1 ELSE 0 END) AS screen_music
  , SUM(CASE WHEN event_name = 'screen_mixes' THEN 1 ELSE 0 END) AS screen_mixes
  , SUM(CASE WHEN event_name = 'screen_recorder' THEN 1 ELSE 0 END) AS screen_recorder
  , SUM(CASE WHEN event_name = 'play_recording' THEN 1 ELSE 0 END) AS play_recording
  , SUM(CASE WHEN event_name = 'listening' THEN 1 ELSE 0 END) AS listening
FROM
  `relax-melodies-android.analytics_151587246.events_*`
, UNNEST (event_params) AS param
WHERE
  -- _table_suffix >= '20240901'
  -- AND _table_suffix <= '20250130'
  _table_suffix >= '20240912'
  AND _table_suffix <= '20240912'
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
    FROM UNNEST(event_params) AS origin_param
    WHERE origin_param.key = 'origin' AND origin_param.value.string_value IS NOT NULL
  )
GROUP BY
  user_pseudo_id, user_id
