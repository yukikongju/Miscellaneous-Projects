DECLARE start_date STRING DEFAULT '20240901';
DECLARE end_date STRING DEFAULT '20250131';

WITH firebase_users AS (
  SELECT
    DISTINCT user_pseudo_id,
    MAX(user_id) AS user_id,
    MAX(CASE WHEN param.key = 'em' THEN param.value.string_value END) AS email
  FROM
    `relax-melodies-android.analytics_151587246.events_*`,
    UNNEST(event_params) AS param
  WHERE
    _table_suffix BETWEEN start_date AND end_date
    AND event_name IN (
      'component_viewed', 'stop_recorder', 'edit_alarm', 'activity_time_saved',
      'screen_content', 'screen_content_playing', 'screen_mixer', 'screen_sounds',
      'screen_music', 'screen_mixes', 'screen_recorder', 'play_recording', 'listening'
    )
    AND user_pseudo_id IS NOT NULL
    AND param.key IN ('origin', 'external_id', 'em', 'app_platform')
    AND EXISTS (
      SELECT 1
      FROM UNNEST(event_params) AS param
      WHERE param.key = 'origin' AND param.value.string_value IS NOT NULL
    )
    AND EXISTS (
      SELECT 1
      FROM UNNEST(event_params) AS param
      WHERE param.key = 'em' AND param.value.string_value IS NOT NULL
    )
  GROUP BY user_pseudo_id
),

screen_home_counts AS (
  SELECT
    user_id,
    user_pseudo_id,
    FORMAT_DATE('%Y-%m', DATE(TIMESTAMP_MICROS(event_timestamp))) AS event_month,
    COUNT(DISTINCT DATE(TIMESTAMP_MICROS(event_timestamp))) AS distinct_event_days
  FROM
    `relax-melodies-android.analytics_151587246.events_*`
  WHERE
    _table_suffix BETWEEN start_date AND end_date
    AND event_name = 'screen_home'
    AND user_pseudo_id IS NOT NULL
    AND user_id IS NOT NULL
    AND user_id IN (SELECT user_id FROM firebase_users)
  GROUP BY user_id, user_pseudo_id, event_month
),

wellhub_usage AS (
  SELECT
    f.user_id,
    f.user_pseudo_id,
    f.email,
    COALESCE(SUM(CASE WHEN s.event_month = '2024-09' THEN s.distinct_event_days END), 0) AS sep2024,
    COALESCE(SUM(CASE WHEN s.event_month = '2024-10' THEN s.distinct_event_days END), 0) AS oct2024,
    COALESCE(SUM(CASE WHEN s.event_month = '2024-11' THEN s.distinct_event_days END), 0) AS nov2024,
    COALESCE(SUM(CASE WHEN s.event_month = '2024-12' THEN s.distinct_event_days END), 0) AS dec2024,
    COALESCE(SUM(CASE WHEN s.event_month = '2025-01' THEN s.distinct_event_days END), 0) AS jan2025
  FROM firebase_users f
  LEFT JOIN screen_home_counts s ON f.user_id = s.user_id
  GROUP BY f.user_id, f.user_pseudo_id, f.email
)

SELECT *
FROM wellhub_usage
ORDER BY user_id;
