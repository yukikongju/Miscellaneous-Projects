-- Wellhub users monthly usage using the new events partitioned table
-- cost for 2 months: 113.41 GB

DECLARE start_date timestamp DEFAULT timestamp('2026-01-01');
DECLARE end_date timestamp DEFAULT timestamp('2026-02-28');

WITH firebase_users AS (
  SELECT
    DISTINCT user_pseudo_id,
    MAX(user_id) AS user_id,
    MAX(CASE WHEN param.key = 'em' THEN param.value.string_value END) AS email
  FROM
    -- `relax-melodies-android.analytics_151587246.events_*`,
    `relax-melodies-android.sandbox.analytics_events_pc`,
    UNNEST(event_params) AS param
  WHERE
    event_date_partition BETWEEN start_date AND end_date
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
    -- `relax-melodies-android.analytics_151587246.events_*`,
    `relax-melodies-android.sandbox.analytics_events_pc`,
    UNNEST(event_params) AS param
  WHERE
    event_date_partition BETWEEN start_date AND end_date
    AND event_name = 'screen_home'
    AND user_pseudo_id IS NOT NULL
    AND user_id IS NOT NULL
    AND user_id IN (SELECT distinct user_id FROM firebase_users)
  GROUP BY user_id, user_pseudo_id, event_month
),
wellhub_usage AS (
  SELECT
    f.user_id,
    f.user_pseudo_id,
    f.email,
    COALESCE(SUM(CASE WHEN s.event_month = '2026-01' THEN s.distinct_event_days END), 0) AS create_records_jan2026,
    COALESCE(SUM(CASE WHEN s.event_month = '2026-02' THEN s.distinct_event_days END), 0) AS create_records_feb2026,
    COALESCE(COUNT(CASE WHEN s.event_month = '2026-01' THEN s.distinct_event_days END), 0) AS DAU_jan2026,
    COALESCE(COUNT(CASE WHEN s.event_month = '2026-02' THEN s.distinct_event_days END), 0) AS DAU_feb2026,
  FROM firebase_users f
  LEFT JOIN screen_home_counts s ON f.user_id = s.user_id
  GROUP BY f.user_id, f.user_pseudo_id, f.email
)

SELECT *
FROM wellhub_usage
ORDER BY user_id;
