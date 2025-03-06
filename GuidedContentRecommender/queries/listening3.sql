DECLARE start_date STRING DEFAULT '20250224';
DECLARE end_date STRING DEFAULT '20250224';

# --- TMP

WITH unpacked AS (
  SELECT
    event_date,
    user_pseudo_id,
    MAX(CASE WHEN param.key = 'guided_content' THEN param.value.string_value END) AS guided_content
  FROM `relax-melodies-android.analytics_151587246.events_*`,
    UNNEST(event_params) AS param
  WHERE
    _table_suffix BETWEEN start_date AND end_date
    AND event_name = 'listening'
  GROUP BY event_date, user_pseudo_id
)
SELECT
  event_date,
  user_pseudo_id,
  guided_content,
  COUNT(*) AS occurrence_count
FROM unpacked
GROUP BY event_date, user_pseudo_id, guided_content
ORDER BY occurrence_count DESC
LIMIT 100;
