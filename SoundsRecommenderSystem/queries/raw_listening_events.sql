-- EXAMPLE
-- user_id, date, sounds, sounds_volume
-- 021823C0-25FD-4737-97C6-5EFF52E4F5A4	2026-03-01	["ambience.rain", "ambience.river", "ambience.ocean", "ambience.brahmslullaby"]	["0.53", "0.31", "0.18", "1.00"]

-- Num Rows: 6,692,213
-- Cost: 6.01 GB
declare start_date timestamp default '2026-02-01';
declare end_date timestamp default '2026-03-01';


WITH listening_events AS (
  SELECT
    user_id,
    PARSE_DATE('%Y%m%d', event_date) AS date,
    (
      SELECT ep.value.string_value
      FROM UNNEST(event_params) ep
      WHERE ep.key = 'sounds'
    ) AS sounds,

    (
      SELECT ep.value.string_value
      FROM UNNEST(event_params) ep
      WHERE ep.key = 'sounds_volume'
    ) AS sounds_volume

  FROM `relax-melodies-android.sandbox.analytics_events_pc`
  WHERE
    event_date_partition between start_date and end_date
    AND event_name = 'listening'
),

normalized AS (
  SELECT
    date,
    user_id,
    sounds,
    sounds_volume,
    ARRAY_TO_STRING(
      ARRAY(
        SELECT sound
        FROM UNNEST(JSON_VALUE_ARRAY(sounds)) AS sound
        WHERE sound IS NOT NULL
        GROUP BY sound
        ORDER BY sound
      ),
      ' | '
    ) AS mix_signature
  FROM listening_events
  WHERE
    sounds IS NOT NULL and sounds != '[]'
    AND sounds_volume IS NOT NULL
    AND ARRAY_LENGTH(JSON_VALUE_ARRAY(sounds)) >= 2
)

SELECT
  user_id,
  date,
  sounds,
  sounds_volume
FROM normalized
WHERE mix_signature != 'ambience.ocean | ambience.birds | ambience.eternity';


-- Cost: 34.44 GB -- unoptimized
--  WITH listening_events AS (
  --  SELECT
    --  PARSE_DATE('%Y%m%d', event_date) AS date,

    --  (
      --  SELECT ep.value.string_value
      --  FROM UNNEST(event_params) ep
      --  WHERE ep.key = 'sounds'
    --  ) AS sounds,

    --  (
      --  SELECT ep.value.string_value
      --  FROM UNNEST(event_params) ep
      --  WHERE ep.key = 'sounds_volume'
    --  ) AS sounds_volume

  --  FROM `relax-melodies-android.analytics_151587246.events_*`
  --  WHERE
    --  _TABLE_SUFFIX BETWEEN '20260322' AND '20260323'
    --  AND event_name = 'listening'
--  ),

--  normalized AS (
  --  SELECT
    --  date,
    --  sounds,
    --  sounds_volume,
    --  ARRAY_TO_STRING(
      --  ARRAY(
        --  SELECT sound
        --  FROM UNNEST(JSON_VALUE_ARRAY(sounds)) AS sound
        --  WHERE sound IS NOT NULL
        --  GROUP BY sound
        --  ORDER BY sound
      --  ),
      --  ' | '
    --  ) AS mix_signature
  --  FROM listening_events
  --  WHERE
    --  sounds IS NOT NULL
    --  AND sounds_volume IS NOT NULL
--  )

--  SELECT
  --  date,
  --  sounds,
  --  sounds_volume
--  FROM normalized
--  WHERE mix_signature != 'ambience.ocean | ambience.birds | ambience.eternity';
