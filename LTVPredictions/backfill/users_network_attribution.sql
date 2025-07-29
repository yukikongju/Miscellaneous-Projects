-- cost: 624.27MB ; 741.63MB => approx. 1GB per day ie 500GB ie 50 cents per year
DECLARE
  conversion_window INT64 DEFAULT 20;

-- DECLARE start_date string DEFAULT '2023-01-01';
-- DECLARE end_date string DEFAULT  '2025-07-22';
  -- insert into `relax-melodies-android.late_conversions.users_network_attribution`
WITH
  hau AS (
  SELECT
    user_id,
    user_pseudo_id,
    LOWER(platform) AS platform,
    geo.country AS country,
    traffic_source.name AS traffic_source_name,
    traffic_source.medium AS traffic_source_medium,
    traffic_source.source AS traffic_source,
    JSON_VALUE(PARSE_JSON(ep2.value.string_value), '$[0]') AS hau,
    event_timestamp
  FROM
    `relax-melodies-android.sandbox.analytics_events_pc`,
    UNNEST(event_params) AS ep1,
    UNNEST(event_params) AS ep2
  WHERE
    -- TIMESTAMP_TRUNC(event_date_partition, DAY) >= timestamp(start_date) and
    -- TIMESTAMP_TRUNC(event_date_partition, DAY) <= timestamp(end_date)
    event_date_partition >= '2023-01-01' and event_date_partition <= '2025-07-22'
    AND event_name = 'answer_question'
    AND ep1.key IN ('question_id')
    AND LOWER(ep1.value.string_value) = 'hearaboutus'
    AND ep2.key IN ('answers',
      'answer')
    AND user_pseudo_id IS NOT NULL),
  utm AS (
  SELECT
    -- DATE(TIMESTAMP_MICROS(event_timestamp)) AS event_date_partition,
    user_id,
    user_pseudo_id,
    LOWER(platform) AS platform,
    geo.country AS country,
    traffic_source.name AS traffic_source_name,
    traffic_source.medium AS traffic_source_medium,
    traffic_source.source AS traffic_source,
    CASE
      WHEN ep.key = 'campaign_id' THEN ep.value.string_value
  END
    AS campaign,
    CASE
      WHEN ep.key IN ('utm_source', 'source') THEN LOWER(ep.value.string_value)
  END
    AS utm_source,
    -- CASE WHEN ep.key = 'page_title' THEN ep.value.string_value END AS page_title,
    -- CASE WHEN ep.key = 'page_location' THEN ep.value.string_value END AS page_location,
    -- CASE WHEN ep.key = 'page_referrer' THEN ep.value.string_value END AS page_referrer,
    -- CASE WHEN ep.key = 'content' THEN ep.value.string_value END AS content,
    event_timestamp
  FROM
    `relax-melodies-android.sandbox.analytics_events_pc`,
    UNNEST(event_params) AS ep
  WHERE
    event_date_partition >= timestamp_sub('2023-01-01', interval conversion_window day)
    AND event_date_partition <= '2025-07-22'
    AND event_name = 'UTM_Visited'
    AND ep.key IN ('campaign_id',
      'utm_source',
      'source',
      'content',
      'page_location',
      'page_referrer')
    AND user_pseudo_id IS NOT NULL ),
  hau_clean AS (
  SELECT
    h.user_id,
    h.user_pseudo_id,
    h.platform,
    h.country,
    h.traffic_source,
    h.traffic_source_name,
    h.hau AS old_hau,
    CASE
      WHEN m.new_hau IS NOT NULL THEN m.new_hau
      ELSE h.hau
  END
    AS hau,
    h.event_timestamp
  FROM
    hau h
  LEFT JOIN
    `relax-melodies-android.mappings.hau_maps` m
  ON
    h.hau = m.original_hau ),
  utm_clean AS (
  SELECT
    u.user_id,
    u.user_pseudo_id,
    u.platform,
    u.country,
    u.utm_source AS old_utm_source,
    CASE
      WHEN m.new_utm IS NOT NULL THEN m.new_utm
      ELSE u.utm_source
  END
    AS utm_source,
    u.campaign,
    u.event_timestamp,
  FROM
    utm u
  LEFT JOIN
    `relax-melodies-android.mappings.utm_maps` m
  ON
    u.utm_source = m.original_utm
  WHERE
    u.utm_source IS NOT NULL ),
  hau_utm AS (
  SELECT
    hau.user_id,
    hau.user_pseudo_id,
    hau.hau,
    hau.traffic_source,
    hau.traffic_source_name,
    hau.old_hau,
    hau.country,
    hau.platform AS platform,
    -- hau.platform AS hau_platform,
    -- utm.platform AS utm_platform,
    utm.old_utm_source,
    utm.utm_source,
    utm.campaign,
    hau.event_timestamp AS hau_timestamp,
    utm.event_timestamp AS utm_timestamp
  FROM
    hau_clean hau
  FULL OUTER JOIN
    utm_clean utm
  ON
    hau.user_pseudo_id = utm.user_pseudo_id
    AND hau.user_id = utm.user_id
  WHERE
    hau.user_id IS NOT NULL
    AND hau.user_pseudo_id IS NOT NULL ),
  users_network_attribution AS (
  SELECT
    user_id,
    user_pseudo_id,
    MAX(platform) AS platform,
    MAX(country) AS country,
    -- most of the time, hau_platform and utm_platform match, but utm_platform is sometimes missing
    -- max(hau_platform) as hau_platform,
    -- max(utm_platform) as utm_platform,
    MAX(hau) AS hau,
    MAX(traffic_source) AS traffic_source,
    MAX(traffic_source_name) AS traffic_source_name,
    MAX(utm_source) AS utm_source,
    MAX(campaign) AS campaign,
    MAX(old_utm_source) AS old_utm_source,
    MAX(old_hau) AS old_hau,
    CASE
      WHEN MAX(utm_source) != 'no user consent' THEN MAX(utm_source) --- cast(coalesce(json_value(parse_json(max(utm_source)), '$[0]'), max(utm_source)) as string)
      WHEN MAX(utm_source) = 'no user consent'
    AND MAX(hau) = 'no answer' THEN 'unavailable'
      ELSE MAX(hau)
  END
    AS network_attribution,
    MAX(hau_timestamp) AS hau_timestamp,
    MAX(utm_timestamp) AS utm_timestamp,
  FROM
    hau_utm
  GROUP BY
    user_id,
    user_pseudo_id )


  -- select * from users_network_attribution
  -- select * from hau


SELECT
  u.user_id,
  u.user_pseudo_id,
  u.platform,
  u.country AS country_name,
  m.country_code,
  u.traffic_source,
  u.traffic_source_name,
  u.campaign,
  u.old_hau,
  u.old_utm_source,
  u.hau,
  u.utm_source,
  u.network_attribution,
  u.hau_timestamp,
  u.utm_timestamp,
  CURRENT_TIMESTAMP() as load_timestamp
FROM
  users_network_attribution u
LEFT JOIN
  `relax-melodies-android.mappings.country_maps` m
ON
  u.country = m.country_name
