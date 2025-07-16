INSERT INTO `relax-melodies-android.late_conversions.users_utm_daily`

WITH daily_utm AS (
  SELECT
    DATE(TIMESTAMP_MICROS(event_timestamp)) AS event_date_partition,
    event_timestamp,
    user_id,
    user_pseudo_id,
    LOWER(platform) AS platform,
    geo.country AS country,
    traffic_source.name AS traffic_source_name,
    traffic_source.medium AS traffic_source_medium,
    traffic_source.source AS traffic_source,
    CASE WHEN ep.key = 'campaign_id' THEN ep.value.string_value END AS campaign,
    CASE WHEN ep.key IN ('utm_source', 'source') THEN LOWER(ep.value.string_value) END AS utm_source,
    CASE WHEN ep.key = 'page_title' THEN ep.value.string_value END AS page_title,
    CASE WHEN ep.key = 'page_location' THEN ep.value.string_value END AS page_location,
    CASE WHEN ep.key = 'page_referrer' THEN ep.value.string_value END AS page_referrer,
    CASE WHEN ep.key = 'content' THEN ep.value.string_value END AS content,
	current_timestamp()
  FROM `relax-melodies-android.sandbox.analytics_events_pc`,
    UNNEST(event_params) AS ep
  WHERE
    -- DATE(event_date_partition) = DATE_SUB(current_date(), interval 2 day)
    --  TIMESTAMP_TRUNC(event_date_partition, DAY) = TIMESTAMP(CURRENT_DATE() - interval 2 day)
    timestamp_trunc(event_date_partition, DAY) >= TIMESTAMP("2025-01-01")
    and timestamp_trunc(event_date_partition, DAY) <= TIMESTAMP("2025-07-13")
    AND event_name = 'UTM_Visited'
    AND ep.key IN ('campaign_id', 'utm_source', 'source', 'content', 'page_location', 'page_referrer')
    AND user_pseudo_id IS NOT NULL
)

SELECT * FROM daily_utm;

--  Notes
--  => populated for "2025-06-01"
