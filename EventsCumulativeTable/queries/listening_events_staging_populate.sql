--- table: `relax-melodies-android.test_cumulative_events_table.listening_events_staging`

declare current_date_ts timestamp default TIMESTAMP("2025-06-10");
-- declare current_date_ts timestamp default TIMESTAMP(CURRENT_DATE());

SELECT
  event_date_partition as event_date,
  event_timestamp,
  user_id,
  user_pseudo_id,
  geo.country as country,
  platform,
  case when ep.key = 'sounds' then ep.value.string_value end as sounds,
  case when ep.key = 'mix_type' then ep.value.string_value end as mix_type,
  case when ep.key = 'guided content' then ep.value.string_value end as guided_content,
  case when ep.key = 'screen' then ep.value.string_value end as screen,
  case when ep.key = 'sounds_selected' then ep.value.string_value end as sounds_selected,
  case when ep.key = 'brainwaves_selected' then ep.value.string_value end as brainwaves_selected,
  case when ep.key = 'sounds_volume' then ep.value.string_value end as sounds_volume,
  case when ep.key = 'content_volume' then ep.value.string_value end as content_volume,
FROM `relax-melodies-android.test_cumulative_events_table.listening_events_partitioned`,
  unnest(event_params) as ep
WHERE
  ep.key in ('sounds', 'mix_type', 'guided content','screen', 'sounds_selected', 'brainwaves_selected', 'sounds_volume', 'content_volume')
  -- and TIMESTAMP_TRUNC(event_date_partition, DAY) = current_date_ts
