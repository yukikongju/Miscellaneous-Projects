SELECT distinct(event_name)
FROM `relax-melodies-android.sandbox.analytics_events_20250327_20250610_partitioned`
WHERE TIMESTAMP_TRUNC(event_date_partition, DAY) = TIMESTAMP("2025-06-01")
order by event_name

--- 865 events
--- ~/Downlaods/bquxjob_59b8d27a_197bf42c5c4.csv
--- screen_content_playing
--- scren_goals
--- listening_session
