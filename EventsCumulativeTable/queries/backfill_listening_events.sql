--- from analytics table
SELECT
  event_date, event_timestamp, event_name
FROM `relax-melodies-android.analytics_151587246.events_*`
WHERE _table_suffix >= '20250626' and _table_suffix <= '20250627'
  and event_name = 'listening';

--- from partitioned analytics table - single day
SELECT * FROM `relax-melodies-android.sandbox.analytics_events_20250327_20250610_partitioned` 
WHERE TIMESTAMP_TRUNC(event_date_partition, DAY) = TIMESTAMP("2025-06-10") 
  and event_name = 'listening';

--- from partitioned analytics table - within range
insert into `relax-melodies-android.test_cumulative_events_table.listening_events_partitioned` 

SELECT * FROM `relax-melodies-android.sandbox.analytics_events_20250327_20250610_partitioned` 
WHERE 
    TIMESTAMP_TRUNC(event_date_partition, DAY) >= TIMESTAMP("2025-06-01") 
    and TIMESTAMP_TRUNC(event_date_partition, DAY) <= TIMESTAMP("2025-06-09") 
    and event_name = 'listening';




