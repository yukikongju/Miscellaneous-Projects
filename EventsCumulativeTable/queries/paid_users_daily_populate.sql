insert into `relax-melodies-android.test_cumulative_events_table.paid_users_daily`

SELECT
  parse_date('%Y%m%d', event_date) as snapshot_date,
  count(distinct user_id) as num_paid,
  current_timestamp() as create_at,
FROM `relax-melodies-android.backend.events`
WHERE
  TIMESTAMP_TRUNC(event_timestamp_s, DAY) = TIMESTAMP(CURRENT_DATE())
  and event_name = 'subscription_start_paid'
GROUP BY event_date
