insert into `relax-melodies-android.sandbox.analytics_events`

select
    *,
    timestamp(CURRENT_DATE("UTC")) as event_date_partitioned
from `relax-melodies-android.analytics_151587246.events_*`
where
    _table_suffix = FORMAT_DATE('%Y%m%d', CURRENT_DATE("UTC"))
