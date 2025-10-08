select
  *
from `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
where
  creation_time >= '2025-10-01'
  and destination_table.dataset_id = 'ua_dashboard_prod'
  and destination_table.table_id like 'mixpanel%'
order by creation_time
