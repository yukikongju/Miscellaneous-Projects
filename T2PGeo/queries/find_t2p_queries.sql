select
  creation_time, job_type, query
from `relax-melodies-android.region-US.INFORMATION_SCHEMA.JOBS_BY_USER`
where
  user_email = '<EMAIL>'
  and creation_time >= '2025-09-01'
  and query like '%t2p%'
order by creation_time;
