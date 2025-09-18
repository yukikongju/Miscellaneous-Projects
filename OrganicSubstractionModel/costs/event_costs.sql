SELECT
  creation_time,
  user_email,
  job_type, statement_type,
  table_id,
  query,
  total_bytes_processed as total_bytes_processed,
  total_bytes_billed as total_bytes_billed,
  total_bytes_billed / 1e12 * 5 as total_cost_usd_billed
FROM
  `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT j,
  unnest(j.referenced_tables) as referenced_table
WHERE
  -- extract(date from creation_time) = '2025-07-17'
  extract(date from creation_time) = '2025-09-17'
  and referenced_table.table_id like '%event%'
  and lower(query) like '%with hau%'
