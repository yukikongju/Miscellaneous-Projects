SELECT
  date_trunc(creation_time, DAY) as day
  , sum(total_bytes_processed) as total_bytes_processed
  , ROUND(sum(total_bytes_processed / 1024 / 1024 / 1024 / 1024), 4) AS tb_processed
  , ROUND(sum(total_bytes_processed / 1024 / 1024 / 1024 / 1024 * 5), 2) AS cost_usd
FROM
    `region-us`.INFORMATION_SCHEMA.JOBS,
    UNNEST(labels) AS labels
WHERE
    creation_time >= '2025-11-15'
    AND labels.key like '%looker%'
    -- IN ('looker_studio_report_id', 'looker_studio_datasource_id')
GROUP BY day
ORDER BY
    day DESC;
