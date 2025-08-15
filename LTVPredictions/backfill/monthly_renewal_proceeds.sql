DECLARE start_date DATE DEFAULT '2023-01-01';
DECLARE end_date DATE DEFAULT '2025-08-01';
DECLARE renewal_threshold FLOAT64 DEFAULT 0.8;
DECLARE default_renewal_proceeds FLOAT64 DEFAULT 50.0;

FOR rec IN (
  SELECT month_start
  FROM UNNEST(GENERATE_DATE_ARRAY(start_date, end_date, INTERVAL 1 MONTH)) AS month_start
) DO
BEGIN

INSERT INTO `relax-melodies-android.late_conversions.monthly_renewal_proceeds`
WITH defaults_network_platform_country AS (
  SELECT
    network,
    platform,
    country_code,
    AVG(`1-Year`) AS avg_1y,
    AVG(`2-Years`) AS avg_2y,
    AVG(`3-Years`) AS avg_3y
  FROM `relax-melodies-android.late_conversions.monthly_renewal_proceeds`
  GROUP BY network, platform, country_code
), defaults_platform_country AS (
  SELECT
    platform,
    country_code,
    AVG(`1-Year`) AS avg_1y,
    AVG(`2-Years`) AS avg_2y,
    AVG(`3-Years`) AS avg_3y
  FROM  `relax-melodies-android.late_conversions.monthly_renewal_proceeds`
  GROUP BY platform, country_code
), defaults_country AS (
  SELECT
    country_code,
    AVG(`1-Year`) AS avg_1y,
    AVG(`2-Years`) AS avg_2y,
    AVG(`3-Years`) AS avg_3y
  FROM  `relax-melodies-android.late_conversions.monthly_renewal_proceeds`
  GROUP BY country_code
), cohorts_ranked AS (
  SELECT
    FORMAT_DATE('%Y-%m', rec.month_start) AS year_month,
    *,
    RANK() OVER (PARTITION BY network, platform, country_code, renewal_bucket
                 ORDER BY paid_year_month DESC) AS rn
  FROM `relax-melodies-android.late_conversions.mature_renewal_cohorts`
  WHERE
    (renewal_bucket = '1-Year' AND paid_year_month < FORMAT_DATE('%Y-%m', DATE_SUB(rec.month_start, INTERVAL 1 YEAR)))
    OR (renewal_bucket = '2-Years' AND paid_year_month < FORMAT_DATE('%Y-%m', DATE_SUB(rec.month_start, INTERVAL 2 YEAR)))
    OR (renewal_bucket = '3-Years' AND paid_year_month < FORMAT_DATE('%Y-%m', DATE_SUB(rec.month_start, INTERVAL 3 YEAR)))
), latest_renewal_rates AS (
  SELECT
    year_month,
    network,
    platform,
    country_code,
    renewal_bucket,
    num_renewals,
    num_paid,
    paid_proceeds,
    renewal_proceeds,
    renewal_percentage
  FROM cohorts_ranked
  WHERE rn = 1
), pivot_table AS (
  SELECT
    year_month,
    platform,
    network,
    country_code,
    `1-Year`,
    `2-Years`,
    `3-Years`
  FROM (
    SELECT
      year_month,
      platform,
      network,
      country_code,
      renewal_bucket,
      renewal_proceeds
    FROM latest_renewal_rates
  )
  PIVOT (
    AVG(renewal_proceeds)
    FOR renewal_bucket IN ('1-Year', '2-Years', '3-Years')
  )
), pivot_table_imputed AS (
  SELECT
    PARSE_DATE('%Y-%m', year_month) AS year_month,
    platform,
    network,
    country_code,
    CASE
      WHEN `1-Year` IS NULL OR `1-Year` > renewal_threshold
        THEN AVG(`1-Year`) OVER (PARTITION BY platform, country_code)
      ELSE `1-Year`
    END AS `1-Year`,
    CASE
      WHEN `2-Years` IS NULL OR `2-Years` > renewal_threshold
        THEN AVG(`2-Years`) OVER (PARTITION BY platform, country_code)
      ELSE `2-Years`
    END AS `2-Years`,
    CASE
      WHEN `3-Years` IS NULL OR `3-Years` > renewal_threshold
        THEN AVG(`3-Years`) OVER (PARTITION BY platform, country_code)
      ELSE `3-Years`
    END AS `3-Years`
  FROM pivot_table
), pivot_table_complete AS (
  SELECT * FROM pivot_table_imputed
  UNION ALL
  SELECT
    year_month, platform,
    'tatari_linear' AS network,
    country_code, `1-Year`, `2-Years`, `3-Years`
  FROM pivot_table_imputed
  WHERE network = 'tatari_streaming'
), keys AS (
  SELECT
    month_start,
    platform,
    network,
    c.country_code
  FROM UNNEST(GENERATE_DATE_ARRAY(rec.month_start, rec.month_start, INTERVAL 1 MONTH)) AS month_start
  CROSS JOIN UNNEST(['ios', 'android']) AS platform
  CROSS JOIN UNNEST([
    'Facebook Ads', 'tiktokglobal_int', 'snapchat_int', 'googleadwords_int',
    'tatari_streaming', 'tatari_linear', 'Apple Search Ads', 'Organic'
  ]) AS network
  CROSS JOIN (SELECT country_code FROM `relax-melodies-android.mappings.country_maps`) AS c
), complete_table AS (
  SELECT
    k.month_start,
    k.platform,
    k.network,
    k.country_code,
    COALESCE(ptc.`1-Year`, npc.avg_1y, pc.avg_1y, c.avg_1y, default_renewal_proceeds) AS `1-Year`,
    COALESCE(ptc.`2-Years`, npc.avg_2y, pc.avg_2y, c.avg_2y, default_renewal_proceeds) AS `2-Years`,
    COALESCE(ptc.`3-Years`, npc.avg_3y, pc.avg_3y, c.avg_3y, default_renewal_proceeds) AS `3-Years`,
    CURRENT_TIMESTAMP() AS loaded_timestamp
  FROM keys k
  LEFT JOIN pivot_table_complete ptc
    ON k.network = ptc.network
    AND k.platform = ptc.platform
    AND k.country_code = ptc.country_code
  LEFT JOIN defaults_network_platform_country npc
    ON k.network = npc.network
    AND k.platform = npc.platform
    AND k.country_code = npc.country_code
  LEFT JOIN defaults_platform_country pc
    ON k.platform = pc.platform
    AND k.country_code = pc.country_code
  LEFT JOIN defaults_country c
    ON k.country_code = c.country_code
)
SELECT * FROM complete_table;

END;
END FOR;
