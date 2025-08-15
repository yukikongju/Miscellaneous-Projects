DECLARE start_date DATE DEFAULT '2023-01-01';
DECLARE end_date DATE DEFAULT '2025-01-01';
DECLARE renewal_threshold FLOAT64 DEFAULT 0.9; -- adjust as needed

FOR month_row IN (
  SELECT month_start
  FROM UNNEST(GENERATE_DATE_ARRAY(start_date, end_date, INTERVAL 1 MONTH)) AS month_start
) DO

  INSERT INTO `relax-melodies-android.late_conversions.monthly_renewal_rates`
  WITH cohorts_ranked AS (
    SELECT
      FORMAT_DATE('%Y-%m', month_row.month_start) AS year_month,
      *,
      RANK() OVER (PARTITION BY network, platform, country_code, renewal_bucket
                   ORDER BY paid_year_month DESC) AS rn
    FROM `relax-melodies-android.late_conversions.mature_renewal_cohorts`
    WHERE
      ((renewal_bucket = '1-Year'
        AND paid_year_month < FORMAT_DATE('%Y-%m', DATE_SUB(month_row.month_start, INTERVAL 1 YEAR)))
      OR (renewal_bucket = '2-Years'
        AND paid_year_month < FORMAT_DATE('%Y-%m', DATE_SUB(month_row.month_start, INTERVAL 2 YEAR)))
      OR (renewal_bucket = '3-Years'
        AND paid_year_month < FORMAT_DATE('%Y-%m', DATE_SUB(month_row.month_start, INTERVAL 3 YEAR))))
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
        renewal_percentage
      FROM latest_renewal_rates
    )
    PIVOT (
      AVG(renewal_percentage)
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
      END AS `3-Years`,
      current_timestamp() as loaded_timestamp,
    FROM pivot_table
  )
  SELECT * FROM pivot_table_imputed;

END FOR;
