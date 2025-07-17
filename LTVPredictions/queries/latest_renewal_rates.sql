WITH
  rates_ma AS (
  SELECT
    paid_year_month,
    platform,
    ROUND(AVG(`1-Year`) OVER (PARTITION BY platform ORDER BY paid_year_month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2) AS `1-Year`,
    ROUND(AVG(`2-Years`) OVER (PARTITION BY platform ORDER BY paid_year_month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2) AS `2-Years`,
    ROUND(AVG(`3-Years`) OVER (PARTITION BY platform ORDER BY paid_year_month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2) AS `3-Years`,
    ROUND(AVG(`4-Years`) OVER (PARTITION BY platform ORDER BY paid_year_month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2) AS `4-Years`,
    ROUND(AVG(`5-Years`) OVER (PARTITION BY platform ORDER BY paid_year_month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2) AS `5-Years`,
    ROUND(AVG(`>5-Years`) OVER (PARTITION BY platform ORDER BY paid_year_month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2) AS `>5-Years`,
  FROM
    `relax-melodies-android.late_conversions.renewal_rates_cohorted` ),
  last_valid_rates AS (
  SELECT
    platform,
    LAST_VALUE(`1-Year`) OVER (PARTITION BY platform ORDER BY CASE WHEN PARSE_DATE('%Y-%m', paid_year_month) < DATE_SUB(CURRENT_DATE(), INTERVAL 1 YEAR) THEN paid_year_month
    END
      ROWS BETWEEN UNBOUNDED PRECEDING
      AND UNBOUNDED FOLLOWING) AS `latest_1Year_renewal_rate`,
    LAST_VALUE(`2-Years`) OVER (PARTITION BY platform ORDER BY CASE WHEN PARSE_DATE('%Y-%m', paid_year_month) < DATE_SUB(CURRENT_DATE(), INTERVAL 2 YEAR) THEN paid_year_month
    END
      ROWS BETWEEN UNBOUNDED PRECEDING
      AND UNBOUNDED FOLLOWING) AS `latest_2Year_renewal_rate`,
    LAST_VALUE(`3-Years`) OVER (PARTITION BY platform ORDER BY CASE WHEN PARSE_DATE('%Y-%m', paid_year_month) < DATE_SUB(CURRENT_DATE(), INTERVAL 3 YEAR) THEN paid_year_month
    END
      ROWS BETWEEN UNBOUNDED PRECEDING
      AND UNBOUNDED FOLLOWING) AS `latest_3Year_renewal_rate`,
    LAST_VALUE(`4-Years`) OVER (PARTITION BY platform ORDER BY CASE WHEN PARSE_DATE('%Y-%m', paid_year_month) < DATE_SUB(CURRENT_DATE(), INTERVAL 4 YEAR) THEN paid_year_month
    END
      ROWS BETWEEN UNBOUNDED PRECEDING
      AND UNBOUNDED FOLLOWING) AS `latest_4Year_renewal_rate`,
    LAST_VALUE(`5-Years`) OVER (PARTITION BY platform ORDER BY CASE WHEN PARSE_DATE('%Y-%m', paid_year_month) < DATE_SUB(CURRENT_DATE(), INTERVAL 5 YEAR) THEN paid_year_month
    END
      ROWS BETWEEN UNBOUNDED PRECEDING
      AND UNBOUNDED FOLLOWING) AS `latest_5Year_renewal_rate`,
    LAST_VALUE(`>5-Years`) OVER (PARTITION BY platform ORDER BY CASE WHEN PARSE_DATE('%Y-%m', paid_year_month) < DATE_SUB(CURRENT_DATE(), INTERVAL 6 YEAR) THEN paid_year_month
    END
      ROWS BETWEEN UNBOUNDED PRECEDING
      AND UNBOUNDED FOLLOWING) AS `latest_5Year_renewal_rate`,
  FROM
    rates_ma )
SELECT
  *
FROM
  last_valid_rates
