-- cost
DECLARE default_t2p_rate FLOAT64 DEFAULT 0.11;
DECLARE MIN_PAID_CONVERSION INT64 DEFAULT 5;

WITH default_country AS (
  SELECT
    DATE_TRUNC(date, ISOWEEK) AS week_start,
    country,
    SUM(trials) AS trials,
    SUM(paid) AS paid,
    SAFE_DIVIDE(SUM(paid), SUM(trials)) AS t2p
  FROM `relax-melodies-android.ua_dashboard_prod.pre_final_view`
  GROUP BY week_start, country
),
default_country_rolling AS (
  SELECT
    week_start,
    country,
    trials,
    paid,
    t2p,
    SUM(paid) OVER (
      PARTITION BY country ORDER BY week_start DESC
      ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
    ) AS rolling_paid,
    SUM(trials) OVER (
      PARTITION BY country ORDER BY week_start DESC
      ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
    ) AS rolling_trials,
    SUM(paid) OVER (
      PARTITION BY country ORDER BY week_start DESC
      ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
    ) / NULLIF(
      SUM(trials) OVER (
        PARTITION BY country ORDER BY week_start DESC
        ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
      ), 0
    ) AS t2p_rolling
  FROM default_country
),
default_network_country AS (
  SELECT
    DATE_TRUNC(date, ISOWEEK) AS week_start,
    network,
    country,
    SUM(trials) AS trials,
    SUM(paid) AS paid,
    SAFE_DIVIDE(SUM(paid), SUM(trials)) AS t2p
  FROM `relax-melodies-android.ua_dashboard_prod.pre_final_view`
  GROUP BY week_start, network, country
),
default_network_country_rolling AS (
  SELECT
    week_start,
    network,
    country,
    trials,
    paid,
    t2p,
    SUM(paid) OVER (
      PARTITION BY network, country ORDER BY week_start DESC
      ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
    ) AS rolling_paid,
    SUM(trials) OVER (
      PARTITION BY network, country ORDER BY week_start DESC
      ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
    ) AS rolling_trials,
    SUM(paid) OVER (
      PARTITION BY network, country ORDER BY week_start DESC
      ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
    ) / NULLIF(
      SUM(trials) OVER (
        PARTITION BY network, country ORDER BY week_start DESC
        ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
      ), 0
    ) AS t2p_rolling
  FROM default_network_country
),
default_platform_country AS (
  SELECT
    DATE_TRUNC(date, ISOWEEK) AS week_start,
    platform,
    country,
    SUM(trials) AS trials,
    SUM(paid) AS paid,
    SAFE_DIVIDE(SUM(paid), SUM(trials)) AS t2p
  FROM `relax-melodies-android.ua_dashboard_prod.pre_final_view`
  GROUP BY week_start, platform, country
),
default_platform_country_rolling AS (
  SELECT
    week_start,
    platform,
    country,
    trials,
    paid,
    t2p,
    SUM(paid) OVER (
      PARTITION BY platform, country ORDER BY week_start DESC
      ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
    ) AS rolling_paid,
    SUM(trials) OVER (
      PARTITION BY platform, country ORDER BY week_start DESC
      ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
    ) AS rolling_trials,
    SUM(paid) OVER (
      PARTITION BY platform, country ORDER BY week_start DESC
      ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
    ) / NULLIF(
      SUM(trials) OVER (
        PARTITION BY platform, country ORDER BY week_start DESC
        ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
      ), 0
    ) AS t2p_rolling
  FROM default_platform_country
),
default_network_platform_country AS (
  SELECT
    DATE_TRUNC(date, ISOWEEK) AS week_start,
    network,
    platform,
    country,
    SUM(trials) AS trials,
    SUM(paid) AS paid,
    SUM(revenues) AS revenues,
    SAFE_DIVIDE(SUM(paid), SUM(trials)) AS t2p
  FROM `relax-melodies-android.ua_dashboard_prod.pre_final_view`
  GROUP BY week_start, network, platform, country
),
default_network_platform_country_rolling AS (
  SELECT
    week_start,
    network,
    platform,
    country,
    trials,
    paid,
    revenues,
    t2p,
    SUM(paid) OVER (
      PARTITION BY network, platform, country ORDER BY week_start DESC
      ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
    ) AS rolling_paid,
    SUM(trials) OVER (
      PARTITION BY network, platform, country ORDER BY week_start DESC
      ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
    ) AS rolling_trials,
    AVG(NULLIF(t2p, 0.0)) OVER (
      PARTITION BY network, platform, country
      ORDER BY week_start DESC
      ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
    ) AS t2p_rolling_rate,
    SUM(paid) OVER (
      PARTITION BY network, platform, country ORDER BY week_start DESC
      ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
    ) / NULLIF(
      SUM(trials) OVER (
        PARTITION BY network, platform, country ORDER BY week_start DESC
        ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
      ), 0
    ) AS t2p_rolling_sum
  FROM default_network_platform_country
),
t2p_rolling AS (
  SELECT
    npc.week_start,
    npc.network,
    npc.platform,
    npc.country,
    npc.rolling_paid AS paid_npc,
    pc.rolling_paid AS paid_pc,
    nc.rolling_paid AS paid_nc,
    dc.rolling_paid AS paid_dc,
    npc.t2p_rolling_sum AS t2p_npc,
    pc.t2p_rolling AS t2p_pc,
    nc.t2p_rolling AS t2p_nc,
    dc.t2p_rolling AS t2p_dc,
    CASE
      WHEN npc.rolling_paid > MIN_PAID_CONVERSION THEN npc.t2p_rolling_sum
      WHEN nc.rolling_paid > MIN_PAID_CONVERSION THEN nc.t2p_rolling
      WHEN pc.rolling_paid > MIN_PAID_CONVERSION THEN pc.t2p_rolling
      WHEN dc.rolling_paid > MIN_PAID_CONVERSION THEN dc.t2p_rolling
      ELSE default_t2p_rate
    END AS t2p_estimation
  FROM default_network_platform_country_rolling npc
  LEFT JOIN default_platform_country_rolling pc
    ON npc.week_start = pc.week_start AND npc.platform = pc.platform AND npc.country = pc.country
  LEFT JOIN default_network_country_rolling nc
    ON npc.week_start = nc.week_start AND npc.network = nc.network AND npc.country = nc.country
  LEFT JOIN default_country_rolling dc
    ON npc.week_start = dc.week_start AND npc.country = dc.country
)

SELECT *
FROM t2p_rolling
WHERE
  network IN ('Facebook Ads', 'googleadwords_int', 'Apple Search Ads')
  AND country = 'CA'
  AND week_start >= '2025-08-01'
  AND week_start < '2025-09-01'
ORDER BY platform, country, network, week_start;
