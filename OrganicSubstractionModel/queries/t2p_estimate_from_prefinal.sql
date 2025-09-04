--- cost: 700 MB
--- condition: min paid = 5 ; min trials: N/A
-- declare N int64 default 7;
declare default_t2p_rate float64 default 0.11;
declare MIN_PAID_CONVERSION int64 default 5;

WITH default_country as (
  select
    date, country,
    sum(trials) as trials,
    sum(paid) as paid,
    case when sum(trials) > 0
      then sum(paid) / sum(trials)
      else null
    end as t2p,
  from `relax-melodies-android.ua_dashboard_prod.pre_final_view`
  group by date, country
), default_country_rolling as (
  select
    date, country,
    trials, paid, t2p,
    sum(paid) over (partition by country order by date desc rows between 7 preceding and current row) as rolling_paid,
    sum(trials) over (partition by country order by date desc rows between 7 preceding and current row) as rolling_trials,
    (sum(paid) over (partition by country order by date desc rows between 7 preceding and current row)) / nullif(sum(trials) over (partition by country order by date desc rows between 7 preceding and current row), 0) as t2p_rolling
  from default_country
), default_network_country as (
  select
    date, network, country,
    sum(trials) as trials,
    sum(paid) as paid,
    case when sum(trials) > 0
      then sum(paid) / sum(trials)
      else null
    end as t2p
  from `relax-melodies-android.ua_dashboard_prod.pre_final_view`
  group by date, network, country
), default_network_country_rolling as (
  select
    date, network, country,
    trials, paid, t2p,
    sum(paid) over (partition by network, country order by date desc rows between 7 preceding and current row) as rolling_paid,
    sum(trials) over (partition by network, country order by date desc rows between 7 preceding and current row) as rolling_trials,
    (sum(paid) over (partition by network, country order by date desc rows between 7 preceding and current row)) / nullif(sum(trials) over (partition by network, country order by date desc rows between 7 preceding and current row), 0) as t2p_rolling
  from default_network_country
), default_platform_country as (
  select
    date, platform, country,
    sum(trials) as trials,
    sum(paid) as paid,
    case when sum(trials) > 0
      then sum(paid) / sum(trials)
      else null
    end as t2p
  from `relax-melodies-android.ua_dashboard_prod.pre_final_view`
  group by date, platform, country
), default_platform_country_rolling as (
  select
    date, platform, country,
    trials, paid, t2p,
    sum(paid) over (partition by platform, country order by date desc rows between 7 preceding and current row) as rolling_paid,
    sum(trials) over (partition by platform, country order by date desc rows between 7 preceding and current row) as rolling_trials,
    (sum(paid) over (partition by platform, country order by date desc rows between 7 preceding and current row)) / nullif(sum(trials) over (partition by platform, country order by date desc rows between 7 preceding and current row), 0) as t2p_rolling
  from default_platform_country
), default_network_platform_country as (
  SELECT
    `date`,
    network,
    platform,
    country,
    SUM(trials) AS trials,
    SUM(paid) AS paid,
    SUM(revenues) AS revenues,
    CASE
      WHEN SUM(trials) > 0 THEN SUM(paid) / SUM(trials)
      ELSE NULL
    END AS t2p
  FROM `relax-melodies-android.ua_dashboard_prod.pre_final_view`
  GROUP BY `date`, network, platform, country
), default_network_platform_country_rolling AS (
  SELECT
    `date`,
    network,
    platform,
    country,
    trials,
    paid,
    revenues,
    t2p,
    SUM(paid) OVER (PARTITION BY network, platform, country ORDER BY `date` DESC ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) as rolling_paid,
    SUM(trials) OVER (PARTITION BY network, platform, country ORDER BY `date` DESC ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) as rolling_trials,
    AVG(NULLIF(t2p, 0.0)) OVER (
      PARTITION BY network, platform, country
      ORDER BY `date` DESC
      ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
    ) AS t2p_rolling_rate,
    (SUM(paid) OVER (
      PARTITION BY network, platform, country
      ORDER BY `date` DESC
      ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
    )) / NULLIF(SUM(trials) OVER (
      PARTITION BY network, platform, country
      ORDER BY `date` DESC
      ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
    ), 0) AS t2p_rolling_sum
  FROM default_network_platform_country
), t2p_rolling as (
  select
    npc.date, npc.network, npc.platform, npc.country,
    npc.rolling_paid as paid_npc,
    pc.rolling_paid as paid_pc,
    nc.rolling_paid as paid_nc,
    dc.rolling_paid as paid_dc,
    ---
    npc.t2p_rolling_sum as t2p_npc,
    pc.t2p_rolling as t2p_pc,
    nc.t2p_rolling as t2p_nc,
    dc.t2p_rolling as t2p_dc,
    ---
    case
      when npc.rolling_paid > MIN_PAID_CONVERSION then npc.t2p_rolling_sum
      when nc.rolling_paid > MIN_PAID_CONVERSION then nc.t2p_rolling
      when pc.rolling_paid > MIN_PAID_CONVERSION then pc.t2p_rolling
      when dc.rolling_paid > MIN_PAID_CONVERSION then dc.t2p_rolling
      else default_t2p_rate
    end as t2p_estimation
  from default_network_platform_country_rolling npc
  left join default_platform_country_rolling pc
    on npc.date = pc.date and npc.platform = pc.platform and npc.country = pc.country
  left join default_network_country_rolling nc
    on npc.date = nc.date and npc.network = nc.network and npc.country = nc.country
  left join default_country_rolling dc
    on npc.date = dc.date and npc.country = dc.country
)

SELECT *
FROM t2p_rolling
WHERE
  network IN ('Facebook Ads', 'googleadwords_int', 'Apple Search Ads')
  AND country = 'CA'
  AND `date` >= '2025-08-01'
  AND `date` < '2025-09-01'
ORDER BY platform, country, network, `date`;
