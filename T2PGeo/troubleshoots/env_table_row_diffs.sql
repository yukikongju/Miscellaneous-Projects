-- This query is performing a table diff between dev and prod. We are looking for the following:
-- 1. is there rows mismatch?
-- 2. is there a difference in count?


-- ====== FINAL TABLE MOBILE =======

--- FIXME: the final table mobile query is discrepent

SELECT
  date, network, platform, campaign_id, campaign_name, cost_usd, clicks, installs, trials
FROM `relax-melodies-android.ua_dashboard_dev.final_table_mobile`
WHERE date >= '2025-09-01'
EXCEPT DISTINCT
SELECT
  date, network, platform, campaign_id, campaign_name, cost_usd, clicks, installs, trials
FROM `relax-melodies-android.ua_dashboard_prod.final_table_mobile`
WHERE date >= '2025-09-01';

select count(*) from `relax-melodies-android.ua_dashboard_dev.final_table_mobile` where date >= '2025-09-01'; -- 78350
select count(*) from `relax-melodies-android.ua_dashboard_prod.final_table_mobile` where date >= '2025-09-01'; -- 71751

-- ====== PREFINAL TABLE ======

-- --- rows in dev, but not in prod
SELECT *
FROM `ua_dashboard_dev.pre_final_view`
WHERE date >= '2025-09-01'
EXCEPT DISTINCT
SELECT *
FROM `ua_dashboard_dev.pre_final_view`
WHERE date >= '2025-09-01';

select count(*) from `ua_dashboard_dev.pre_final_view` where date >= '2025-09-01'; -- 81108
select count(*) from `ua_dashboard_dev.pre_final_view` where date >= '2025-09-01'; -- 81108

-- ======= SELECTED T2P MODEL TABLE ========

select
  install_date, network, platform, country, continent, count(*)
from `relax-melodies-android.ua_transform_dev.trial2paid_selected_model`
where
  install_date >= '2025-09-01'
  and network in ('Apple Search Ads', 'Facebook Ads', 'googleadwords_int', 'snapchat_int', 'tiktokglobal_int', 'tatari_linear')
  and country = 'US'
group by
  install_date, network, platform, country, continent
order by
  network, platform, country, continent, install_date


-- ======= MODEL SELECTION MOBILE TABLE ========

select
  *
from `relax-melodies-android.ua_transform_dev.model_selection_mobile`
where
  network in ('Apple Search Ads', 'Facebook Ads', 'googleadwords_int', 'snapchat_int', 'tiktokglobal_int', 'tatari_linear')
order by
  network, platform


-- ======= FACEBOOK ADS =======

-- --- rows in dev, but not in prod
SELECT *
FROM `ua_extract_dev.facebook_ads_country`
WHERE date >= '2025-09-01'
EXCEPT DISTINCT
SELECT *
FROM `ua_extract_prod.facebook_ads_country`
WHERE date >= '2025-09-01';

-- --- rows in dev, but not in prod
SELECT *
FROM `ua_extract_prod.facebook_ads_country`
WHERE date >= '2025-09-01'
EXCEPT DISTINCT
SELECT *
FROM `ua_extract_dev.facebook_ads_country`
WHERE date >= '2025-09-01';

select count(*) from `ua_extract_dev.facebook_ads_country` where date >= '2025-09-01'; -- 19759
select count(*) from `ua_extract_prod.facebook_ads_country` where date >= '2025-09-01'; -- 19759
