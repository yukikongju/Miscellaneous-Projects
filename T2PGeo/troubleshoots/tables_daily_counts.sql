--- Aggregate Number
--  select
--    max(install_date) as max_install_date
--  from `relax-melodies-android.ua_transform_dev.geobydate_aggregate_numbers`
-- 2025-09-05

--- Agggregate Numbers
select
  install_date,
  count(*),
from `relax-melodies-android.ua_transform_dev.geobydate_aggregate_numbers_unique`
where
    install_date >= '2025-09-01'
group by
  install_date
order by install_date

--- Geobydate T2P
select
  install_date,
  count(*),
from `relax-melodies-android.ua_transform_dev.trial2paid_geobydate_model_unique`
where
    install_date >= '2025-09-01'
group by
  install_date
order by install_date
