with keys as (
  select
    month_start,
    platform,
    -- n.network,
    network,
    c.country_code
  from UNNEST(GENERATE_DATE_ARRAY('2025-01-01', '2025-01-01', INTERVAL 1 MONTH)) AS month_start
  cross join unnest(['ios', 'android']) as platform
  -- cross join (select distinct network from `relax-melodies-android.ua_transform_prod.spend`) as n
  cross join unnest(['Facebook Ads', 'tiktokglobal_int', 'snapchat_int', 'googleadwords_int',
    'tatari_streaming', 'tatari_linear', 'Apple Search Ads', 'Organic']) as network
  cross join (select country_code from `relax-melodies-android.mappings.country_maps`) as c
)

select * from keys
