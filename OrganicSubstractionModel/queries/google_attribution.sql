declare start_date date default '2025-07-01';
declare end_date date default '2025-09-01';

---
select
  utm_source, country_code,
  countif(campaign = 'googleadwords_int') as google_attr,
  --  countif(campaign != 'googleadwords_int') as organic_attr,
  countif(campaign = 'Organic') as organic_attr,
from `relax-melodies-android.late_conversions.users_network_attribution`
where
    hau_date >= start_date and hau_date < end_date
    and utm_source like '%google%' --- google ; google-play
group by utm_source, country_code
order by utm_source, country_code;



---
declare start_date date default '2025-07-01';
declare end_date date default '2025-09-01';

with counts as (
  select
    -- extract(isoweek from hau_date) as start_week,
    date_trunc(hau_date, ISOWEEK) as start_week,
    country_code, platform,
    count(*) as total_attr,
    countif(campaign = 'googleadwords_int') as google_attr,
    --  countif(campaign != 'googleadwords_int') as organic_attr,
    countif(campaign = 'Organic') as organic_attr,
  from `relax-melodies-android.late_conversions.users_network_attribution`
  where
      hau_date >= start_date and hau_date < end_date
      and utm_source like '%google%' --- google ; google-play
  group by start_week, country_code, platform
), double_counts as (
  select
    start_week,
    country_code,
    platform,
    total_attr,
    google_attr,
    organic_attr,
    organic_attr / NULLIF(google_attr + organic_attr, 0) as double_count_organic_perc,
    organic_attr / NULLIF(total_attr, 0) as double_count_total_perc,
  from counts
)

select * from double_counts
order by platform, country_code, start_week;

--- US double counting estimation: 30%


---- ======

select
    date_trunc(hau_date, isoweek) as week_start,
    platform,
    country_code,
    COUNTIF(utm_source = 'google' AND old_hau = 'tvstreaming') as double_count,
    COUNTIF(utm_source = 'google') as attribution_count,
    SAFE_DIVIDE(COUNTIF(utm_source = 'google' AND old_hau = 'tvstreaming'), COUNTIF(utm_source = 'google')) as perc,
from `relax-melodies-android.late_conversions.users_network_attribution`
where
    platform = 'android'
    and hau_date >= start_date and hau_date <= end_date
    and hau is not null
    and utm_source is not null
group by
    date_trunc(hau_date, isoweek), platform, country_code
order by platform, country_code, week_start

--- ESTIMATION US: 22-31%


----
select
  distinct old_utm_source, utm_source, old_hau, campaign
from `relax-melodies-android.late_conversions.users_network_attribution`
where
    hau_date >= start_date and hau_date < end_date
    and utm_source like '%google%' --- google ; google-play
order by utm_source, old_hau, campaign;
