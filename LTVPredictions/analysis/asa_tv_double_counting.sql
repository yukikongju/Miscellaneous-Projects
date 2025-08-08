--- utm_source = 'tvstreaming' and traffic_source = 'apple search ads' and network_attribution = 'Apple Search Ads'
--- hau = 'tvstreaming' and traffic_source = 'apple search ads'

------- FORMULA -------

--- Appsflyer Aggregate = sum all
--- Organic = Appsflyer Aggregate - ASA * tv_double_counting + tv
---



declare start_date date default '2025-01-01';
declare end_date date default '2025-06-30';

with counts as (
  select
    extract(year from hau_date) as year,
    extract(month from hau_date) as month,
    extract(week from hau_date) as week,
    platform,
    COUNTIF((utm_source = 'tvstreaming' AND traffic_source = 'apple search ads') OR (hau = 'tvstreaming' AND traffic_source = 'apple search ads')) AS tv_asa_count,
    countif(network_attribution = 'Apple Search Ads') as asa_attribution_count,
    -- count(*) filter where (utm_source = 'tvstreaming' and traffic_source = 'apple search ads') or (hau = 'tvstreaming' and traffic_source = 'apple search ads') as tv_asa,
  from `relax-melodies-android.late_conversions.users_network_attribution`
  where
    platform = 'ios'
    and hau_date >= start_date and hau_date <= end_date
  group by
    extract(year from hau_date), extract(month from hau_date),
    extract(week from hau_date), platform
)

select
  platform,
  year,
  month,
  week,
  tv_asa_count,
  asa_attribution_count,
  case when asa_attribution_count > 0
    then tv_asa_count / asa_attribution_count
    else null
  end as double_counting_perc,
from counts
order by platform, year, month
