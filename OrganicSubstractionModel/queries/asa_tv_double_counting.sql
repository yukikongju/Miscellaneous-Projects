-- cost:

declare start_date date default '2025-08-01';
declare end_date date default '2025-08-10';

--- iOS: utm_source = 'Apple' and old_hau in ('appstore', 'websearch', 'tvstreaming')
--- android: utm_source = 'google' and old_hau in ('playstore', 'websearch') ;

with counts as (
  select
    extract(year from hau_date) as year,
    extract(month from hau_date) as month,
    extract(week from hau_date) as week,
    platform,
    COUNTIF(utm_source = 'Apple' AND old_hau = 'tvstreaming') as double_count,
    countif(utm_source = 'Apple') as attribution_count,
  from `relax-melodies-android.late_conversions.users_network_attribution`
  where
    platform = 'ios'
    and hau_date >= start_date and hau_date <= end_date
    and hau is not null
    and utm_source is not null
  group by
    extract(year from hau_date), extract(month from hau_date),
    extract(week from hau_date), platform
)

select
  platform,
  year,
  month,
  week,
  double_count,
  attribution_count,
  case when attribution_count > 0
    then double_count / attribution_count
    else null
  end as double_counting_perc,
from counts
order by platform, year, month
