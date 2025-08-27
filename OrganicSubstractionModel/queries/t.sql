declare start_date default date '2025-08-01';
declare end_date default date '2025-08-01';


with ios_double_counts as (
  select
    extract(year from hau_date) as year,
    extract(month from hau_date) as month,
    extract(week from hau_date) as week,
    platform,
    country_code,
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
    extract(week from hau_date), platform, country_code
), ios_double_counting as (
  select
    year,
    month,
    week,
    platform,
    country_code,
    double_count,
    attribution_count,
    case when attribution_count > 0
      then double_count / attribution_count
      else null
    end as double_counting_perc,
  from ios_double_counts
), ios_af_aggregate as (
  select
    *
  from `relax-melodies-android.ua_dashboard_prod.pre_final_view`
  where
    date >= start_date and date <= end_date
    and network = 'Appsflyer Aggregate'
    and platform = 'ios'
), ios_asa as (
  select
    *
  from `relax-melodies-android.ua_dashboard_prod.pre_final_view`
  where
    date >= start_date and date <= end_date
    and network = 'Apple Search Ads'
    and platform = 'ios'
), ios_organic_estimation as (
  select
    af.date,
    af.platform,
    af.network,
    af.country,
    af.installs - dc.double_counting_perc * asa.installs as installs,
  from ios_af_aggregate af
  left join ios_asa asa
    on af.date = asa.date
      and af.platform = asa.platform
      and af.country = asa.country
  left join ios_double_counting dc
    on extract(year from af.date) = dc.year
      and extract(month from af.date) = dc.month
      and extract(week from DATE_TRUNC(af.date, WEEK)) = dc.week
      and af.platform = dc.platform
      and af.country = dc.country_code
)

select
  *
from ios_organic_estimation
where
  country = 'US'
