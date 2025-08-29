--- QUERY: organic estimation for android using Appsflyer Aggregate instead
-- of substracting from geobydate 'Organic' directly
--- cost: 910.9 MB
declare start_date default date '2025-08-02';
declare end_date default date '2025-08-03';

-- DONE: substract with all known android networks (ASA, tiktokglobal_int,
-- snapchat_int, Facebook Ads, googleadwords_int, )
--- DONE: remove duplicate by fixing daily with monthly joins
--- DONE: impute negative metrics with rolling average of valid days


with android_double_counts as (
  select
    date_trunc(hau_date, isoweek) as isoweek,
    --  extract(year from hau_date) as year,
    --  extract(month from hau_date) as month,
    --  extract(week from hau_date) as week,
    platform,
    country_code,
    COUNTIF(utm_source = 'google' AND old_hau = 'tvstreaming') as double_count,
    countif(utm_source = 'google') as attribution_count,
  from `relax-melodies-android.late_conversions.users_network_attribution`
  where
    platform = 'android'
    and hau_date >= start_date and hau_date <= end_date
    and hau is not null
    and utm_source is not null
  group by
    --  extract(year from hau_date), extract(month from hau_date),
    --  extract(week from hau_date),
    date_trunc(hau_date, isoweek), platform, country_code
), android_double_counting as (
  select
    isoweek,
    --  year,
    --  month,
    --  week,
    platform,
    country_code,
    double_count,
    attribution_count,
    case when attribution_count > 0
      then double_count / attribution_count
      -- else null
      else 0
    end as double_counting_perc,
  from android_double_counts
), android_af_aggregate as (
  select
    *
  from `relax-melodies-android.ua_dashboard_prod.pre_final_view`
  where
    date >= start_date and date <= end_date
    and network = 'Appsflyer Aggregate'
    and platform = 'android'
), android_google as (
  select
    date, network, platform, country,
    sum(cost_cad) as cost_cad,
    sum(cost_usd) as cost_usd,
    sum(clicks) as clicks,
    sum(impressions) as impressions,
    sum(installs) as installs,
    sum(mobile_trials) as mobile_trials,
    sum(web_trials) as web_trials,
    sum(trials) as trials,
    sum(paid) as paid,
    sum(revenues) as revenues
  from `relax-melodies-android.ua_dashboard_prod.pre_final_view`
  where
    date >= start_date and date <= end_date
    and network = 'googleadwords_int'
    and platform = 'android'
  group by date, network, platform, country
), android_accounted_networks as (
  select
    date, platform, country,
    sum(cost_cad) as cost_cad,
    sum(cost_usd) as cost_usd,
    sum(clicks) as clicks,
    sum(impressions) as impressions,
    sum(installs) as installs,
    sum(mobile_trials) as mobile_trials,
    sum(web_trials) as web_trials,
    sum(trials) as trials,
    sum(paid) as paid,
    sum(revenues) as revenues
      from `relax-melodies-android.ua_dashboard_prod.pre_final_view`
      where
	network in ('Apple Search Ads', 'Facebook Ads', 'snapchat_int', 'tiktokglobal_int', 'googleadwords_int', 'tatari_linear', 'tatari_streaming')
	and platform = 'android'
      group by date, platform, country
), android_organic_estimation as (
  select
    af.date,
    af.platform,
    'Organic' as network,
    af.country,
    0 as cost_cad,
    0 as cost_usd,
    0 as impressions,
    0 as clicks,
    --  af.impressions + dc.double_counting_perc * asa.impressions - n.impressions as impressions,
    --  af.clicks + dc.double_counting_perc * asa.clicks - n.clicks as clicks,
    case
	when af.installs + dc.double_counting_perc * asa.installs - n.installs < 0
	then avg(case when af.installs + dc.double_counting_perc * asa.installs - n.installs > 0
	then af.installs + dc.double_counting_perc * asa.installs - n.installs end)
	over (partition by af.country, af.platform order by af.date rows between 7 preceding and current row)
	else af.installs + dc.double_counting_perc * asa.installs - n.installs
    end as installs,
    case
	when af.mobile_trials + dc.double_counting_perc * asa.mobile_trials - n.mobile_trials < 0
	then avg(case when af.mobile_trials + dc.double_counting_perc * asa.mobile_trials - n.mobile_trials > 0
	then af.mobile_trials + dc.double_counting_perc * asa.mobile_trials - n.mobile_trials end)
	over (partition by af.country, af.platform order by af.date rows between 7 preceding and current row)
	else af.mobile_trials + dc.double_counting_perc * asa.mobile_trials - n.mobile_trials
    end as mobile_trials,
    0 as web_trials,
    case
	when af.trials + dc.double_counting_perc * asa.trials - n.trials < 0
	then avg(case when af.trials + dc.double_counting_perc * asa.trials - n.trials > 0
	then af.trials + dc.double_counting_perc * asa.trials - n.trials end)
	over (partition by af.country, af.platform order by af.date rows between 7 preceding and current row)
	else af.trials + dc.double_counting_perc * asa.trials - n.trials
    end as trials,
    case
	when af.paid + dc.double_counting_perc * asa.paid - n.paid < 0
	then avg(case when af.paid + dc.double_counting_perc * asa.paid - n.paid > 0
	then af.paid + dc.double_counting_perc * asa.paid - n.paid end)
	over (partition by af.country, af.platform order by af.date rows between 7 preceding and current row)
	else af.paid + dc.double_counting_perc * asa.paid - n.paid
    end as paid,
    case
	when af.revenues + dc.double_counting_perc * asa.revenues - n.revenues < 0
	then avg(case when af.revenues + dc.double_counting_perc * asa.revenues - n.revenues > 0
	then af.revenues + dc.double_counting_perc * asa.revenues - n.revenues end)
	over (partition by af.country, af.platform order by af.date rows between 7 preceding and current row)
	else af.revenues + dc.double_counting_perc * asa.revenues - n.revenues
    end as revenues,
    --  af.installs + dc.double_counting_perc * asa.installs - n.installs as installs,
    --  af.mobile_trials + dc.double_counting_perc * asa.mobile_trials - n.mobile_trials as mobile_trials,
    --  af.web_trials + dc.double_counting_perc * asa.web_trials - n.web_trials as web_trials,
    --  af.trials + dc.double_counting_perc * asa.trials - n.trials as trials,
    --  af.paid + dc.double_counting_perc * asa.paid - n.paid as paid,
    --  af.revenues + dc.double_counting_perc * asa.revenues - n.revenues as revenues,
    af.agency,
    af.need_modeling,
    dc.double_counting_perc as double_counting_perc,
  from android_af_aggregate af
  left join android_google asa
    on af.date = asa.date
      and af.platform = asa.platform
      and af.country = asa.country
  left join android_double_counting dc
    on
	date_trunc(af.date, isoweek) = dc.isoweek
      --  extract(year from af.date) = dc.year
      --  and extract(month from af.date) = dc.month
      --  and extract(week from DATE_TRUNC(af.date, WEEK)) = dc.week
      and af.platform = dc.platform
      and af.country = dc.country_code
  left join android_accounted_networks n
    on af.date = n.date
    and af.platform = n.platform
    and af.country = n.country
)

--- 261 vs 261
-- select
--     count(*)
-- from android_af_aggregate
-- union all (
--     select count(*) from (
--       select distinct date, platform, country
--     from android_af_aggregate)
-- )

-- select *
-- from android_double_counting
-- where
--   country_code in ('US', 'CA', 'AU', 'UK')

-- no duplicates
-- select *
-- from android_af_aggregate
-- where
--   country in ('US', 'CA', 'AU', 'UK')
-- order by country, date asc

--- 173
-- select count(*) from android_google

select
    *
from android_organic_estimation
where
    country in ('US', 'CA', 'AU', 'UK')
order by country, date asc
