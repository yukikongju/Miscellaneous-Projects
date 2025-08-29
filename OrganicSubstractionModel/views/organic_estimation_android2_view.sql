create or replace view `relax-melodies-android.organics.organic_estimation_android2` as (
with android_double_counts as (
  select
    date_trunc(hau_date, isoweek) as isoweek,
    platform,
    country_code,
    COUNTIF(utm_source = 'google' AND old_hau = 'tvstreaming') as double_count,
    countif(utm_source = 'google') as attribution_count,
  from `relax-melodies-android.late_conversions.users_network_attribution`
  where
    platform = 'android'
    and hau is not null
    and utm_source is not null
  group by
    date_trunc(hau_date, isoweek), platform, country_code
), android_double_counting as (
  select
    isoweek,
    platform,
    country_code,
    double_count,
    attribution_count,
    case when attribution_count > 0
      then double_count / attribution_count
      else 0
    end as double_counting_perc,
  from android_double_counts
), android_af_aggregate as (
  select
    *
  from `relax-melodies-android.ua_dashboard_prod.pre_final_view`
  where
    network = 'Appsflyer Aggregate'
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
    network = 'googleadwords_int'
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
      and af.platform = dc.platform
      and af.country = dc.country_code
  left join android_accounted_networks n
    on af.date = n.date
    and af.platform = n.platform
    and af.country = n.country
)
select
    *
from android_organic_estimation
);
