--- cost: 910.9 MB
declare start_date default date '2025-08-02';
declare end_date default date '2025-08-03';

-- DONE: substract with all known ios networks (ASA, tiktokglobal_int,
-- snapchat_int, Facebook Ads, googleadwords_int, )
--- DONE: remove duplicate by fixing daily with monthly joins
--- FIXME/DEPRECATED: impute negative metrics with rolling average of valid days


with ios_double_counts as (
  select
    date_trunc(hau_date, isoweek) as isoweek,
    --  extract(year from hau_date) as year,
    --  extract(month from hau_date) as month,
    --  extract(week from hau_date) as week,
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
    --  extract(year from hau_date), extract(month from hau_date),
    --  extract(week from hau_date),
    date_trunc(hau_date, isoweek), platform, country_code
), ios_double_counting as (
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
    and network = 'Apple Search Ads'
    and platform = 'ios'
  group by date, network, platform, country
), ios_accounted_networks as (
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
      where network in ('Apple Search Ads', 'Facebook Ads', 'snapchat_int', 'tiktokglobal_int', 'googleadwords_int', 'tatari_linear', 'tatari_streaming')
      group by date, platform, country
), ios_organic_estimation as (
  select
    af.date,
    af.platform,
    af.network,
    af.country,
    0 as cost_cad,
    0 as cost_usd,
    0 as impressions,
    0 as clicks,
    --  af.impressions + dc.double_counting_perc * asa.impressions - n.impressions as impressions,
    --  af.clicks + dc.double_counting_perc * asa.clicks - n.clicks as clicks,
    af.installs + dc.double_counting_perc * asa.installs - n.installs as installs,
    af.mobile_trials + dc.double_counting_perc * asa.mobile_trials - n.mobile_trials as mobile_trials,
    af.web_trials + dc.double_counting_perc * asa.web_trials - n.web_trials as web_trials,
    af.trials + dc.double_counting_perc * asa.trials - n.trials as trials,
    af.paid + dc.double_counting_perc * asa.paid - n.paid as paid,
    af.revenues + dc.double_counting_perc * asa.revenues - n.revenues as revenues,
    af.agency,
    af.need_modeling,
    dc.double_counting_perc as double_counting_perc,
  from ios_af_aggregate af
  left join ios_asa asa
    on af.date = asa.date
      and af.platform = asa.platform
      and af.country = asa.country
  left join ios_double_counting dc
    on
	date_trunc(af.date, isoweek) = dc.isoweek
      --  extract(year from af.date) = dc.year
      --  and extract(month from af.date) = dc.month
      --  and extract(week from DATE_TRUNC(af.date, WEEK)) = dc.week
      and af.platform = dc.platform
      and af.country = dc.country_code
  left join ios_accounted_networks n
    on af.date = n.date
    and af.platform = n.platform
    and af.country = n.country
)

--- 261 vs 261
-- select
--     count(*)
-- from ios_af_aggregate
-- union all (
--     select count(*) from (
--       select distinct date, platform, country
--     from ios_af_aggregate)
-- )

-- select *
-- from ios_double_counting
-- where
--   country_code in ('US', 'CA', 'AU', 'UK')

-- no duplicates
-- select *
-- from ios_af_aggregate
-- where
--   country in ('US', 'CA', 'AU', 'UK')
-- order by country, date asc

--- 173
-- select count(*) from ios_asa

 select
  *
 from ios_organic_estimation
  where
    country in ('US', 'CA', 'AU', 'UK')
