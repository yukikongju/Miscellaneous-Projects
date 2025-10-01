-- This view estimate the android organics using the substraction method with the Tatari Fix
-- Estimated Organics = Appsflyer geobydate Organic - google double counting
-- google double counting is estimated by computing the weekly ratio of TV and android; and then substracting it to the
-- Notes:
-- a. Appsflyer Geobydate Organic = Oranic + Tatari not tagged with google
-- b. True Organic = Appsflyer Geobydate Organic - (1-tatari_google double counting) * tatari

create or replace view `relax-melodies-android.organics.organic_estimation_android3` as (
with android_double_counts as (
  select
    date_trunc(hau_date, isoweek) as week_start,
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
    date_trunc(hau_date, isoweek)
    , platform, country_code
), android_double_counting as (
  select
    week_start,
    platform,
    country_code,
    double_count,
    attribution_count,
    SAFE_DIVIDE(double_count, attribution_count) as double_counting_perc,
    --  case when attribution_count > 0
    --    then double_count / attribution_count
    --    else null
    --  end as double_counting_perc,
  from android_double_counts
), android_af_organic as (
  select
    *
  from `relax-melodies-android.ua_dashboard_prod.pre_final_view`
  where
    network = 'Organic'
    and platform = 'android'
), android_tatari as (
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
    network in ('tatari_streaming', 'tatari_linear', 'tatari_programmatic')
    and platform = 'android'
  group by date, platform, country
), android_organic_estimation as (
  select
    af.date,
    af.platform,
    af.network,
    af.country,
    0 as cost_cad,
    0 as cost_usd,
    0 as impressions,
    0 as clicks,
    case when af.installs - (1.0 - dc.double_counting_perc) * tatari.installs < 0
      then avg(case when af.installs - (1.0 - dc.double_counting_perc) * tatari.installs > 0 then af.installs - (1.0 - dc.double_counting_perc) * tatari.installs end) over (partition by af.country, af.platform order by af.date rows between 7 preceding and current row)
      else af.installs - (1.0 - dc.double_counting_perc) * tatari.installs end as installs,
    0 as web_trials,
    case when af.mobile_trials - (1.0 - dc.double_counting_perc) * tatari.mobile_trials < 0
      then avg(case when af.mobile_trials - (1.0 - dc.double_counting_perc) * tatari.mobile_trials > 0 then af.mobile_trials - (1.0 - dc.double_counting_perc) * tatari.mobile_trials end) over (partition by af.country, af.platform order by af.date rows between 7 preceding and current row)
      else af.mobile_trials - (1.0 - dc.double_counting_perc) * tatari.mobile_trials end as mobile_trials,
    case when af.trials - (1.0 - dc.double_counting_perc) * tatari.trials < 0
      then avg(case when af.trials - (1.0 - dc.double_counting_perc) * tatari.trials > 0 then af.trials - (1.0 - dc.double_counting_perc) * tatari.trials end) over (partition by af.country, af.platform order by af.date rows between 7 preceding and current row)
      else af.trials - (1.0 - dc.double_counting_perc) * tatari.trials end as trials,
    case when af.paid - (1.0 - dc.double_counting_perc) * tatari.paid < 0
      then avg(case when af.paid - (1.0 - dc.double_counting_perc) * tatari.paid > 0 then af.paid - (1.0 - dc.double_counting_perc) * tatari.paid end) over (partition by af.country, af.platform order by af.date rows between 7 preceding and current row)
      else af.paid - (1.0 - dc.double_counting_perc) * tatari.paid end as paid,
    case when af.revenues - (1.0 - dc.double_counting_perc) * tatari.revenues < 0
      then avg(case when af.revenues - (1.0 - dc.double_counting_perc) * tatari.revenues > 0 then af.revenues - (1.0 - dc.double_counting_perc) * tatari.revenues end) over (partition by af.country, af.platform order by af.date rows between 7 preceding and current row)
      else af.revenues - (1.0 - dc.double_counting_perc) * tatari.revenues end as revenues,
    af.agency,
    af.need_modeling,
    (1.0 - dc.double_counting_perc) as double_counting_perc
  from android_af_organic af
  left join android_tatari tatari
    on af.date = tatari.date
      and af.platform = tatari.platform
      and af.country = tatari.country
  left join android_double_counting dc
    on
      date_trunc(af.date, isoweek) = dc.week_start
      and af.platform = dc.platform
      and af.country = dc.country_code
)

select *
from android_organic_estimation
);
