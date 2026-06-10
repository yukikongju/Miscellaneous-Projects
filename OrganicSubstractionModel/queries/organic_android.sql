--- Organic Estimation using Anthony's substraction method
-- E: Total Paid (Appsflyer Aggregate)
-- A: Total Attributed Network Paid (sum ASA, FB, google, snapchat, tiktok, tatari)
-- D: TV HAU + Search ios (utm=google, asa & hau=tvstreaming)
-- G: TV HAU + Search android (utm=google & hau=tvstreaming)
-- C: TV HAU
-- B: Tatari
-- T: Estimated Organics
-- FORMULA: T = E - ( A - \frac{D+G}{C} * B )

with double_counts as (
  select
    date_trunc(hau_date, isoweek) as isoweek,
    platform,
    country_code,
    COUNTIF(utm_source in ('google') AND old_hau = 'tvstreaming') as double_count,
    countif(old_hau = 'tvstreaming') as attribution_count,
  from `relax-melodies-android.late_conversions.users_network_attribution`
  where
    platform = 'android'
    and hau is not null
    and utm_source is not null
  group by
    date_trunc(hau_date, isoweek), platform, country_code
), double_counting as (
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
  from double_counts
), af_aggregate as (
  select
    *
  from `relax-melodies-android.ua_dashboard_prod.pre_final_view`
  where
    network = 'Appsflyer Aggregate'
    and platform = 'android'
), tatari_aggregate as (
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
    network in ('tatari_streaming', 'tatari_linear', 'tatari_programmatic')
    and platform = 'android'
  group by date, network, platform, country
), accounted_networks as (
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
    network in ('Apple Search Ads', 'Facebook Ads', 'snapchat_int', 'tiktokglobal_int', 'googleadwords_int', 'tatari_linear', 'tatari_streaming', 'tatari_programmatic')
    and platform = 'android'
  group by date, platform, country
), organic_estimation as (
  select
    af.date,
    af.platform,
    'Organic' as network,
    af.country,

    -- Formula: Appsflyer Aggregate - (Total Attributed Paid - Double Counting % * Tatari)
    -- Which simplifies to: AF - Total Paid + (Double Counting % * Tatari)
    -- af.installs - n.installs + dc.double_counting_perc * t.installs as installs,
    -- af.trials - n.trials + dc.double_counting_perc * t.trials as trials,
    -- af.paid - n.paid + dc.double_counting_perc * t.paid as paid,
    -- af.revenues - n.revenues + dc.double_counting_perc * t.revenues as revenues,
    -- dc.double_counting_perc as double_counting_perc,

    --- DONT ADD DOUBLE COUNTING BACK
    -- af.installs - n.installs as installs,
    -- af.trials - n.trials as trials,
    -- af.paid - n.paid as paid,
    -- af.revenues - n.revenues  as revenues,
    -- dc.double_counting_perc as double_counting_perc,

    -- ONLY DOUBLE COUNTING
    dc.double_counting_perc * t.installs as installs,
    dc.double_counting_perc * t.trials as trials,
    dc.double_counting_perc * t.paid as paid,
    dc.double_counting_perc * t.revenues as revenues,
    dc.double_counting_perc as double_counting_perc,


  from af_aggregate af
  left join tatari_aggregate t
    on af.date = t.date
      and af.platform = t.platform
      and af.country = t.country
  left join double_counting dc
    on date_trunc(af.date, isoweek) = dc.isoweek
      and af.platform = dc.platform
      and af.country = dc.country_code
  left join accounted_networks n
    on af.date = n.date
    and af.platform = n.platform
    and af.country = n.country
)

select
  date_trunc(date, month) as month,
  country,
  platform,
  sum(installs) as installs,
  sum(trials) as trials,
  sum(paid) as paid,
  sum(revenues) as revenue
from organic_estimation
where
  date >= '2025-06-01'
  and country in ('US', 'CA')
group by month, platform, country
order by month, platform, country


--- 65-75% double counting
-- select
--   *
-- from double_counting
-- where
--   country_code = 'US'
-- order by isoweek, country_code
