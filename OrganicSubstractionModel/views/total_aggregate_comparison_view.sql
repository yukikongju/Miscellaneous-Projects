create or replace view `relax-melodies-android.organics.total_aggregate_comparison` as (

with appsflyer_aggregate as (
    select
        *
    from `relax-melodies-android.ua_dashboard_prod.pre_final_view`
    where
        network = 'Appsflyer Aggregate'
        and platform in ('ios', 'android')
), attributed_networks as (
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
        and platform in ('ios', 'android')
    group by date, platform, country
), organic_estimation as (
    select
        *
    from `relax-melodies-android.organics.organic_estimation_all`
), attributed_networks_and_organics as (
    select
        n.date, n.platform, n.country,
        n.cost_cad + o.cost_cad as cost_cad,
        n.cost_usd + o.cost_usd as cost_usd,
        n.clicks + o.clicks as clicks,
        n.impressions + o.impressions as impressions,
        n.installs + o.installs as installs,
        n.mobile_trials + o.mobile_trials as mobile_trials,
        n.web_trials + o.web_trials as web_trials,
        n.trials + o.trials as trials,
        n.paid + o.paid as paid,
        n.revenues + o.revenues as revenues,
        SAFE_DIVIDE(o.installs, n.installs + o.installs) as organic_perc
    from attributed_networks n
    left join organic_estimation o
        on n.date = o.date
       and n.platform = o.platform
       and n.country = o.country
), comparison as (
    select
        t.date, t.platform, t.country,
        t.organic_perc,
        aa.installs as aggregate_installs,
        t.installs as total_installs,
        SAFE_DIVIDE(aa.installs - t.installs, aa.installs) as perc_diff_installs,
        aa.trials as aggregate_trials,
        t.trials as total_trials,
        SAFE_DIVIDE(aa.trials - t.trials, aa.trials) as perc_diff_trials,
        aa.paid as aggregate_paid,
        t.paid as total_paid,
        SAFE_DIVIDE(aa.paid - t.paid, aa.paid) as perc_diff_paid,
        aa.revenues as aggregate_revenues,
        t.revenues as total_revenues,
        SAFE_DIVIDE(aa.revenues - t.revenues, aa.revenues) as perc_diff_revenues
    from attributed_networks_and_organics as t
    left join appsflyer_aggregate as aa
        on t.date = aa.date
       and t.platform = aa.platform
       and t.country = aa.country
)

select
    *
from comparison

);
