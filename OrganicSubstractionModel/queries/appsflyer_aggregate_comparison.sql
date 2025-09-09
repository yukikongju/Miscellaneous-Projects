--- QUERY: Compare Appsflyer Organic + Attributed Networks vs Appsflyer Aggregate
--- cost: 1.07GB
declare start_date default date '2025-07-02';
declare end_date default date '2025-08-03';

with appsflyer_aggregate as (
  select
    *
  from `relax-melodies-android.ua_dashboard_prod.pre_final_view`
  where
    date >= start_date and date <= end_date
    and network = 'Appsflyer Aggregate'
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
	date >= start_date and date <= end_date
	and network in ('Apple Search Ads', 'Facebook Ads', 'snapchat_int', 'tiktokglobal_int', 'googleadwords_int', 'tatari_linear', 'tatari_streaming')
	and platform in ('ios', 'android')
    group by date, platform, country
), estimated_organic as (
    select *
    from `relax-melodies-android.organics.organic_estimation_all`
    where
	date >= start_date and date <= end_date
	and platform in ('ios', 'android')
	and network = 'Organic'
), appsflyer_organic as (
  select
    *
  from `relax-melodies-android.ua_dashboard_prod.pre_final_view`
  where
    date >= start_date and date <= end_date
    and network = 'Organic'
    and platform in ('ios', 'android')
), comparison as (
	select
		aa.date, aa.platform, aa.country,
		aa.installs as aggregated_installs,
		an.installs as attributed_installs,
		eo.installs as estimated_organics,
		an.installs / NULLIF(aa.installs, 0) as attributed_installs_perc,
		(aa.installs - an.installs - eo.installs) / NULLIF(aa.installs, 0) as perc_diff_installs,
	from appsflyer_aggregate aa
	left join attributed_networks an
	on aa.date = an.date
	and aa.platform = an.platform
	and aa.country = an.country
    left join appsflyer_organic ao
	on aa.date = ao.date
	and aa.platform = ao.platform
	and aa.country = ao.country
    left join estimated_organic eo
	on aa.date = eo.date
	and aa.platform = eo.platform
	and aa.country = eo.country
)

--  ), attributed_networks_and_organics as (
--      select
--          n.date, n.platform, n.country,
--          n.cost_cad + o.cost_cad as cost_cad,
--          n.cost_usd + o.cost_usd as cost_usd,
--          n.clicks + o.clicks as clicks,
--          n.impressions + o.impressions as impressions,
--          n.installs + o.installs as installs,
--          n.mobile_trials + o.mobile_trials as mobile_trials,
--          n.web_trials + o.web_trials as web_trials,
--          n.trials + o.trials as trials,
--          n.paid + o.paid as paid,
--          n.revenues + o.revenues as revenues,
--          o.installs / (n.installs + o.installs) as organic_perc,
--      from attributed_networks n
--      left join appsflyer_organic o
--      on n.date = o.date
--          and n.platform = o.platform
--          and n.country = o.country
--  ), comparison as (
--          select
--                  t.date, t.platform, t.country,
--                  t.organic_perc,
--                  aa.installs as aggregate_installs,
--                  t.installs as total_installs,
--                  case when aa.installs > 0
--                                  then (aa.installs - t.installs) / aa.installs
--                                  else null
--                  end as perc_diff_installs,
--                  aa.trials as aggregate_trials,
--                  t.trials as total_trials,
--                  case when aa.trials > 0
--                                  then (aa.trials - t.trials) / aa.trials
--                                  else null
--                  end as perc_diff_trials,
--                  aa.paid as aggregate_paid,
--                  t.paid as total_paid,
--                  case when aa.paid > 0
--                                  then (aa.paid - t.paid) / aa.paid
--                                  else null
--                  end as perc_diff_paid,
--                  aa.revenues as aggregate_revenues,
--                  t.revenues as total_revenues,
--                  case when aa.revenues > 0
--                                  then (aa.revenues - t.revenues) / aa.revenues
--                                  else null
--                  end as perc_diff_revenues,
--          from attributed_networks_and_organics as t
--          left join appsflyer_aggregate as aa
--                  on t.date = aa.date
--                                  and t.platform = aa.platform
--                                  and t.country = aa.country
--  )

select
	*
from comparison
where
	country in ('US', 'CA', 'UK', 'AU', 'MX', 'DE')
order by country, platform, date
