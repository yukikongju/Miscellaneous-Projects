declare start_date date default '2025-06-01';
declare end_date date default '2025-10-01';
declare networks default array<string> ['Apple Search Ads', 'Facebook Ads'];
declare platforms default array<string> ['ios', 'android'];
declare countries default array<string> ['US', 'CA', 'FR', 'MX', 'AU', 'UK', 'GB'];
declare MIN_TRIALS default 5;
declare MIN_PAID default 2;

with final_table as (
  select
    date, network, platform, country
    , sum(trials) as trials
    , sum(paid) as paid
    , safe_divide(sum(paid), sum(trials)) as t2p
  from `ua_dashboard_dev.final_table_mobile`
  group by date, network, platform, country
), t2p_internal as (
  select
    install_date, network, platform, country
    , trial
    , paid
    , modeled_trial2paid as t2p
  from `ua_transform_dev.trial2paid_model_unique`
), t2p_geobydate as (
  select
    install_date, network, platform, country
    , trial
    , paid
    , modeled_trial2paid as t2p
  from `ua_transform_dev.trial2paid_geobydate_model_unique`
), joined_table as (
  select
    f.date, f.network, f.platform, f.country
    , f.trials
    , f.paid
    , f.t2p as t2p_actual
    , i.t2p as t2p_internal
    , g.t2p as t2p_geobydate
  from final_table f
  left join t2p_internal i
    on f.date = i.install_date
      and f.network = i.network
      and f.platform = i.platform
      and f.country = i.country
  left join t2p_geobydate g
    on f.date = g.install_date
      and f.network = g.network
      and f.platform = g.platform
      and f.country = g.country
), accuracy_table as (
  select
    network, platform, country
    , avg(safe_divide((t2p_geobydate - t2p_actual), t2p_actual)) as geobydate_mse
    , avg(safe_divide((t2p_internal - t2p_actual), t2p_actual)) as internal_mse
  from joined_table
  where
    date >= start_date and date <= end_date
    and trials > MIN_TRIALS
    and paid > MIN_PAID
  group by network, platform, country
)

-- ==== TODO: CORRELATION ====


-- ==== ACCURACY TABLE ====
select
  *
from accuracy_table
where
  -- date >= start_date and date <= end_date
  network in unnest(networks)
  and platform in unnest(platforms)
  and country in unnest(countries)
order by network, platform, country

-- ==== JOINED_TABLE ====
-- select * from joined_table
-- where
--   date >= start_date and date <= end_date
--   and network in unnest(networks)
--   and platform in unnest(platforms)
--   and country in unnest(countries)
-- order by network, platform, country, date asc
