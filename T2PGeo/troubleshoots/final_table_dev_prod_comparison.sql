--  This query compare the rates impacted by t2p in dev (which use the model selection method) and prod (which only use the trial2paid table using internal data), mainly: t2p, i2p, roas

-- cost: 3.26GB
with final_dev as (
  select
    date, network, platform, country
    , sum(cost_usd) as spend
    , sum(installs) as installs
    , sum(modeled_paid) as modeled_paid
    , sum(modeled_revenue) as modeled_revenue
    --  , sum(trials) as trials
    --  , sum(paid) as paid
    , avg(modeled_trial2paid) as modeled_trial2paid
    , safe_divide(sum(modeled_paid), sum(installs)) as i2p
    , safe_divide(sum(modeled_revenue), sum(cost_usd)) as roas
  from `relax-melodies-android.ua_dashboard_dev.final_table_dev`
  group by
    date, network, platform, country
), final_prod as (
  select
    date, network, platform, country
    , sum(cost_usd) as spend
    , sum(installs) as installs
    , sum(modeled_paid) as modeled_paid
    --  , sum(trials) as trials
    --  , sum(paid) as paid
    , avg(modeled_trial2paid) as modeled_trial2paid
    , safe_divide(sum(modeled_paid), sum(installs)) as i2p
    , safe_divide(sum(modeled_revenue), sum(cost_usd)) as roas
  from `relax-melodies-android.ua_dashboard_prod.final_table`
  group by
    date, network, platform, country
), comparison as (
  select
    coalesce(dev.date, prod.date) as install_date,
    coalesce(dev.platform, prod.platform) as platform,
    coalesce(dev.network, prod.network) as network,
    coalesce(dev.country, prod.country) as country,

    --  dev.spend as spend_dev,
    --  prod.spend as spend_prod,
    --  dev.trials as trials_dev,
    --  prod.trials as trials_prod,
    --  dev.paid as paid_dev,
    --  prod.paid as paid_prod,
    --- T2P diff
    dev.modeled_trial2paid as t2p_dev,
    prod.modeled_trial2paid as t2p_prod,
    safe_divide((prod.modeled_trial2paid - dev.modeled_trial2paid), dev.modeled_trial2paid) as t2p_diff,
    --- I2P diff: i2p
    dev.i2p as i2p_dev,
    prod.i2p as i2p_prod,
    safe_divide((prod.i2p - dev.i2p), dev.i2p) as i2p_diff,
    --- I2P diff: roas
    dev.roas as roas_dev,
    prod.roas as roas_prod,
    safe_divide((prod.roas - dev.roas), dev.roas) as roas_diff,
  from final_dev as dev
  join final_prod as prod
    on dev.date = prod.date
      and dev.network = prod.network
      and dev.platform = prod.platform
      and dev.country = prod.country
)

select
  *
from comparison
where
  install_date >= '2025-09-01' and install_date <= '2025-10-01'
  and country = 'US'
  and network = 'Facebook Ads'
order by network, platform, install_date
