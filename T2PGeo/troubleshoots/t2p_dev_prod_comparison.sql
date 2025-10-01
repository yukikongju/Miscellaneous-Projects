--  This query compare the t2p rates in dev (which use the model selection method) and prod (which only use the trial2paid table using internal data)


with t2p_dev as (
  select
    install_date, network, platform, country, trial, paid, modeled_trial2paid
  from `relax-melodies-android.ua_transform_dev.trial2paid_selected_model`
), t2p_prod as (
  select
    install_date, network, platform, country, trial, paid, modeled_trial2paid
  --  from `relax-melodies-android.ua_transform_prod.trial2paid_model_unique`
  from `relax-melodies-android.ua_transform_prod.trial2paid_unique`
), comparison as (
  select
    coalesce(dev.install_date, prod.install_date) as install_date,
    coalesce(dev.platform, prod.platform) as platform,
    coalesce(dev.network, prod.network) as network,
    coalesce(dev.country, prod.country) as country,
    dev.modeled_trial2paid as t2p_dev,
    prod.modeled_trial2paid as t2p_prod,
    safe_divide((prod.modeled_trial2paid - dev.modeled_trial2paid), dev.modeled_trial2paid) as t2p_diff,
  from t2p_dev as dev
  join t2p_prod as prod
    on dev.install_date = prod.install_date
      and dev.network = prod.network
      and dev.platform = prod.platform
      and dev.country = prod.country
)

select
  *
from comparison
where
  install_date >= '2025-08-01' and install_date <= '2025-09-01'
  and country = 'US'
  and network = 'Facebook Ads'
order by network, platform, install_date
