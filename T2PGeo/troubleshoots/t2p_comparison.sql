--- cost: 246.47 MB
with geo_t2p as (
  select
    install_date, network, platform, country, trial, paid, modeled_trial2paid
  from `relax-melodies-android.ua_transform_dev.trial2paid_geobydate_model`
), internal_t2p as (
  select
    install_date, network, platform, country, trial, paid, modeled_trial2paid
  from `relax-melodies-android.ua_transform_dev.trial2paid_model_unique`
), selected_t2p as (
  select
    install_date, network, platform, country, trial, paid, modeled_trial2paid
  from `relax-melodies-android.ua_transform_dev.trial2paid_selected_model`
), comparison as (
  select
    coalesce(s.install_date, g.install_date, i.install_date) as install_date,
    coalesce(s.platform, g.platform, i.platform) as platform,
    coalesce(s.network, g.network, i.network) as network,
    coalesce(s.country, g.country, i.country) as country,
    s.modeled_trial2paid as t2p_selected,
    g.modeled_trial2paid as t2p_geobydate,
    i.modeled_trial2paid as t2p_internal,
  from selected_t2p as s
  join geo_t2p as g
    on s.install_date = g.install_date
      and s.network = g.network
      and s.platform = g.platform
      and s.country = g.country
  join internal_t2p as i
    on s.install_date = i.install_date
      and s.network = i.network
      and s.platform = i.platform
      and s.country = i.country
)

select
  *
from comparison
where
  install_date >= '2025-08-01' and install_date <= '2025-09-01'
  and country = 'US'
  and network = 'Facebook Ads'
order by network, platform, install_date
