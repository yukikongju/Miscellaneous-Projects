"""SQL query templates consumed by the Streamlit application."""

monthly_spend_conversions_query = """
with base_conversions as (
  select
    case
      when network in ('tatari_streaming', 'tatari_linear') then 'tatari'
      else network
    end as network, --- putting tatari_streamling and tatari_linear per Kevin's request
    platform,
    case when country = 'US' then 'US' else 'ROW' end as country,
    extract(year from date) as year,
    extract(month from date) as month
    , sum(cost_usd) as spend
    , sum(impressions) as impressions
    , sum(clicks) as clicks
    , sum(installs) as installs
    , sum(trials) as trials
    , sum(case
      when need_modeling=True then modeled_paid
      else paid
      end) as paid
    , sum(case
      when need_modeling=True then modeled_revenue
      else revenue
      end) as revenue
  from `relax-melodies-android.ua_dashboard_prod.final_table`
  where
    date >= '2023-01-01' and date <= date_sub(current_date(), interval 8 day)
    and network in ('Apple Search Ads', 'Facebook Ads', 'tiktokglobal_int', 'googleadwords_int', 'tatari_linear', 'tatari_streaming', 'snapchat_int')
  group by
    network, platform, country, year, month
)

select * from base_conversions
order by network, platform, country, year, month

"""


weekly_conversions_query = """
# --- 250.99 MB
with base_conversions as (
  select
    case
      when network in ('tatari_streaming', 'tatari_linear') then 'tatari'
      else network
    end as network, --- putting tatari_streamling and tatari_linear per Kevin's request
    platform,
    case when country = 'US' then 'US' else 'ROW' end as country,
    extract(year from date) as year,
    extract(month from date) as month,
    extract(isoweek from date) as isoweek
    , sum(cost_usd) as spend
    , sum(impressions) as impressions
    , sum(clicks) as clicks
    , sum(installs) as installs
    , sum(trials) as trials
    , sum(case
      when need_modeling=True then modeled_paid
      else paid
      end) as paid
    , sum(case
      when need_modeling=True then modeled_revenue
      else revenue
      end) as revenue
  from `relax-melodies-android.ua_dashboard_prod.final_table`
  where
    date >= '2023-01-01' and date <= date_sub(current_date(), interval 8 day)
    and network in ('Apple Search Ads', 'Facebook Ads', 'tiktokglobal_int', 'googleadwords_int', 'tatari_linear', 'tatari_streaming', 'snapchat_int')
    -- and network in ('Apple Search Ads', 'Facebook Ads', 'tiktokglobal_int', 'googleadwords_int', 'tatari', 'snapchat_int')
  group by
    network, platform, country, year, month, isoweek
)

select * from base_conversions
order by network, platform, country, year, month, isoweek
"""

weekly_spend_overall_query = """
with base_conversions as (
  select
    extract(year from date) as year,
    extract(month from date) as month,
    extract(isoweek from date) as isoweek
    , sum(cost_usd) as spend
  from `relax-melodies-android.ua_dashboard_prod.final_table`
  where
    date >= date_sub(current_date(), interval 60 day) and date <= date_sub(current_date(), interval 8 day)
    and network in ('Apple Search Ads', 'Facebook Ads', 'tiktokglobal_int', 'googleadwords_int', 'tatari_linear', 'tatari_streaming')
  group by
  year, month, isoweek
)

select * from base_conversions
order by year, month, isoweek

"""

organics_monthly_query = """
declare start_date date default '2023-01-01';
-- declare end_date date default '2023-01-01';

with paid_spend as (
  select
    extract(year from date) as year,
    extract(month from date) as month,
    sum(cost_usd) as spend,
  from `relax-melodies-android.ua_dashboard_prod.final_table`
  where
    date >= start_date and date <= date_sub(current_date(), interval 10 day)
  group by year, month
), organics_conversions as (
  select
    extract(year from date) as year,
    extract(month from date) as month,
    sum(installs) as installs,
    sum(trials) as trials,
    sum(paid) as paid,
    sum(revenues) as revenue,
  from `relax-melodies-android.ua_organics_prod.organics_substraction_overall`
  where
      date >= start_date and date <= date_sub(current_date(), interval 10 day)
  group by year, month

), joined as (
  select
    o.*,
    p.spend,
  from organics_conversions as o
  left join paid_spend as p
  on
    o.year = p.year
    and o.month = p.month
)

select
  year, month,
  'Organics' as network,
  spend, installs, trials, paid, revenue
from joined
order by year, month
"""

daily_t2p_comparison = """
declare mature_days int64 default 8;
declare window_start date default '2026-02-01';
declare window_end date default date_sub(current_date(), interval mature_days day);

with metrics_selected as (
  select
    date, network, platform, country
    , sum(cost_usd) as spend
    , sum(impressions) as impressions
    , sum(clicks) as clicks
    , sum(installs) as installs
    , sum(trials) as trials
  from `relax-melodies-android.ua_dashboard_prod.ua_source_resolution`
  where
    date >= window_start and date <= window_end
  group by date, network, platform, country
), t2p_prefinal as (
  select
    date, network, platform, country,
    sum(trials) as trials,
    sum(paid) as paid,
    sum(revenues) as revenue,
    safe_divide(sum(paid), sum(trials)) as t2p,
    safe_divide(sum(revenues), sum(paid)) as rev_per_paid,
  from `relax-melodies-android.ua_dashboard_prod.ua_source_resolution`
  where
    date >= window_start and date <= window_end
  group by date, network, platform, country
), t2p_backend as (
  select
    install_date, network, platform, country,
    sum(trial) as trials,
    sum(paid) as paid,
    sum(revenue) as revenue,
    safe_divide(sum(paid), sum(trial)) as t2p,
    safe_divide(sum(revenue), sum(paid)) as rev_per_paid,
  from `relax-melodies-android.ua_transform_prod.trial_and_paid_hau_utm_internal_aggregate`
  where
    install_date >= window_start and install_date <= window_end
  group by install_date, network, platform, country
) , daily_backend_modeled as (
  select
    m.date, m.network, m.platform, m.country
    , m.spend, m.impressions, m.clicks, m.installs, m.trials
    , m.trials * p.t2p as paid
    , m.trials * p.t2p * p.rev_per_paid as revenue
    , p.t2p as t2p
    , p.rev_per_paid as rev_per_paid
  from metrics_selected m
  left join t2p_backend p
  on
    m.date = p.install_date
    and m.network = p.network
    and m.platform = p.platform
    and m.country = p.country
), daily_prefinal_modeled as (
  select
    m.date, m.network, m.platform, m.country
    , m.spend, m.impressions, m.clicks, m.installs, m.trials
    , m.trials * p.t2p as paid
    , m.trials * p.t2p * p.rev_per_paid as revenue
    , p.t2p as t2p
    , p.rev_per_paid as rev_per_paid
  from metrics_selected m
  left join t2p_prefinal p
  on
    m.date = p.date
    and m.network = p.network
    and m.platform = p.platform
    and m.country = p.country
), final_table_daily as (
  select
    date, network, platform, country
    , sum(cost_usd) as spend
    , sum(impressions) as impressions
    , sum(clicks) as clicks
    , sum(installs) as installs
    , sum(trials) as trials
    , case when max(need_modeling) or date >= date_sub(current_date(), interval mature_days day) then sum(modeled_paid)
    else sum(paid) end as paid
    , case when max(need_modeling) or date >= date_sub(current_date(), interval mature_days day) then sum(modeled_revenue)
    else sum(revenue) end as revenue
    , avg(modeled_trial2paid) as t2p
    , avg(modeled_revenue_per_paid) as rev_per_paid
  from `relax-melodies-android.ua_dashboard_prod.final_table`
  where
    date >= window_start and date <= window_end
  group by date, network, platform, country
), combined as (
  select *, 'current' as source
  from final_table_daily
  union all (
    select *, 'backend_daily' as source
    from daily_backend_modeled
  )
  union all (
    select *, 'partners_daily' as source
    from daily_prefinal_modeled
  )
)

select * from combined
where
 -- country = 'US'
  network in ('Apple Search Ads', 'Facebook Ads', 'googleadwords_int', 'tiktokglobal_int', 'snapchat_int', 'tatari_streaming', 'tatari_linear')
order by
network, platform, country, date, source

"""
