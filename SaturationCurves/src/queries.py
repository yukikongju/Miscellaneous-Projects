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
