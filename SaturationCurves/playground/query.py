weekly_conversions_query = """
# --- 250.99 MB
with base_conversions as (
  select
    network, platform, country,
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
