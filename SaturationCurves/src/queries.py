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

yearly_final_table_data = """
--- query cost: 221.87 MB
select
  date,
  network,
  platform,
  country,
  campaign_name,
  cost_cad,
  cost_usd,
  impressions,
  clicks,
  installs,
  mobile_trials,
  web_trials,
  trials,
  metrics_paid,
  backend_trials,
  backend_paid,
  backend_revenue,
  backend_refund,
  backend_refunded_amount,
  agency,
  initial_revenue,
  need_modeling,
  modeled_trial2paid,
  t2p_web,
  t2p_mobile,
  rpp_mobile,
  modeled_revenue_per_trial,
  modeled_revenue_per_paid,
  modeled_paid,
  modeled_revenue,
  modeled_refund,
  modeled_refunded_amount,
  recent_date,
  paid,
  page_landing,
  revenue,
  diff_revenues,
  refund,
  refunded_amount,
  renewal_rate_year1,
  renewal_rate_year2,
  renewal_rate_year3,
  renewal_proceeds_year1,
  renewal_proceeds_year2,
  renewal_proceeds_year3,
from `relax-melodies-android.ua_dashboard_prod.final_table`
where
  date >= date_sub(current_date("America/Los_Angeles"), interval 12 MONTH)
  and network in ('Apple Search Ads', 'Facebook Ads', 'googleadwords_int', 'tiktokglobal_int', 'snapchat_int', 'tatari_streaming', 'tatari_linear')
"""

yearly_final_table_data_aggregated = """
SELECT
  date,
  CASE
    WHEN network IN ('tatari_streaming', 'tatari_linear') THEN 'tatari'
    ELSE network
  END AS network,
  platform,
  CASE WHEN country = 'US' THEN 'US' ELSE 'ROW' END AS country,
  campaign_name,
  SUM(cost_usd)                                                          AS cost_usd,
  SUM(impressions)                                                        AS impressions,
  SUM(clicks)                                                             AS clicks,
  SUM(installs)                                                           AS installs,
  SUM(trials)                                                             AS trials,
  SUM(CASE WHEN need_modeling THEN modeled_paid    ELSE paid    END)      AS paid,
  SUM(CASE WHEN need_modeling THEN modeled_revenue ELSE revenue END)      AS revenue
FROM `relax-melodies-android.ua_dashboard_prod.final_table`
WHERE
  date >= DATE_SUB(CURRENT_DATE('America/Los_Angeles'), INTERVAL 12 MONTH)
  AND network IN (
    'Apple Search Ads', 'Facebook Ads', 'tiktokglobal_int',
    'googleadwords_int', 'tatari_linear', 'tatari_streaming', 'snapchat_int'
  )
GROUP BY
  date, network, platform, country, campaign_name
"""

adset_master_api_query = """
--- cost: 411.72 MB
declare start_date date default date_sub(current_date('UTC'), interval 90 day);
declare end_date date default current_date('UTC');

with networks_adset_aggregation as (
    select
        install_date, app_id, pid, geo, c, af_adset as ad, extracted_timestamp,
        sum(cost) as cost,
        sum(impressions) as impressions,
        sum(clicks) as clicks,
        sum(installs) as installs,
        sum(unique_users_subscription_process_succeed) as subscription_process_succeed,
        sum(unique_users_af_subscribe) as af_subscribe_unique_users,
        sum(sales_in_usd_af_subscribe) as af_subscribe_sales_in_usd,
        sum(unique_users_af_refund) as af_refund_unique_users,
        sum(sales_in_usd_af_refund) as af_refund_sales_in_usd,
    FROM `relax-melodies-android.ua_extract_prod.appsflyer_master_complete`
    WHERE
        and pid in ('Apple Search Ads', 'googleadwords_int')
        install_date between start_date and end_date
    GROUP BY
        install_date, app_id, pid, geo, c, af_adset, extracted_timestamp
), networks_ad_aggregation as (
    select
        install_date, app_id, pid, geo, c, af_ad as ad, extracted_timestamp,
        sum(cost) as cost,
        sum(impressions) as impressions,
        sum(clicks) as clicks,
        sum(installs) as installs,
        sum(unique_users_subscription_process_succeed) as subscription_process_succeed,
        sum(unique_users_af_subscribe) as af_subscribe_unique_users,
        sum(sales_in_usd_af_subscribe) as af_subscribe_sales_in_usd,
        sum(unique_users_af_refund) as af_refund_unique_users,
        sum(sales_in_usd_af_refund) as af_refund_sales_in_usd,
    FROM `relax-melodies-android.ua_extract_prod.appsflyer_master_complete`
    WHERE
        install_date between start_date and end_date
        and pid in ('Facebook Ads', 'tiktokglobal_int')
    GROUP BY
        install_date, app_id, pid, geo, c, af_ad, extracted_timestamp
), grouped as (
    SELECT
      *
    from networks_adset_aggregation
    UNION ALL (
        SELECT * from networks_ad_aggregation
    )
), renamed as (
    SELECT
        install_date,
        pid as network,
        case
            when app_id = 'ipnossoft.rma.free' then 'android'
            when app_id = 'id314498713' then 'ios'
            else 'unknown'
        end as platform,
        geo as country,
        c as campaign,
        ad,
        extracted_timestamp,
        cost, impressions, clicks, installs,
        subscription_process_succeed,
        af_subscribe_unique_users, af_subscribe_sales_in_usd,
        af_refund_unique_users, af_refund_sales_in_usd
    FROM grouped
), deduped as (
    SELECT
        * except(extracted_timestamp)
        , ROW_NUMBER() OVER (
            PARTITION BY install_date, network, platform, country, campaign, ad
            ORDER BY extracted_timestamp DESC
        ) AS row_num
    FROM renamed
    WHERE
        cost > 0
    QUALIFY row_num = 1
    ORDER BY
    install_date, network, platform, country, campaign, ad
)

select * from deduped

"""

query_create_wellhub_record = """
DECLARE start_date DATE DEFAULT DATE_SUB(CURRENT_DATE('UTC'), INTERVAL 120 DAY);

SELECT
  date, user_id, country_code, count
FROM `relax-melodies-android.ua_mobile_staging_prod.stg_create_wellhub_record`
WHERE date >= start_date
ORDER BY date, user_id
"""
