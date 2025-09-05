-- cost: 200 MB
with geobydate as (
  select
    date, media_source_pid as network, platform, country,
    sum(subscription_process_succeed_unique_users) as trials,
    sum(af_subscribe_unique_users) as paid,
    sum(af_subscribe_sales_in_usd) as revenue,
    sum(subscription_entitlement_paid_unique_users) as entitlement_paid,
    sum(subscription_entitlement_paid_sales_in_usd) as entitlement_revenue,
    sum(subscription_partial_refund_unique_users) as partial_refund,
    sum(subscription_partial_refund_sales_in_usd) as partial_refunded_amount,
    sum(subscription_refund_unique_users) as refund,
    sum(subscription_refund_sales_in_usd) as refunded_amount,
  from `relax-melodies-android.ua_extract_prod.appsflyer_geo_by_date_report`
  group by date, media_source_pid, platform, country
)

select * from geobydate
where
  country = 'US'
  and network in ('Facebook Ads', 'Apple Search Ads', 'tiktokglobal_int')
  -- and date >= '2025-08-01' and date < '2025-09-01'
  -- and geobydate.entitlement_paid > 0
order by network, platform, country, date
